"""
Diffusion Planner.

This script implements the LLaDA-based diffusion planner described in the
paper. The planner observes the full table and question simultaneously
through bidirectional denoising, then emits a numbered natural-language
plan in a single inference call.

The plan is consumed by `plan_executor.py`, which grounds each step to a
tool call and executes the reasoner loop.

Saved fields per record
-----------------------
  plan_steps : list[str] -- parsed numbered steps
  plan_text  : str       -- raw model output
  usage      : dict      -- token counts

Dependencies (helper modules, to be added to this repository)
-------------------------------------------------------------
  - dataset_iterator      : yields (table_id, df, question, ground_truth)
                            tuples per benchmark sample
  - table_serializer      : serialize_dataframe_to_text(df) -> str
  - model_factory         : ModelFactory.create(config) -> generator,
                            with a "llada" provider that wraps the open-weight
                            LLaDA-8B-Instruct release

These three modules will be released alongside the training pipeline and
the 1,600 cell-level attention standards.

Reproduction
------------
  python diffusion_planner.py \\
    --model-name GSAI-ML/LLaDA-8B-Instruct \\
    --llada-device cuda:0 \\
    --max-tokens 256 \\
    --llada-steps 128 \\
    --llada-cfg-scale 1.5 \\
    --llada-block-length 32 \\
    --infotabs-path /path/to/infotabs --infotabs-split dev \\
    --start 0 --end 200 \\
    --output plans/llada_infotabs_dev_0_200.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Planning prompt
# ---------------------------------------------------------------------------

def _build_planning_prompt(serialized_table: str, question: str) -> str:
    return (
        "You are a table reasoning agent. Study the table and question carefully, "
        "then write a precise step-by-step plan to find the answer.\n\n"
        "Available operations:\n"
        "- filter: select rows where a column satisfies a condition\n"
        "- sort: order rows by a column value (ascending or descending)\n"
        "- aggregate: compute sum, count, average, min, or max over a column\n"
        "- lookup: retrieve the value at a specific row and column\n"
        "- compare: compare values across rows or columns\n"
        "- select: extract a subset of columns for further analysis\n\n"
        "Rules:\n"
        "- Number every step starting from 1, one sentence each.\n"
        "- Each step MUST name the operation used (from the list above) and reference "
        "  the exact column names or cell values it targets.\n"
        "- End every step with a [target: col_name] tag (or [target: col_name, row N] "
        "  for a specific cell) indicating which table cells the step operates on.\n"
        "  Example: 'Filter the \"Country\" column to rows where value is \"Algeria\". "
        "  [target: Country]'\n"
        "- Do NOT write generic steps like 'identify relevant information' or "
        "  'analyse the table'.\n"
        "- The last step must state the final answer directly.\n\n"
        f"Table:\n```\n{serialized_table}\n```\n\n"
        f"Question: {question}\n\n"
        "Plan:"
    )


def _build_infotabs_planning_prompt(serialized_table: str, question: str) -> str:
    return (
        "You are a table reasoning agent. Study the table and question carefully, "
        "then write a precise step-by-step plan to find the answer.\n\n"
        "Available operations:\n"
        "- filter: select rows where a column satisfies a condition\n"
        "- sort: order rows by a column value (ascending or descending)\n"
        "- aggregate: compute sum, count, average, min, or max over a column\n"
        "- lookup: retrieve the value at a specific row and column\n"
        "- compare: compare values across rows or columns\n"
        "- select: extract a subset of columns for further analysis\n\n"
        "Rules:\n"
        "- Number every step starting from 1, one sentence each.\n"
        "- Each step MUST name the operation used (from the list above) and reference "
        "  the exact column names or cell values it targets.\n"
        "- End every step with a [target: col_name] tag (or [target: col_name, row N] "
        "  for a specific cell) indicating which table cells the step operates on.\n"
        "  Example: 'Lookup the \"Award\" column for the row where Name is \"X\". "
        "  [target: Award, Name]'\n"
        "- Do NOT write generic steps like 'identify relevant information' or "
        "  'analyse the table'.\n"
        "- The last step must state the final answer directly as one of: "
        "entailment, contradiction, or neutral.\n\n"
        "Label definitions:\n"
        "- entailment: the table provides clear, direct evidence that the statement is true.\n"
        "- contradiction: the table provides clear, direct evidence that the statement is false.\n"
        "- neutral: the table does NOT contain enough information to prove or disprove the "
        "statement. Treat the table as the ONLY source of evidence; do not use outside knowledge.\n\n"
        "Before concluding entailment or contradiction, run these three checks. "
        "If ANY check fails, the answer must be neutral:\n\n"
        "1. QUANTIFIER CHECK. If the statement uses 'only', 'always', 'all', 'every', "
        "'never', 'exclusively', 'no other', 'first', 'last', 'any', or a similar "
        "exhaustive word, a single matching cell is NOT enough. The table must explicitly "
        "enumerate ALL cases to confirm or refute such a claim.\n\n"
        "2. COMPARISON CHECK. If the statement compares two items (A before B, A after B, "
        "A greater/less than B), the table must give an explicit value for BOTH sides on "
        "the compared dimension. Listing order in a cell is NOT evidence of temporal order.\n\n"
        "3. SCOPE CHECK. If the statement makes a broader claim than any single cell can "
        "directly support (e.g. 'lived there his entire life', 'is most known for'), check "
        "whether the exact scope is in the table. When in doubt, output neutral.\n\n"
        f"Table:\n```\n{serialized_table}\n```\n\n"
        f"Question: {question}\n\n"
        "Plan:"
    )


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

def _parse_plan_steps(raw: str) -> List[str]:
    """
    Extract numbered steps from free-form model output.

    Handles "1. Do X" / "1) Do X" / "Step 1: Do X". Falls back to non-empty lines.
    """
    pattern = re.compile(
        r"(?:^|\n)\s*(?:step\s*)?\d+[.):\-]\s*(.+?)(?=\n\s*(?:step\s*)?\d+[.):\-]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    steps = []
    for m in pattern.finditer(raw):
        text = re.sub(r"\s*\n\s*", " ", m.group(1).strip())
        if text:
            steps.append(text)
    if steps:
        return steps
    return [l.strip() for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # NOTE: the three imports below are placeholders for helper modules that
    # will be released alongside the training pipeline.
    #   - dataset_iterator: yields per-sample dicts {table_id, table, question,
    #     ground_truth, dataset} for each supported benchmark
    #   - table_serializer: serialize_dataframe_to_text(df) -> str (Markdown table)
    #   - model_factory: provider="llada" wraps the public LLaDA-8B-Instruct release
    import dataset_iterator                                              # type: ignore
    from table_serializer import serialize_dataframe_to_text             # type: ignore
    from model_factory import ModelFactory                               # type: ignore

    parser = argparse.ArgumentParser(description="LLaDA Diffusion Planner")

    # Model
    parser.add_argument("--model-name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="gen_length for the plan. Must be divisible by --llada-block-length.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy.")
    parser.add_argument("--llada-steps", type=int, default=128,
                        help="Denoising steps. 256-512 gives better plan quality at ~2x the cost.")
    parser.add_argument("--llada-block-length", type=int, default=32,
                        help="Semi-autoregressive block size.")
    parser.add_argument("--llada-cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--llada-device", default="cuda")
    parser.add_argument("--llada-no-4bit", action="store_true")
    parser.add_argument("--llada-max-prompt-tokens", type=int, default=3500)

    # Datasets
    parser.add_argument("--wikitablequestions-path")
    parser.add_argument("--tatqa-path")
    parser.add_argument("--tatqa-split", default="dev",
                        choices=["train", "dev", "test", "test_gold"])
    parser.add_argument("--tatqa-answer-from", default=None,
                        choices=["table", "text", "table-text"])
    parser.add_argument("--fetaqa-path")
    parser.add_argument("--fetaqa-split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--tabmwp-path")
    parser.add_argument("--tabmwp-split", default="test",
                        choices=["train", "dev", "dev1k", "test", "test1k"])
    parser.add_argument("--hitab-path")
    parser.add_argument("--hitab-split", default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--mmqa-path")
    parser.add_argument("--mmtu-path")
    parser.add_argument("--infotabs-path")
    parser.add_argument("--infotabs-split", default="dev",
                        choices=["train", "dev", "test_alpha1", "test_alpha2", "test_alpha3"])
    parser.add_argument("--tabfact-path")
    parser.add_argument("--tabfact-split", default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # Build LLaDA model
    model_config: Dict[str, Any] = {
        "provider":          "llada",
        "model_name":        args.model_name,
        "max_tokens":        args.max_tokens,
        "temperature":       args.temperature,
        "steps":             args.llada_steps,
        "block_length":      args.llada_block_length,
        "cfg_scale":         args.llada_cfg_scale,
        "device":            args.llada_device,
        "load_in_4bit":      not args.llada_no_4bit,
        "max_prompt_tokens": args.llada_max_prompt_tokens,
        "single_turn":       True,
    }
    model = ModelFactory.create(model_config)
    print(
        f"[diffusion_planner] model={args.model_name}  "
        f"max_tokens={args.max_tokens}  steps={args.llada_steps}  "
        f"cfg={args.llada_cfg_scale}  block={args.llada_block_length}  "
        f"temp={args.temperature}",
        flush=True,
    )

    dataset_name = dataset_iterator.dataset_name(args)
    print(f"[diffusion_planner] dataset={dataset_name}", flush=True)

    # Resume support
    done_ids: set = set()
    if args.resume:
        done_ids = dataset_iterator.load_done_ids(args.output)
        if done_ids:
            print(f"[diffusion_planner] resuming -- {len(done_ids)} already done")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_done = n_err = 0

    with open(args.output, "a", encoding="utf-8") as out_fh:
        for sample in dataset_iterator.iter_samples(args):
            table_id     = sample.get("table_id", "")
            question     = sample.get("question", "")
            ground_truth = sample.get("answer", sample.get("ground_truth", ""))
            df           = sample["table"]

            if not question:
                continue
            if args.resume and table_id in done_ids:
                print(f"  [skip] {table_id}", flush=True)
                continue

            print(f"  {table_id} | {question[:70]}", flush=True)
            t0 = time.time()

            try:
                serialized  = serialize_dataframe_to_text(df)
                if args.infotabs_path:
                    prompt_text = _build_infotabs_planning_prompt(serialized, question)
                else:
                    prompt_text = _build_planning_prompt(serialized, question)
                messages    = [{"role": "user", "content": prompt_text}]
                columns     = df.columns.tolist()

                response   = model.generate([messages])[0]
                plan_text  = response.get("text", "").strip()
                plan_steps = _parse_plan_steps(plan_text)
                last_usage = response.get("usage", {})

                elapsed = time.time() - t0
                print(
                    f"    -> {len(plan_steps)} steps  "
                    f"preview={repr(plan_steps[0][:60]) if plan_steps else '(empty)'}  "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

                record = {
                    "table_id":      table_id,
                    "question":      question,
                    "ground_truth":  ground_truth,
                    "dataset":       dataset_name,
                    "planner_model": args.model_name,
                    "planner_cfg": {
                        "steps":        args.llada_steps,
                        "cfg_scale":    args.llada_cfg_scale,
                        "block_length": args.llada_block_length,
                        "temperature":  args.temperature,
                    },
                    "table_csv":     df.to_csv(index=False),
                    "table_columns": columns,
                    "table_shape":   list(df.shape),
                    "plan_text":     plan_text,
                    "plan_steps":    plan_steps,
                    "usage":         last_usage,
                    "elapsed_seconds": round(elapsed, 2),
                    "collected_at":  datetime.now(timezone.utc).isoformat(),
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_done += 1

            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                n_err += 1

    print(f"\n[diffusion_planner] done -- {n_done} plans written  {n_err} errors")
    print(f"  output -> {args.output}")


if __name__ == "__main__":
    main()
