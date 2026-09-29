"""
Plan Executor.

Executes the LLaDA-generated plans through the multi-step reasoning loop
described in the paper. Each plan step is grounded to a real tool call by
the executor LLM, and the table state evolves between steps.

Reads the JSONL output of `diffusion_planner.py` (each record carries a
list of plain-English plan steps under `plan_steps`).

For each record the executor:
  1. Reconstructs the table DataFrame from `table_csv`.
  2. Prepends the LLaDA plan to the question so it appears in every step's
     prompt inside `run_reasoning_loop`.
  3. Calls `run_reasoning_loop`, which executes tool calls for real and
     feeds updated table state back to the model on the next step.
  4. Extracts the `final_answer` and computes exact-match plus a numeric
     ROUGE-style overlap metric against the ground truth.
  5. Saves the augmented record to a new JSONL.

Why an executed reasoning loop rather than a single prompt
----------------------------------------------------------
A single-prompt executor sees the full table but never actually runs the
tool calls -- it hallucinates results. `run_reasoning_loop` runs each tool
for real (filter, group_by, sort, ...), so the model sees true intermediate
state, which is materially more accurate for multi-step numerical reasoning.

Dependencies (helper modules, to be added to this repository)
-------------------------------------------------------------
  - reasoning_loop.run_reasoning_loop : executes a tool-grounded reasoning
                                        trace step by step.
  - model_factory.ModelFactory        : provider-agnostic LLM wrapper.

Both will be released alongside the training pipeline.

Reproduction
------------
  python plan_executor.py \\
    --plans  plans/llada_tabmwp_0_200.jsonl \\
    --output results/executed_tabmwp.jsonl \\
    --provider ollama --model-name qwen3-vl:8b \\
    --max-tokens 8192 --max-steps 8 \\
    --resume

Add `--thinking-budget 512` for Qwen3 thinking mode on harder benchmarks.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Plan injection
# ---------------------------------------------------------------------------

def _render_steps(plan_steps: List[str]) -> str:
    if not plan_steps:
        return "(no plan)"
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan_steps))


def _inject_plan(question: str, plan_steps: List[str]) -> str:
    """
    Append the LLaDA plan to the question so `run_reasoning_loop` sees it at
    every step inside the tool prompt (which embeds `Question: {question}`).
    """
    if not plan_steps:
        return question
    steps_text = _render_steps(plan_steps)
    return (
        f"{question}\n\n"
        "[Reasoning plan -- follow this order, mapping each step to a tool]\n"
        f"{steps_text}"
    )


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s\.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def _extract_numbers(text: str) -> List[float]:
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", str(text))
    values = []
    for m in re.findall(r"-?\d+\.?\d*", cleaned):
        try:
            values.append(float(m))
        except ValueError:
            pass
    return values


def _quant_rouge(pred: str, gold: str) -> float:
    gold_nums = _extract_numbers(gold)
    if gold_nums:
        pred_nums = set(_extract_numbers(pred))
        if not pred_nums:
            return 0.0
        hits = sum(
            1 for g in gold_nums
            if any(abs(p - g) / max(abs(g), 1e-9) < 0.05 for p in pred_nums)
        )
        return hits / len(gold_nums)
    gold_toks = set(_normalize(gold).split())
    pred_toks = set(_normalize(pred).split())
    if not gold_toks:
        return 1.0
    return len(gold_toks & pred_toks) / len(gold_toks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # NOTE: the two imports below are placeholders for helper modules that
    # will be released alongside the training pipeline.
    from reasoning_loop import run_reasoning_loop                        # type: ignore
    from model_factory import ModelFactory                               # type: ignore

    parser = argparse.ArgumentParser(description="Plan Executor (reasoning loop)")

    # Input / output
    parser.add_argument("--plans",  required=True,
                        help="JSONL file from diffusion_planner.py")
    parser.add_argument("--output", required=True,
                        help="JSONL file to write execution results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=None)

    # Executor LLM
    parser.add_argument("--provider", default="ollama",
                        choices=["openai", "ollama", "vllm", "claude", "gemini", "together"])
    parser.add_argument("--model-name",  default="qwen3-vl:8b")
    parser.add_argument("--base-url",    default=None)
    parser.add_argument("--max-tokens",  type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Qwen3 thinking budget tokens (0 = disabled)")

    # Reasoning loop
    parser.add_argument("--max-steps",  type=int, default=6,
                        help="Max tool-call steps per question. The paper uses 6.")
    parser.add_argument("--json-retries", type=int, default=2)
    parser.add_argument("--table-input-mode", default="text",
                        choices=["text", "image", "multimodal"],
                        help="Text mode is fastest and sufficient when the plan "
                             "already names the relevant columns.")

    args = parser.parse_args()

    # Build executor model
    model_config: Dict[str, Any] = {
        "provider":    args.provider,
        "model_name":  args.model_name,
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
    }
    if args.base_url:
        model_config["base_url"] = args.base_url
    if args.thinking_budget > 0:
        model_config["extra_body"] = {
            "chat_template_kwargs": {"thinking_budget": args.thinking_budget}
        }
    model = ModelFactory.create(model_config)
    print(
        f"[plan_executor] executor={args.provider}/{args.model_name}  "
        f"max_steps={args.max_steps}  table_mode={args.table_input_mode}",
        flush=True,
    )

    # Resume support
    done_ids: set = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line).get("table_id", ""))
                except json.JSONDecodeError:
                    pass
        if done_ids:
            print(f"[plan_executor] resuming -- {len(done_ids)} already done", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load plans
    plans: List[Dict] = []
    with open(args.plans, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    plans.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    lo = args.start
    hi = args.end if args.end is not None else len(plans)
    plans = plans[lo:hi]
    print(f"[plan_executor] loaded {len(plans)} plans (indices {lo}-{hi})", flush=True)

    # Metrics
    n_done = n_err = 0
    em_total = 0
    qr_total = 0.0

    with open(args.output, "a", encoding="utf-8") as out_fh:
        for i, record in enumerate(plans):
            table_id     = record.get("table_id", f"row_{lo + i}")
            question     = record.get("question", "")
            ground_truth = record.get("ground_truth", "")
            plan_steps   = record.get("plan_steps", [])
            table_csv    = record.get("table_csv", "")

            if not question:
                continue
            if args.resume and table_id in done_ids:
                print(f"  [skip] {table_id}", flush=True)
                continue

            print(f"  [{lo + i}] {table_id} | {question[:70]}", flush=True)
            if plan_steps:
                print(f"    plan: {len(plan_steps)} steps -- {repr(plan_steps[0][:60])}", flush=True)

            t0 = time.time()
            try:
                df = pd.read_csv(io.StringIO(table_csv))
                question_with_plan = _inject_plan(question, plan_steps)

                final_answer, reasoning_trace, usage = run_reasoning_loop(
                    model=model,
                    df=df.copy(),
                    question=question_with_plan,
                    max_steps=args.max_steps,
                    json_retries=args.json_retries,
                    max_tokens=args.max_tokens,
                    table_input_mode=args.table_input_mode,
                )

                elapsed = time.time() - t0

                em = _exact_match(final_answer, ground_truth)
                qr = _quant_rouge(final_answer, ground_truth)
                em_total += int(em)
                qr_total += qr
                n_done   += 1

                print(
                    f"    -> answer={repr(final_answer[:60])}  "
                    f"gold={repr(str(ground_truth)[:40])}  "
                    f"steps={len(reasoning_trace)}  "
                    f"EM={int(em)}  QR={qr:.2f}  time={elapsed:.1f}s",
                    flush=True,
                )

                out_record = {
                    "table_id":              table_id,
                    "question":              question,
                    "ground_truth":          ground_truth,
                    "dataset":               record.get("dataset", ""),
                    "model_config":          model_config,
                    "pipeline_mode":         "diffusion_plan",
                    "table_csv_initial":     table_csv,
                    "table_columns_initial": record.get("table_columns", []),
                    "table_shape_initial":   record.get("table_shape", []),
                    "final_answer":          final_answer,
                    "reasoning_trace":       reasoning_trace,
                    "usage":                 usage,
                    "elapsed_seconds":       round(elapsed, 2),
                    "collected_at":          datetime.now(timezone.utc).isoformat(),
                    "planner_model":         record.get("planner_model", ""),
                    "planner_cfg":           record.get("planner_cfg", {}),
                    "plan_text":             record.get("plan_text", ""),
                    "plan_steps":            plan_steps,
                }
                out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                out_fh.flush()

            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                n_err += 1

    print(f"\n[plan_executor] done -- {n_done} executed  {n_err} errors")
    if n_done:
        print(f"  Exact-match : {em_total/n_done:.3f}  ({em_total}/{n_done})")
        print(f"  Quant-rouge : {qr_total/n_done:.3f}")
    print(f"  output -> {args.output}")


if __name__ == "__main__":
    main()
