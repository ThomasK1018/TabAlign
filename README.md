# TabAlign

Official code for the NeurIPS 2026 paper **"From Table to Cell: Attention for
Better Reasoning with TabAlign"**.

Tung Sum Thomas Kwok\*, Zeyong Zhang\*, Xinyu Wang, Chunhe Wang, Xiaofeng Lin,
Hanwei Wu, Lei Ding, Guang Cheng, Zhijiang Guo\*

<!-- TODO: add paper link (arXiv / OpenReview) once available -->

TabAlign is a planned table reasoning framework. A masked diffusion language
model (DLM) planner emits cell-grounded plan steps through bidirectional
denoising, and `TabAttn`, a lightweight verifier trained on 1,600
human-verified attention standards, scores each step by its attention overlap
with the plan-designated cells. This repository provides the inference-time
reasoning pipeline that produces the main accuracy results reported in the
paper.

## Repository layout

```
TabAlign/
├── README.md                 # this file
├── LICENSE                   # Apache 2.0
├── requirements.txt          # Python dependencies
├── src/
│   ├── diffusion_planner.py  # LLaDA-8B-Instruct planner
│   └── plan_executor.py      # tool-grounded reasoning loop
└── scripts/
    └── run_pipeline.sh       # end-to-end driver
```

## Release status

The inference-time planner and executor are available now. The following
components are being prepared for release and will be added to this
repository:

1. **The 1,600 human-verified cell-level attention standards** used to
   calibrate `TabAttn`. The curation protocol is fully documented in
   Appendix C of the paper, including the seed-annotation instructions, the
   GPT-5 model-comparison procedure, and the IoU >= 0.70 acceptance
   threshold.
2. **`TabAttn` calibration code**, including the logistic regression target
   and the 80/20 train/early-stop split.
3. **Helper modules** referenced by the planner and executor:
   `dataset_iterator`, `table_serializer`, `model_factory`, and
   `reasoning_loop`. Their expected interfaces are documented in the
   docstrings of `src/diffusion_planner.py` and `src/plan_executor.py`.
4. **End-to-end training scripts** for the `TabAttn` reward.

The eight evaluation benchmarks (WTQ, MMQA, TabMWP, TAT-QA, HiTab, FeTaQA,
TabFact, InfoTabs) are publicly released academic resources and are not
redistributed here. Citations and version notes are listed in Appendix A
of the paper.

## Installation

```bash
git clone https://github.com/ThomasK1018/TabAlign.git
cd TabAlign
pip install -r requirements.txt
```

## Reproduction

The pipeline runs in two stages:

1. **Plan generation** (`diffusion_planner.py`). For each (table, question)
   the LLaDA-8B-Instruct diffusion model emits a numbered natural-language
   plan in a single inference call. Default decoding follows the paper:
   canvas length 256, block length 32, 128 denoising steps, classifier-free
   guidance scale 1.5, greedy planner decoding.
2. **Plan execution** (`plan_executor.py`). The executor LLM (Qwen3-VL-8B
   in the paper) grounds each plan step to a tool call from a fixed
   vocabulary (`filter`, `sort`, `aggregate`, `lookup`, `compare`,
   `select`). Tools are executed for real, and the table state evolves
   between steps. The reasoning loop terminates after at most six tool
   steps (`T_max = 6`) or earlier under the stagnation halt criterion
   (`delta_stag = 0.02`, `K_stag = 2`). Reasoner temperature 0.7,
   maximum 16,384 output tokens.

A reference driver is provided in `scripts/run_pipeline.sh`. Set
`DATASET_NAME` and `DATASET_PATH` to point to a local benchmark copy, then
run the driver. It executes both stages and writes JSONL outputs under
`plans/` and `results/` in the current directory.

```bash
DATASET_NAME=tabmwp DATASET_PATH=/path/to/tabmwp bash scripts/run_pipeline.sh
```

## Hardware

Inference was performed on a single NVIDIA RTX 4090 (24 GB) workstation
GPU. Three model processes share the 24 GB envelope:

1. **Qwen3-VL-8B reasoner** served via Ollama (`qwen3-vl:8b`, GGUF Q4
   quantization) at roughly 7.7 GB VRAM, reached through the Ollama
   OpenAI-compatible local endpoint by `plan_executor.py`.
2. **HF Qwen3-VL-8B-Instruct attention extractor** (used by the `TabAttn`
   calibration loop, not by the inference pipeline in this repository)
   under 4-bit NF4 `bitsandbytes` quantization with double-quant and
   `attn_implementation="eager"`, capped at 10 GiB VRAM.
3. **LLaDA-8B-Instruct planner** loaded as a separate HuggingFace process
   in 4-bit `bitsandbytes` (default `load_in_4bit=True`, pass
   `--llada-no-4bit` to disable). The planner is invoked sequentially
   rather than concurrently with the reasoner so that planning compute
   does not contend with the reasoner co-residency budget.

Steady-state co-resident footprint during reasoning is approximately
11.2 GB / 24 GB. Per-query wall-clock latency is reported in Appendix D
of the paper (134.8 s/q total for the full TabAlign pipeline).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kwok2026tabalign,
  title     = {From Table to Cell: Attention for Better Reasoning with TabAlign},
  author    = {Kwok, Tung Sum Thomas and Zhang, Zeyong and Wang, Xinyu and
               Wang, Chunhe and Lin, Xiaofeng and Wu, Hanwei and Ding, Lei and
               Cheng, Guang and Guo, Zhijiang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Contact

For questions, please open a GitHub issue or contact Tung Sum Thomas Kwok
(tk1018@ucla.edu) or Zhijiang Guo (zhijiangguo@hkust-gz.edu.cn).

## License

This project is released under the [Apache 2.0 License](LICENSE).
