#!/usr/bin/env bash
# End-to-end run of the TabAlign inference pipeline.
#
# Stage 1: LLaDA diffusion planner emits a numbered cell-grounded plan.
# Stage 2: Plan executor grounds each plan step to a tool call and runs the
#          multi-step reasoning loop.
#
# Set DATASET_NAME and DATASET_PATH to match your local benchmark layout.
# Outputs are written under plans/ and results/ in the current directory.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

DATASET_NAME="${DATASET_NAME:-tabmwp}"
DATASET_PATH="${DATASET_PATH:-/path/to/${DATASET_NAME}}"
SPLIT="${SPLIT:-test}"
START="${START:-0}"
END="${END:-200}"

PLAN_OUT="plans/llada_${DATASET_NAME}_${SPLIT}_${START}_${END}.jsonl"
EXEC_OUT="results/executed_${DATASET_NAME}_${SPLIT}_${START}_${END}.jsonl"

mkdir -p plans results

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python "${SRC_DIR}"/diffusion_planner.py \
    --model-name GSAI-ML/LLaDA-8B-Instruct \
    --llada-device cuda:0 \
    --max-tokens 256 \
    --llada-steps 128 \
    --llada-block-length 32 \
    --llada-cfg-scale 1.5 \
    --temperature 0.0 \
    --"${DATASET_NAME}-path" "${DATASET_PATH}" \
    --"${DATASET_NAME}-split" "${SPLIT}" \
    --start "${START}" --end "${END}" \
    --output "${PLAN_OUT}" \
    --resume

python "${SRC_DIR}"/plan_executor.py \
    --plans  "${PLAN_OUT}" \
    --output "${EXEC_OUT}" \
    --provider ollama --model-name qwen3-vl:8b \
    --max-tokens 16384 \
    --temperature 0.7 \
    --max-steps 6 \
    --table-input-mode text \
    --resume

echo
echo "Plan output:     ${PLAN_OUT}"
echo "Execution output: ${EXEC_OUT}"
