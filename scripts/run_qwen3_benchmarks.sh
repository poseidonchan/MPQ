#!/usr/bin/env bash
set -euo pipefail

# Full benchmark matrix: Qwen3 0.6B / 4B / 8B at 2/3/4-bit
# 9 invocations, 69 total quantization+eval runs
# Order: 4-bit → 3-bit → 2-bit per model, smallest model first

export HF_ALLOW_CODE_EVAL=1

COMMON_TASKS="mmlu arc_challenge gsm8k humaneval hellaswag"

log() { echo "===== $(date '+%Y-%m-%d %H:%M:%S') $* ====="; }

# === Qwen3-0.6B ===
log "Qwen3-0.6B 4-bit (1/9)"
python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-0.6B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_0.6b_b4 --resume-existing

log "Qwen3-0.6B 3-bit (2/9)"
python scripts/benchmark.py --bits 3 --model-id Qwen/Qwen3-0.6B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_0.6b_b3 --resume-existing

log "Qwen3-0.6B 2-bit (3/9)"
python scripts/benchmark.py --bits 2 --base-model Qwen/Qwen3-0.6B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_0.6b_b2 --resume-existing

# === Qwen3-4B ===
log "Qwen3-4B 4-bit (4/9)"
python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_4b_b4 --resume-existing

log "Qwen3-4B 3-bit (5/9)"
python scripts/benchmark.py --bits 3 --model-id Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_4b_b3 --resume-existing

log "Qwen3-4B 2-bit (6/9)"
python scripts/benchmark.py --bits 2 --base-model Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_4b_b2 --resume-existing

# === Qwen3-8B ===
log "Qwen3-8B 4-bit (7/9)"
python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_8b_b4 --resume-existing

log "Qwen3-8B 3-bit (8/9)"
python scripts/benchmark.py --bits 3 --model-id Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_8b_b3 --resume-existing

log "Qwen3-8B 2-bit (9/9)"
python scripts/benchmark.py --bits 2 --base-model Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS \
  --run-id qwen3_8b_b2 --resume-existing

log "ALL DONE"
