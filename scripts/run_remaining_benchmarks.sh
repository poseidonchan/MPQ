#!/usr/bin/env bash
set -uo pipefail

# Remaining benchmark runs: 4B (retry vamp) + 8B (all)
# --resume-existing skips already-completed combos

export HF_ALLOW_CODE_EVAL=1

COMMON_TASKS="arc_easy arc_challenge piqa winogrande hellaswag"
COMMON_ARGS="--lm-eval-batch-size auto --resume-existing"

log() { echo "===== $(date '+%Y-%m-%d %H:%M:%S') $* ====="; }

# === Qwen3-4B (retry failed combos) ===
log "Qwen3-4B 4-bit"
python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_4b_b4

log "Qwen3-4B 3-bit"
python scripts/benchmark.py --bits 3 --model-id Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_4b_b3

log "Qwen3-4B 2-bit"
python scripts/benchmark.py --bits 2 --base-model Qwen/Qwen3-4B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_4b_b2

# === Qwen3-8B ===
log "Qwen3-8B 4-bit"
python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_8b_b4

log "Qwen3-8B 3-bit"
python scripts/benchmark.py --bits 3 --model-id Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_8b_b3

log "Qwen3-8B 2-bit"
python scripts/benchmark.py --bits 2 --base-model Qwen/Qwen3-8B --seeds 0 \
  --lm-eval-tasks $COMMON_TASKS $COMMON_ARGS --run-id qwen3_8b_b2

log "ALL DONE"
