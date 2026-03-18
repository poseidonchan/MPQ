#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Full Dense PTQ Benchmark — plan.md execution
# Each step uses --resume-existing so interrupted runs can be restarted.

echo "=== Step 1: Sanity Pair — Llama-3.2-3B ==="

echo "--- Llama-3.2-3B bits=2 (QUIP, 14 combos) ---"
python3 scripts/benchmark.py --bits 2 \
  --base-model meta-llama/Llama-3.2-3B \
  --run-id sanity_llama3b_b2 \
  --resume-existing

echo "--- Llama-3.2-3B bits=3 (GPTQ, 36 combos) ---"
python3 scripts/benchmark.py --bits 3 \
  --model-id meta-llama/Llama-3.2-3B \
  --run-id sanity_llama3b_b3 \
  --resume-existing

echo "--- Llama-3.2-3B bits=4 (GPTQ, 36 combos) ---"
python3 scripts/benchmark.py --bits 4 \
  --model-id meta-llama/Llama-3.2-3B \
  --run-id sanity_llama3b_b4 \
  --resume-existing

echo "=== Step 2: Sanity Pair — Qwen3-4B ==="

echo "--- Qwen3-4B bits=2 (QUIP, 14 combos) ---"
python3 scripts/benchmark.py --bits 2 \
  --base-model Qwen/Qwen3-4B \
  --run-id sanity_qwen4b_b2 \
  --resume-existing

echo "--- Qwen3-4B bits=3 (GPTQ, 36 combos) ---"
python3 scripts/benchmark.py --bits 3 \
  --model-id Qwen/Qwen3-4B \
  --run-id sanity_qwen4b_b3 \
  --resume-existing

echo "--- Qwen3-4B bits=4 (GPTQ, 36 combos) ---"
python3 scripts/benchmark.py --bits 4 \
  --model-id Qwen/Qwen3-4B \
  --run-id sanity_qwen4b_b4 \
  --resume-existing

echo "=== Step 3: Remaining small/medium models ==="

for model in "Qwen/Qwen3-0.6B" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-2-7b-hf"; do
  slug=$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
  echo "--- $model bits=2 ---"
  python3 scripts/benchmark.py --bits 2 \
    --base-model "$model" \
    --run-id "${slug}_b2" \
    --resume-existing

  echo "--- $model bits=3 ---"
  python3 scripts/benchmark.py --bits 3 \
    --model-id "$model" \
    --run-id "${slug}_b3" \
    --resume-existing

  echo "--- $model bits=4 ---"
  python3 scripts/benchmark.py --bits 4 \
    --model-id "$model" \
    --run-id "${slug}_b4" \
    --resume-existing
done

echo "=== Step 4: Large models ==="

for model in "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "meta-llama/Llama-2-13b-hf"; do
  slug=$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
  echo "--- $model bits=2 ---"
  python3 scripts/benchmark.py --bits 2 \
    --base-model "$model" \
    --run-id "${slug}_b2" \
    --resume-existing

  echo "--- $model bits=3 ---"
  python3 scripts/benchmark.py --bits 3 \
    --model-id "$model" \
    --run-id "${slug}_b3" \
    --resume-existing

  echo "--- $model bits=4 ---"
  python3 scripts/benchmark.py --bits 4 \
    --model-id "$model" \
    --run-id "${slug}_b4" \
    --resume-existing
done

echo "=== All benchmark runs complete ==="
