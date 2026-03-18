# Repro & Install

This repo integrates `mpgq` into `GPTQModel` and `QUIP#` while keeping third-party logic patch-only.

## Patch workflow

- Apply patches before building CUDA extensions:

```bash
python patches/apply_patches.py
```

- Keep algorithm code in this repo:
  - `mpgq/solver.py`
  - `mpgq/adapters/*`
  - `mpgq/polish/*`
  - `scripts/*`

- Keep third-party changes thin:
  - dispatch hooks
  - config plumbing
  - import / compatibility glue

## Recommended environments

Use two conda envs because GPTQModel and QUIP# pin different stacks.

- `mpgq-gptq`
- `mpgq-quip`

In each env:

```bash
python -m pip install -e .
python patches/apply_patches.py
```

Backend-specific deps:

```bash
# GPTQModel env
python -m pip install -r third_party/GPTQModel/requirements.txt

# QUIP env
python -m pip install -r third_party/quip-sharp/requirements.txt
```

### fast-hadamard-transform

On newer GPUs it is usually safest to build from the vendored submodule:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
  python -m pip install --no-build-isolation -v third_party/fast-hadamard-transform
```

## Combo interface

The public benchmark/backend interface is now:

- `backend`
- `codebook`
- `preprocess`
- `rotation`
- `solver`
- `polish_mode`

Shared polish modes:

- `none`
- `block_cd`
- `topk_block_cd`

Top-k polish always uses `topk_source=aux_vamp`.

## Single-backend examples

### GPTQModel normal grid

`GPTQ + topk block CD`:

```bash
python scripts/backend_gptq.py \
  --quant-backend gptqmodel \
  --quant-codebook grid_g128 \
  --quant-preprocess none \
  --quant-rotation hadamard \
  --quant-solver gptq \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 8 \
  --model-id Qwen/Qwen3-0.6B \
  --bits 4 \
  --group-size 128 \
  --seed 0 \
  --output-dir runs/gptq_topk_example \
  --json-out runs/gptq_topk_example/backend_results.json
```

`VAMP + topk block CD`:

```bash
python scripts/backend_gptq.py \
  --quant-backend gptqmodel \
  --quant-codebook grid_g128 \
  --quant-preprocess none \
  --quant-rotation hadamard \
  --quant-solver vamp \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 8 \
  --model-id Qwen/Qwen3-0.6B \
  --bits 4 \
  --group-size 128 \
  --seed 0 \
  --mpgq-iters 16 \
  --mpgq-damping 0.7 \
  --mpgq-prior-tau 0.25 \
  --output-dir runs/vamp_topk_example \
  --json-out runs/vamp_topk_example/backend_results.json
```

### GPTQModel AWQ

```bash
python scripts/backend_gptq.py \
  --quant-backend gptqmodel \
  --quant-codebook grid_g128 \
  --quant-preprocess awq \
  --quant-rotation hadamard \
  --quant-solver gptq \
  --quant-polish-mode block_cd \
  --model-id Qwen/Qwen3-0.6B \
  --bits 4 \
  --group-size 128 \
  --seed 0 \
  --output-dir runs/awq_gptq_cd \
  --json-out runs/awq_gptq_cd/backend_results.json
```

### QUIP# E8P12

Native LDLQ baseline:

```bash
python scripts/backend_quip.py \
  --quant-backend quip \
  --quant-codebook e8p12 \
  --quant-preprocess none \
  --quant-rotation had \
  --quant-solver ldlq \
  --quant-polish-mode none \
  --base-model meta-llama/Llama-3.2-1B \
  --seed 0 \
  --output-dir runs/ldlq_example \
  --json-out runs/ldlq_example/backend_results.json
```

Default QUIP# baseline (`LDLQ + tune10`):

```bash
python scripts/backend_quip.py \
  --quant-backend quip \
  --quant-codebook e8p12 \
  --quant-preprocess none \
  --quant-rotation had \
  --quant-solver ldlq_tune10 \
  --quant-polish-mode none \
  --base-model meta-llama/Llama-3.2-1B \
  --seed 0 \
  --output-dir runs/ldlq_tune10_example \
  --json-out runs/ldlq_tune10_example/backend_results.json
```

`VAMP + topk block CD`:

```bash
python scripts/backend_quip.py \
  --quant-backend quip \
  --quant-codebook e8p12 \
  --quant-preprocess none \
  --quant-rotation had \
  --quant-solver vamp \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 8 \
  --base-model meta-llama/Llama-3.2-1B \
  --seed 0 \
  --mpgq-iters 16 \
  --mpgq-damping 0.7 \
  --mpgq-prior-tau 0.25 \
  --output-dir runs/quip_vamp_topk_example \
  --json-out runs/quip_vamp_topk_example/backend_results.json
```

## Benchmark examples

### E8P12 matrix

```bash
python scripts/benchmark.py \
  --bits 2 \
  --base-model meta-llama/Llama-3.2-1B \
  --quant-solvers ldlq vamp ldlq_tune10 \
  --quant-polish-modes none block_cd topk_block_cd \
  --quant-rotations had \
  --seeds 0
```

### Normal grid matrix

```bash
python scripts/benchmark.py \
  --bits 4 \
  --model-id Qwen/Qwen3-0.6B \
  --quant-solvers gptq vamp \
  --quant-polish-modes none block_cd topk_block_cd \
  --quant-preprocesses none awq \
  --quant-rotations hadamard \
  --seeds 0
```

Outputs:

- `results.json`
- `benchmark.md`

## Ablation examples

Fix a combo, then sweep VAMP hyperparameters:

```bash
python scripts/ablation.py \
  --bits 4 \
  --model-id Qwen/Qwen3-0.6B \
  --quant-preprocess none \
  --quant-rotation hadamard \
  --quant-solver vamp \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 8
```

QUIP-side ablation:

```bash
python scripts/ablation.py \
  --bits 2 \
  --base-model meta-llama/Llama-3.2-1B \
  --quant-rotation had \
  --quant-solver vamp \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 8
```

## Legacy compatibility

Old method aliases still work:

- `scripts/backend_gptq.py --method mpgq`
- `scripts/backend_quip.py --method quip+mpgq`
- `scripts/benchmark.py --methods ...`

They are kept for compatibility only; new experiments should use the combo flags.
