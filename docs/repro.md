# Repro & Install

This repo integrates `mpq` into `GPTQModel` and `QUIP#` while keeping third-party logic patch-only.

Repository:
- GitHub: `https://github.com/poseidonchan/MPQ`
- Clone URL: `https://github.com/poseidonchan/MPQ.git`

## Clone with submodules

Recommended:

```bash
git clone --recurse-submodules https://github.com/poseidonchan/MPQ.git
cd MPQ
```

If the repo is already cloned without submodules:

```bash
git submodule update --init --recursive
```

Do not manually clone arbitrary upstream revisions into `third_party/`. The repo expects the pinned submodule SHAs and applies local compatibility patches on top.

## Patch workflow

- Apply patches before building CUDA extensions:

```bash
python patches/apply_patches.py
```

- Keep algorithm code in this repo:
  - `mpq/solver.py`
  - `mpq/adapters/*`
  - `mpq/polish/*`
  - `scripts/*`

- Keep third-party changes thin:
  - dispatch hooks
  - config plumbing
  - import / compatibility glue

## Recommended environments

Use two conda envs because GPTQModel and QUIP# pin different stacks.

- `mpq-gptq`
- `mpq-quip`

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
  --quant-topk-k 16 \
  --model-id Qwen/Qwen3-0.6B \
  --bits 4 \
  --group-size 128 \
  --seed 0 \
  --output-dir runs/gptq_topk_example \
  --json-out runs/gptq_topk_example/backend_results.json
```

`VAMP + block CD`:

```bash
python scripts/backend_gptq.py \
  --quant-backend gptqmodel \
  --quant-codebook grid_g128 \
  --quant-preprocess none \
  --quant-rotation hadamard \
  --quant-solver vamp \
  --quant-polish-mode block_cd \
  --model-id Qwen/Qwen3-0.6B \
  --bits 4 \
  --group-size 128 \
  --seed 0 \
  --mpq-iters 16 \
  --mpq-damping 0.7 \
  --mpq-prior-tau 0.25 \
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
  --quant-topk-k 16 \
  --base-model meta-llama/Llama-3.2-1B \
  --seed 0 \
  --mpq-iters 16 \
  --mpq-damping 0.7 \
  --mpq-prior-tau 0.25 \
  --output-dir runs/quip_vamp_topk_example \
  --json-out runs/quip_vamp_topk_example/backend_results.json
```

## Quantizing your own model

To quantize a custom HuggingFace model, use the single-backend scripts directly.

### 4-bit (GPTQModel)

```bash
conda run -n mpq-gptq python scripts/backend_gptq.py \
  --model-id <your-hf-model> \
  --bits 4 --group-size 128 \
  --quant-solver vamp --quant-polish-mode block_cd \
  --output-dir runs/my_model \
  --json-out runs/my_model/results.json
```

Key flags:
- `--bits`: 2, 3, 4, or 8
- `--group-size`: quantization group size (128 is standard for 4-bit)
- `--quant-solver`: `gptq` (baseline) or `vamp` (MPQ)
- `--quant-polish-mode`: `none` or `block_cd` for the public scalar benchmark path

### 2-bit (QUIP#)

```bash
conda run -n mpq-quip python scripts/backend_quip.py \
  --base-model <your-hf-model> \
  --quant-solver vamp --quant-polish-mode topk_block_cd \
  --quant-topk-k 16 \
  --output-dir runs/my_model \
  --json-out runs/my_model/results.json
```

Key flags:
- `--quant-solver`: `ldlq` (QUIP# baseline), `ldlq_tune10`, or `vamp` (MPQ)
- `--quant-polish-mode`: `none`, `block_cd`, or `topk_block_cd`
- `--quant-rotation`: `had` (default) or `kron`

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
  --quant-polish-modes none block_cd \
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
  --quant-topk-k 16
```

QUIP-side ablation:

```bash
python scripts/ablation.py \
  --bits 2 \
  --base-model meta-llama/Llama-3.2-1B \
  --quant-rotation had \
  --quant-solver vamp \
  --quant-polish-mode topk_block_cd \
  --quant-topk-k 16
```

## Legacy compatibility

Old method aliases still work:

- `scripts/backend_gptq.py --method mpq`
- `scripts/backend_quip.py --method quip+mpq`
- `scripts/benchmark.py --methods ...`

They are kept for compatibility only; new experiments should use the combo flags.
