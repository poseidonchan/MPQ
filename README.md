# MPQ (GPTQModel + QUIP#)

A unified quantization solver adapted to two upstream codebases via patch scripts:

- `third_party/GPTQModel` (upstream submodule)
- `third_party/quip-sharp` (upstream submodule)

## Quick Start

```bash
git clone --recurse-submodules <url>
cd mpq-unified
bash install.sh              # both envs
bash install.sh --gptq-only  # GPTQModel only
bash install.sh --quip-only  # QUIP# only
```

On HPC where clone is already done:

```bash
bash install.sh
```

## Running benchmarks

```bash
# 4-bit GPTQModel
conda run -n mpq-gptq python scripts/benchmark.py --bits 4 --model-id Qwen/Qwen3-0.6B --seeds 0

# 2-bit QUIP#
conda run -n mpq-quip python scripts/benchmark.py --bits 2 --base-model meta-llama/Llama-3.2-1B --seeds 0
```

## Solver is unified

The same solver code is used in both integration paths:

- Core solver: `mpq/solver.py`
- GPTQModel adapter: `mpq/adapters/gptqmodel.py` (scalar/groupwise quantization)
- QUIP# adapter: `mpq/adapters/quip_sharp.py` (E8P12 codebook refinement)

The primary solver family lives behind an orthogonal combo interface (`backend`, `codebook`, `preprocess`, `rotation`, `solver`, `polish_mode`). The main research path is `VAMP + topk block CD polish`.

## Key entrypoints

- Full benchmark runner: `python scripts/benchmark.py ...`
- Single-backend entrypoints:
  - `python scripts/backend_gptq.py ...`
  - `python scripts/backend_quip.py ...`
- Single-module QUIP# proxy: `python scripts/bench_quip_module_proxy.py ...`
- Qwen3-0.6B 3-bit suite: `python scripts/bench_qwen3_0p6b_3bit.py ...`
- GPTQ-side ablations: `python scripts/ablation.py ...`

## Two conda environments

GPTQModel and QUIP# pin different Torch/Transformers stacks:

- `mpq-gptq`: GPTQModel backend
- `mpq-quip`: QUIP# backend

## Third-party policy

- **Do not edit** `third_party/*` directly.
- All compatibility changes live in `patches/*.patch` and are applied by `python patches/apply_patches.py`.
- Keep algorithm logic in `mpq/*` and `scripts/*`; third-party patches should stay as thin dispatch/config glue.

## Quantizing your own model

### 4-bit (GPTQModel backend)

```bash
conda run -n mpq-gptq python scripts/backend_gptq.py \
  --model-id <your-hf-model> \
  --bits 4 --group-size 128 \
  --quant-solver vamp --quant-polish-mode topk_block_cd \
  --output-dir runs/my_model \
  --json-out runs/my_model/results.json
```

### 2-bit (QUIP# backend)

```bash
conda run -n mpq-quip python scripts/backend_quip.py \
  --base-model <your-hf-model> \
  --quant-solver vamp --quant-polish-mode topk_block_cd \
  --output-dir runs/my_model \
  --json-out runs/my_model/results.json
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_TOOLKIT_VERSION` | `12.8` | CUDA toolkit version for conda install |
| `TORCH_CUDA_ARCH_LIST` | `8.0+PTX` | GPU arch for CUDA extension compilation |
| `INSTALL_FLASH_ATTN` | `0` | Set to `1` to install flash-attention (QUIP# env) |

See [docs/repro.md](docs/repro.md) for detailed installation, combo examples, and benchmark/ablation recipes.
See [docs/benchmark_matrix.md](docs/benchmark_matrix.md) for the supported combo table.
