# MPQ (GPTQModel + QUIP#)

A unified quantization solver adapted to two upstream codebases via patch scripts:

- `third_party/GPTQModel` (upstream submodule)
- `third_party/quip-sharp` (upstream submodule)

Repository:
- GitHub: `https://github.com/poseidonchan/MPQ`
- Clone URL: `https://github.com/poseidonchan/MPQ.git`

Documentation:
- Install / repro: [docs/repro.md](docs/repro.md)
- Benchmark matrix: [docs/benchmark_matrix.md](docs/benchmark_matrix.md)

## Installation

```bash
git clone --recurse-submodules https://github.com/poseidonchan/MPQ.git
cd MPQ
python patches/apply_patches.py --check
bash install.sh              # both envs
bash install.sh --gptq-only  # GPTQModel only
bash install.sh --quip-only  # QUIP# only
```

### Installation environment variables

These variables affect environment creation and CUDA extension builds, so they should be set before running `install.sh`.

| Variable | Default | Description |
|---|---|---|
| `ENV_NAME` | backend default | Override the conda env name, e.g. `mpq-gptq-test` or `mpq-quip-test` |
| `PYTHON_VERSION` | `3.11` | Python version used when creating the conda env |
| `CUDA_TOOLKIT_VERSION` | `12.8` | CUDA toolkit version installed into the conda env |
| `TORCH_CUDA_ARCH_LIST` | `8.0+PTX` | GPU arch list used when compiling CUDA extensions |
| `INSTALL_FLASH_ATTN` | `0` | QUIP# env only; set to `1` to install flash-attention from upstream requirements |

Example:

```bash
ENV_NAME=mpq-quip-test CUDA_TOOLKIT_VERSION=12.8 TORCH_CUDA_ARCH_LIST=8.0+PTX bash install.sh --quip-only
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

On HPC where the clone already exists:

```bash
bash install.sh
```

For the full environment and patch workflow, see [docs/repro.md](docs/repro.md).

Patch health can be checked independently at any time:

```bash
python patches/apply_patches.py --check
```

## Running benchmarks

```bash
# 4-bit GPTQModel-family benchmark (scalar grid)
conda run -n mpq-gptq python scripts/benchmark.py \
  --bits 4 \
  --model-id Qwen/Qwen3-0.6B \
  --seeds 0

# 2-bit QUIP#
conda run -n mpq-quip python scripts/benchmark.py \
  --bits 2 \
  --base-model meta-llama/Llama-3.2-1B \
  --seeds 0
```

For benchmark options and result-table conventions, see:
- [docs/repro.md](docs/repro.md)
- [docs/benchmark_matrix.md](docs/benchmark_matrix.md)

Default PPL eval protocol is unified across both backends and matches the repo-style GPTQ / QUIP setup:
- `Wikitext2`: full test split
- `C4`: fixed `256 * seqlen` validation tokens

Override only if you intentionally want a token-capped eval:

```bash
--eval-max-eval-tokens 131072
```

## Reproducing PPL

Use the commands below to reproduce the current paper-style PPL numbers.

### BF16 reference

```bash
conda run -n mpq-gptq python scripts/eval_hf_ppl.py \
  --model-id Qwen/Qwen3-4B \
  --eval-datasets wikitext2 c4 \
  --eval-seqlen 2048 \
  --max-eval-tokens 0 \
  --json-out runs/qwen3_4b_bf16_ppl.json
```

`--max-eval-tokens 0` is important:
- `Wikitext2` uses the full `test` split
- `C4` uses the default `256 * 2048` validation-token budget

### Scalar-grid PPL benchmark used for current Qwen results

For fair `GPTQ` vs `VAMP` comparison on `3-bit` / `4-bit` scalar-grid runs, use the same internal rotation on both solvers:
- `rotation=internal_hadamard`
- `preprocess in {none, awq}`
- `solver in {gptq, vamp}`
- `polish_mode in {none, block_cd}`

Example: reproduce the `Qwen3-4B / 4-bit` scalar-grid PPL sweep

```bash
conda run -n mpq-gptq python scripts/benchmark.py \
  --bits 4 \
  --model-id Qwen/Qwen3-4B \
  --seeds 0 \
  --quant-preprocesses none awq \
  --quant-rotations internal_hadamard \
  --quant-solvers gptq vamp \
  --quant-polish-modes none block_cd \
  --run-id qwen3_4b_b4_internalhad_evalfix \
  --resume-existing
```

Example: reproduce the `Qwen3-4B / 3-bit` scalar-grid PPL sweep

```bash
conda run -n mpq-gptq python scripts/benchmark.py \
  --bits 3 \
  --model-id Qwen/Qwen3-4B \
  --seeds 0 \
  --quant-preprocesses none awq \
  --quant-rotations internal_hadamard \
  --quant-solvers gptq vamp \
  --quant-polish-modes none block_cd \
  --run-id qwen3_4b_b3_internalhad_evalfix \
  --resume-existing
```

The same pattern applies to `Qwen3-0.6B` and `Qwen3-8B`; only `--model-id`, `--bits`, and `--run-id` change.

## Solver is unified

The same solver code is used in both integration paths:

- Core solver: `mpq/solver.py`
- GPTQModel adapter: `mpq/adapters/gptqmodel.py` (scalar/groupwise quantization)
- QUIP# adapter: `mpq/adapters/quip_sharp.py` (E8P12 codebook refinement)

The primary solver family lives behind an orthogonal combo interface (`backend`, `codebook`, `preprocess`, `rotation`, `solver`, `polish_mode`). The main research path is `VAMP + topk block CD polish`.

For the GPTQModel backend, `rotation` names are explicit:
- `model_hadamard`: GPTQModel whole-model randomized Hadamard rotation
- `model_random`: GPTQModel whole-model dense random orthogonal rotation
- `internal_hadamard`: MPQ/VAMP internal right-side block Hadamard

For the current scalar-grid PPL benchmark, use `internal_hadamard` for both `gptq` and `vamp`. This keeps the rotation axis aligned when comparing solvers.

## Supported model backbones

Public benchmark paths target dense decoder-only HuggingFace causal LMs.

- `scripts/backend_gptq.py` follows GPTQModel's architecture support, including Llama and Qwen families.
- Public `3-bit` and `4-bit` runs are handled through the GPTQModel backend path.
- `scripts/backend_quip.py` now supports Llama/Qwen/Mistral-like dense decoder stacks through a repo-side adapter layer.
- Multimodal, MoE, and non-decoder architectures are still out of scope for the public benchmark path.

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
  --quant-solver vamp --quant-polish-mode block_cd \
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

## Saved artifacts

Both backends save a native artifact and a HuggingFace-exported artifact by default.

- GPTQModel backend:
  - native packed model: `<output-dir>/quantized`
  - HF export: `<output-dir>/hf`
- QUIP# backend:
  - native packed checkpoint: `<output-dir>/ckpt`
  - HF export: `<output-dir>/hf`

`backend_results.json` also exposes normalized paths:
- `paths.native_dir`
- `paths.hf_dir`
