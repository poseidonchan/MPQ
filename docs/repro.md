# Repro & gates

This repo integrates `mpgq` into **two upstream submodules** (GPTQModel + QUIP#) while keeping
third-party code upstream-clean. All third-party compatibility changes live in `patches/*.patch`
and are applied via `python patches/apply_patches.py`.

Most entrypoints apply patches automatically (e.g. `scripts/backend_gptq.py`, `scripts/backend_quip.py`),
but you still need to run patching manually before **building** any third-party CUDA extensions.

## Environments (recommended)

GPTQModel and QUIP# pin different Torch/Transformers stacks. To avoid dependency conflicts we
recommend two conda envs:

- `mpgq-gptq`: GPTQModel backend runs (3/4-bit)
- `mpgq-quip`: QUIP# backend runs (2-bit E8P12)

At minimum, in each env:

- Install this repo: `python -m pip install -e .`
- Install the backend deps you plan to use:
  - GPTQModel deps: `python -m pip install -r third_party/GPTQModel/requirements.txt`
  - QUIP# deps: `python -m pip install -r third_party/quip-sharp/requirements.txt`

## Prereq: fast_hadamard_transform (recommended)

On new GPUs (e.g. SM 12.0 / RTX 5090), you typically need to **build from source**.
We vendor `fast-hadamard-transform` as a pinned submodule and patch it for robust installs.

Install (inside the target conda env):
```bash
python patches/apply_patches.py
# Build with the conda-provided CUDA toolchain (avoid picking up /usr/local/cuda).
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
  python -m pip install --no-build-isolation -v third_party/fast-hadamard-transform
```

## Gate A (GPTQModel, 4-bit Qwen3 0.6B)
- Target: MPGQ exact-full achieves ~22.1 PPL on Wikitext2 (±0.3).
- Recipe: C4 token blocks 128×2048, seed=0, iters=10, tau=0.2, damping=0.5, polish=3, rotation=hadamard.

Repro:
```bash
python scripts/benchmark.py \
  --bits 4 --model-id Qwen/Qwen3-0.6B \
  --methods mpgq gptq \
  --seeds 0 \
  --calibration-dataset c4 --calibration-split train --calibration-limit 128 --calibration-seqlen 2048 \
  --eval-seqlen 2048 --eval-max-eval-tokens 131072 \
  --mpgq-iters 10 --mpgq-prior-tau 0.2 --mpgq-damping 0.5 --mpgq-polish-sweeps 3 --mpgq-em-obs-precision
```

## Gate B (QUIP#, 2-bit Llama-3.2-1B)
- Target: `quip+mpgq` (MMSE-VAMP + gram polish + random init) beats LDLQ baseline on Wikitext2 for seeds {0,1,2}.
- Additionally: per-layer/per-block surrogate error must satisfy `mpgq <= ldlq`.
- Latency: wall-time ≤ 5× LDLQ.

Repro:
```bash
python scripts/benchmark.py \
  --bits 2 --base-model meta-llama/Llama-3.2-1B \
  --methods quip quip+mpgq \
  --seeds 0 1 2 \
  --devset-size 128 --ctx-size 2048 --batch-size 16 --sample-proc 1 --scale-override 0.9 \
  --mpgq-init random --mpgq-iters 4 --mpgq-damping 0.5 --mpgq-prior-tau 0.0 \
  --mpgq-polish-sweeps 3 --mpgq-polish-metric gram
```

## Convenience benchmarks

- Single-module QUIP# proxy (deterministic, for solver debugging): `scripts/bench_quip_module_proxy.py`
- Qwen3-0.6B 3-bit Wikitext2 suite (mpgq / awq+mpgq / gptq-strong): `scripts/bench_qwen3_0p6b_3bit.py`
