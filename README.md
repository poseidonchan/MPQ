# MPGQ Unified (GPTQModel + QUIP#)

This repo provides a **single canonical MPGQ solver implementation** (VAMP-style message passing + greedy polish)
and adapts it to two upstream codebases via patch scripts:

- `third_party/GPTQModel` (upstream submodule)
- `third_party/quip-sharp` (upstream submodule)

## Third-party policy (important)

- **Do not edit** `third_party/*` directly.
- All compatibility changes live in `patches/*.patch` and are applied by `python patches/apply_patches.py`.
- Most entrypoints call `apply_patches.py` automatically (notably `scripts/backend_gptq.py` and `scripts/backend_quip.py`).
- If a submodule becomes “dirty” due to build artifacts, clean it with:
  - `git -C third_party/<name> reset --hard`
  - `git -C third_party/<name> clean -fdx`

## Solver is unified

The same solver code is used in both integration paths:

- Core solver: `mpgq/solver.py`
- GPTQModel adapter: `mpgq/adapters/gptqmodel.py` (scalar/groupwise quantization)
- QUIP# adapter: `mpgq/adapters/quip_sharp.py` (E8P12 codebook refinement)

## Key entrypoints

- Apply patches: `python patches/apply_patches.py`
- Full benchmark runner: `python scripts/benchmark.py ...`
- Single-module QUIP# proxy: `python scripts/bench_quip_module_proxy.py ...`
- Qwen3-0.6B 3-bit suite: `python scripts/bench_qwen3_0p6b_3bit.py ...`
- GPTQ-side ablations: `python scripts/ablation.py ...`

## Setup notes (recommended)

We typically use **two conda envs** because GPTQModel and QUIP# pin different Torch/Transformers stacks:

- `mpgq-gptq`: GPTQModel backend (`scripts/backend_gptq.py`, `scripts/bench_qwen3_0p6b_3bit.py`)
- `mpgq-quip`: QUIP# backend (`scripts/backend_quip.py`, `scripts/bench_quip_module_proxy.py`)

See `docs/repro.md` for installation tips + the quality/latency gates we enforce.
