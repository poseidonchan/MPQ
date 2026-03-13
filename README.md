# MPGQ Unified (GPTQModel + QUIP#)

This repo provides a **single canonical MPGQ solver implementation** (Message Passing + Greedy Quantization)
and integrates it into two upstream codebases via patch scripts:

- `third_party/GPTQModel` (upstream submodule)
- `third_party/quip-sharp` (upstream submodule)

Key entrypoints:

- Apply patches: `python patches/apply_patches.py`
- Unified benchmark: `python scripts/benchmark.py ...`
- GPTQ-side ablations: `python scripts/ablation.py ...`

See `docs/repro.md` for the quality/latency gates we enforce.

