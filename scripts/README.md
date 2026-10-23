# Scripts

Users should normally call `mpq quantize`. The files in this directory fall
into three groups.

## Backend Entrypoints

- `backend_gptq.py`: grouped scalar quantization through GPTQModel.
- `backend_quip.py`: E8P12 quantization through QUIP#.

The public CLI resolves a recipe and delegates to one of these entrypoints.
Their full argument surfaces are retained for specialized reproduction and
debugging.

## Backend Helpers

The E8P12 path invokes the Hessian collection, quantization, Hugging Face
export, and evaluation helpers in this directory. Environment setup is handled
by `setup_env_gptq.sh` and `setup_env_quip.sh`.

## Research Orchestration

Benchmark grids, ablations, smoke probes, and cluster launchers are development
tools. They are not part of the stable user API and are excluded from public
source archives.
