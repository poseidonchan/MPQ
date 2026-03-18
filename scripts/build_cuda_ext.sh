#!/usr/bin/env bash
set -euo pipefail

# Optional CUDA builds.
# - QUIP#: quiptools_cuda (required for fast decode/decompress)
# - fast-hadamard-transform (optional)

ENV_NAME="${ENV_NAME:-mpq-quip}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

echo "[build] quiptools_cuda"
conda run -n "${ENV_NAME}" python -m pip install -e third_party/quip-sharp/quiptools

echo "[build] fast-hadamard-transform (optional)"
conda run -n "${ENV_NAME}" python -m pip install -U "git+https://github.com/Dao-AILab/fast-hadamard-transform.git" --no-build-isolation || true

echo "OK"

