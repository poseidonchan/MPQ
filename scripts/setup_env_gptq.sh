#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mpgq-gptq}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; please install Miniconda/Anaconda first" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install -U pip
conda run -n "${ENV_NAME}" python -m pip install -e .

# GPTQModel's editable build expects torch to already be present.
conda run -n "${ENV_NAME}" python -m pip install -U "torch>=2.8.0"

# Install a CUDA toolkit inside the env for building CUDA extensions (fast_hadamard_transform).
conda install -y -n "${ENV_NAME}" -c nvidia "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"

# Build fast_hadamard_transform from the pinned submodule against the installed torch.
conda run -n "${ENV_NAME}" env FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python -m pip install --no-build-isolation -v third_party/fast-hadamard-transform

# Avoid compiling GPTQModel CUDA extensions in this research repo by default.
# This makes the env robust to mismatched system nvcc vs torch CUDA (e.g. nvcc 11.x with torch cu12.x).
conda run -n "${ENV_NAME}" env BUILD_CUDA_EXT=0 python -m pip install -e third_party/GPTQModel --no-build-isolation

echo "OK: created/updated conda env ${ENV_NAME}"
