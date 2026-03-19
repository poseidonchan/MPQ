#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mpq-quip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; please install Miniconda/Anaconda first" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -U pip
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -e .

# Install torch with CUDA first.
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -U "torch>=2.8.0"

# Install a CUDA toolkit inside the env for building CUDA extensions.
# This avoids mismatched system nvcc vs torch CUDA (e.g. nvcc 11.x with torch cu12.x).
conda install -y -n "${ENV_NAME}" -c nvidia "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"

# quip-sharp pins flash-attn which may fail to build on some systems; allow skipping.
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
REQ="third_party/quip-sharp/requirements.txt"
if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  TMP_REQ="$(mktemp)"
  # Install torch/fast-hadamard-transform separately to avoid build-isolation failures.
  grep -vE '^(torch==|fast_hadamard_transform==|transformers==|huggingface_hub==)' "${REQ}" > "${TMP_REQ}"
  conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
else
  TMP_REQ="$(mktemp)"
  # Install torch/fast-hadamard-transform separately to avoid build-isolation failures.
  grep -vE '^(flash_attn==|torch==|fast_hadamard_transform==|transformers==|huggingface_hub==)' "${REQ}" > "${TMP_REQ}"
  conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
fi

# Upgrade lm_eval to 0.4+ for unified lm-evaluation-harness API.
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir "lm_eval>=0.4.7"

# Llama-3.2 configs require a newer `transformers` for `rope_scaling.rope_type="llama3"`.
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -U "transformers>=4.45,<5" "huggingface_hub>=0.34"

# Build fast_hadamard_transform from source against the already-installed torch.
# We pin and patch the upstream repo as a submodule for reproducibility.
conda run -n "${ENV_NAME}" env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --no-build-isolation -v third_party/fast-hadamard-transform

# Build quiptools_cuda; force a PTX-only arch to avoid requiring a brand-new nvcc.
# quiptools' setup.py imports torch; disable build isolation so the build env can see the installed torch.
conda run -n "${ENV_NAME}" env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip install --no-cache-dir -e third_party/quip-sharp/quiptools --no-build-isolation

echo "OK: created/updated conda env ${ENV_NAME}"
