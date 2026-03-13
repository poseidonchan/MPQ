#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mpgq-quip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; please install Miniconda/Anaconda first" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install -U pip
conda run -n "${ENV_NAME}" python -m pip install -e .
conda run -n "${ENV_NAME}" python -m pip install -r third_party/quip-sharp/requirements.txt
conda run -n "${ENV_NAME}" python -m pip install -e third_party/quip-sharp/quiptools

echo "OK: created/updated conda env ${ENV_NAME}"

