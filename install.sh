#!/usr/bin/env bash
set -euo pipefail

# Usage: bash install.sh [--gptq-only | --quip-only]
# Default: install both environments

INSTALL_GPTQ=1
INSTALL_QUIP=1

for arg in "$@"; do
  case "$arg" in
    --gptq-only) INSTALL_QUIP=0 ;;
    --quip-only) INSTALL_GPTQ=0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "==> Initializing submodules..."
git submodule update --init --recursive

echo "==> Verifying patch health..."
python patches/apply_patches.py --check

echo "==> Applying patches..."
python patches/apply_patches.py

if [[ "${INSTALL_GPTQ}" == "1" ]]; then
  echo "==> Setting up mpq-gptq environment..."
  bash scripts/setup_env_gptq.sh
fi

if [[ "${INSTALL_QUIP}" == "1" ]]; then
  echo "==> Setting up mpq-quip environment..."
  bash scripts/setup_env_quip.sh
fi

echo "==> Done."
