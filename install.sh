#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash install.sh [--gptq-only | --quip-only]

Bootstrap MPQ and its pinned backend dependencies. By default, both the
GPTQModel and QUIP# environments are installed.

Options:
  --gptq-only  Install only the scalar GPTQModel backend.
  --quip-only  Install only the E8P12 QUIP# backend.
  -h, --help   Show this help message.
EOF
}

INSTALL_GPTQ=1
INSTALL_QUIP=1

for arg in "$@"; do
  case "$arg" in
    --gptq-only) INSTALL_QUIP=0 ;;
    --quip-only) INSTALL_GPTQ=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${INSTALL_GPTQ}" == "0" && "${INSTALL_QUIP}" == "0" ]]; then
  echo "--gptq-only and --quip-only are mutually exclusive" >&2
  exit 1
fi

for command_name in python git conda; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 1
  fi
done

if ! python -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "MPQ bootstrap requires Python 3.10 or newer" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "==> Preparing pinned third-party backends..."
PATCH_ARGS=(--bootstrap)
if [[ "${INSTALL_GPTQ}" == "1" && "${INSTALL_QUIP}" == "0" ]]; then
  PATCH_ARGS+=(--gptq-only)
elif [[ "${INSTALL_GPTQ}" == "0" && "${INSTALL_QUIP}" == "1" ]]; then
  PATCH_ARGS+=(--quip-only)
fi
python patches/apply_patches.py "${PATCH_ARGS[@]}"

if [[ "${INSTALL_GPTQ}" == "1" ]]; then
  echo "==> Setting up mpq-gptq environment..."
  bash scripts/setup_env_gptq.sh
fi

if [[ "${INSTALL_QUIP}" == "1" ]]; then
  echo "==> Setting up mpq-quip environment..."
  bash scripts/setup_env_quip.sh
fi

echo "==> Done."
