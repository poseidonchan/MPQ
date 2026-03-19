from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))

from mpq.quip_generic import export_dense_hf_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized_path", type=str, required=True)
    parser.add_argument("--hf_output_path", type=str, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    export_dense_hf_from_ckpt(args)


if __name__ == "__main__":
    main()

