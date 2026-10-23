from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))

from mpq.quip_generic import collect_hessians_generic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--calibration_seed", default=None, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--devset_size", default=256, type=int)
    parser.add_argument("--ctx_size", default=4096, type=int)
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--sample_proc", default=1, type=int)
    parser.add_argument("--calibration_dataset", default="c4", type=str)
    parser.add_argument("--calibration_split", default="train", type=str)
    parser.add_argument("--model-arch", default="auto", type=str)
    args = parser.parse_args()
    collect_hessians_generic(args)


if __name__ == "__main__":
    main()
