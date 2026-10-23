from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))

from mpq.data import _iter_text_rows, tensor_sha256


def main() -> None:
    from quantize_llama import hessian_offline_llama as quip_hessian

    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--calibration-manifest-out", required=True)
    wrapper_args, passthrough = wrapper.parse_known_args()
    args = quip_hessian.parser.parse_args(passthrough)

    original_sample = quip_hessian.utils.sample_token_blocks
    quip_hessian.utils._iter_text_rows = _iter_text_rows

    def sample_and_record(*sample_args, **sample_kwargs):
        tokens = original_sample(*sample_args, **sample_kwargs)
        payload = {
            "protocol": "token_ids_sha256_v1",
            "sha256": tensor_sha256(tokens.to(dtype=torch.int64)),
            "shape": [int(dim) for dim in tokens.shape],
            "tokens": int(tokens.numel()),
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "seed": int(args.seed),
            "size": int(args.devset_size),
            "seqlen": int(args.ctx_size),
        }
        out = Path(wrapper_args.calibration_manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return tokens

    quip_hessian.utils.sample_token_blocks = sample_and_record
    if quip_hessian.mp.get_start_method(allow_none=True) is None:
        quip_hessian.mp.set_start_method("spawn")
    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    numpy.random.seed(int(args.seed))
    os.makedirs(args.save_path, exist_ok=True)
    quip_hessian.main(args)


if __name__ == "__main__":
    main()
