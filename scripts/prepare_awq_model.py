from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.data import load_or_build_calibration_blocks
from scripts import backend_gptq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--awq-batch-size", type=int, default=4)
    parser.add_argument("--apply-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gptqmodel-extra-site-packages", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    meta_path = out_dir / "awq_preprocess_meta.json"
    expected_meta = {
        "model_id": str(args.model_id),
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "sym": bool(args.sym),
        "calibration_dataset": str(args.calibration_dataset),
        "calibration_split": str(args.calibration_split),
        "calibration_limit": int(args.calibration_limit),
        "calibration_seqlen": int(args.calibration_seqlen),
        "seed": int(args.seed),
        "awq_batch_size": int(args.awq_batch_size),
        "apply_clip": bool(args.apply_clip),
    }
    if (out_dir / "config.json").exists() and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if all(existing.get(k) == v for k, v in expected_meta.items()):
            print(f"[awq] cache hit: {out_dir}", flush=True)
            return

    backend_gptq._force_system_compilers()
    backend_gptq._apply_patches()
    backend_gptq._ensure_gptqmodel_importable(extra_site_packages=args.gptqmodel_extra_site_packages)

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT
    from gptqmodel import QuantizeConfig
    from transformers import AutoTokenizer

    device = backend_gptq._resolve_device_str(args.device)
    model_path = backend_gptq._resolve_model_path(str(args.model_id))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    calib_blocks = load_or_build_calibration_blocks(
        tokenizer=tokenizer,
        dataset_name=str(args.calibration_dataset),
        split=str(args.calibration_split),
        limit=int(args.calibration_limit),
        seqlen=int(args.calibration_seqlen),
        seed=int(args.seed),
        output_dir=str(out_dir / "calibration_cache"),
        log_fn=lambda message: print(message, flush=True),
    )

    qcfg = QuantizeConfig(
        bits=int(args.bits),
        group_size=int(args.group_size),
        sym=bool(args.sym),
        format=FORMAT.GPTQ_V2,
        device=device,
        desc_act=False,
        act_group_aware=False,
    )
    model = GPTQModel.load(model_path, qcfg)
    tracked_cuda_devices = backend_gptq._tracked_cuda_devices()
    backend_gptq._reset_cuda_peak_memory_stats(tracked_cuda_devices)
    backend_gptq._cuda_synchronize_all(tracked_cuda_devices)
    start = time.time()
    backend_gptq._apply_awq_preprocess(
        qmodel=model,
        calibration_blocks=calib_blocks,
        bits=int(args.bits),
        group_size=int(args.group_size),
        sym=bool(args.sym),
        apply_clip=bool(args.apply_clip),
        batch_size=int(args.awq_batch_size),
    )
    backend_gptq._cuda_synchronize_all(tracked_cuda_devices)
    elapsed = float(time.time() - start)
    peak_memory = backend_gptq._snapshot_cuda_peak_memory(tracked_cuda_devices)

    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model = getattr(model, "model", model)
    hf_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    expected_meta.update(
        {
            "source_model_path": str(model_path),
            "time_s": elapsed,
            "peak_memory": peak_memory,
        }
    )
    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[awq] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
