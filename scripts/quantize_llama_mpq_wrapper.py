from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))
from quantize_llama import quantize_finetune_llama as quip_quant  # noqa: E402
from mpq.data import _iter_text_rows, tensor_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mpq-method", type=str, default="vamp")
    parser.add_argument("--mpq-solver", type=str, default=None)
    parser.add_argument("--mpq-polish-mode", type=str, default=None)
    parser.add_argument("--mpq-topk-k", type=int, default=16)
    parser.add_argument("--mpq-best-iterate", "--mpq_best_iterate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mpq-compile-prior", "--mpq_compile_prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant-backend", type=str, default=None)
    parser.add_argument("--quant-codebook", type=str, default=None)
    parser.add_argument("--quant-preprocess", type=str, default=None)
    parser.add_argument("--quant-rotation", type=str, default=None)
    parser.add_argument("--quant-solver", type=str, default=None)
    parser.add_argument("--quant-polish-mode", type=str, default=None)
    parser.add_argument("--quant-topk-source", type=str, default=None)
    parser.add_argument("--quant-topk-k", type=int, default=None)
    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper")
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-after-layer", type=int, default=None)
    parser.add_argument("--calibration-seed", type=int, default=None)
    parser.add_argument("--expected-calibration-manifest", type=str, default=None)
    wrapper_args, passthrough = parser.parse_known_args()

    torch.set_grad_enabled(False)
    try:
        quip_quant.mp.set_start_method("spawn")
    except RuntimeError:
        pass
    quip_quant.mp.set_sharing_strategy(
        os.environ.get("MPQ_TORCH_SHARING_STRATEGY", "file_system")
    )
    args = quip_quant.parser.parse_args(passthrough)
    calibration_seed = int(
        args.seed if wrapper_args.calibration_seed is None else wrapper_args.calibration_seed
    )
    torch.manual_seed(int(args.seed))
    os.makedirs(args.save_path, exist_ok=True)
    setattr(args, "mpq_method", str(wrapper_args.mpq_method))
    if wrapper_args.quant_backend is not None:
        setattr(args, "quant_backend", str(wrapper_args.quant_backend))
    if wrapper_args.quant_codebook is not None:
        setattr(args, "quant_codebook", str(wrapper_args.quant_codebook))
    if wrapper_args.quant_preprocess is not None:
        setattr(args, "quant_preprocess", str(wrapper_args.quant_preprocess))
    if wrapper_args.quant_rotation is not None:
        setattr(args, "quant_rotation", str(wrapper_args.quant_rotation))
        setattr(args, "incoh_mode", str(wrapper_args.quant_rotation))
    if wrapper_args.quant_solver is not None:
        setattr(args, "quant_solver", str(wrapper_args.quant_solver))
    if wrapper_args.quant_polish_mode is not None:
        setattr(args, "quant_polish_mode", str(wrapper_args.quant_polish_mode))
    if wrapper_args.quant_topk_source is not None:
        setattr(args, "quant_topk_source", str(wrapper_args.quant_topk_source))
    if wrapper_args.quant_topk_k is not None:
        setattr(args, "quant_topk_k", int(wrapper_args.quant_topk_k))
    setattr(args, "ojbkq_random_trials", int(wrapper_args.ojbkq_random_trials))
    setattr(args, "ojbkq_alpha_mode", str(wrapper_args.ojbkq_alpha_mode))
    setattr(args, "ojbkq_alpha_override", None if wrapper_args.ojbkq_alpha_override is None else float(wrapper_args.ojbkq_alpha_override))
    setattr(args, "ojbkq_alpha_scale", float(wrapper_args.ojbkq_alpha_scale))
    setattr(args, "ojbkq_sample_topk", None if wrapper_args.ojbkq_sample_topk is None else int(wrapper_args.ojbkq_sample_topk))
    if wrapper_args.stop_after_layer is not None:
        setattr(args, "stop_after_layer", int(wrapper_args.stop_after_layer))
    if wrapper_args.mpq_solver is not None:
        setattr(args, "mpq_solver", str(wrapper_args.mpq_solver))
    if wrapper_args.mpq_polish_mode is not None:
        setattr(args, "mpq_polish_mode", str(wrapper_args.mpq_polish_mode))
    setattr(args, "mpq_topk_k", int(wrapper_args.mpq_topk_k))
    setattr(args, "mpq_best_iterate", bool(wrapper_args.mpq_best_iterate))
    setattr(args, "mpq_compile_prior", bool(wrapper_args.mpq_compile_prior))
    setattr(args, "summary_only", bool(wrapper_args.summary_only))
    original_sample = quip_quant.utils.sample_token_blocks
    original_quantize_linear = quip_quant.quip.quantize_linear
    quip_quant.utils._iter_text_rows = _iter_text_rows

    def sample_with_calibration_seed(*sample_args, **sample_kwargs):
        sample_kwargs["seed"] = calibration_seed
        tokens = original_sample(*sample_args, **sample_kwargs)
        payload = {
            "protocol": "token_ids_sha256_v1",
            "sha256": tensor_sha256(tokens.to(dtype=torch.int64)),
            "shape": [int(dim) for dim in tokens.shape],
            "tokens": int(tokens.numel()),
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "seed": calibration_seed,
            "size": int(args.devset_size),
            "seqlen": int(args.ctx_size),
        }
        manifest_out = Path(args.save_path) / "calibration_manifest_quant.json"
        manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        expected_path = wrapper_args.expected_calibration_manifest
        if expected_path:
            expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
            if str(expected.get("sha256")) != str(payload["sha256"]):
                raise RuntimeError(
                    "quantization calibration hash does not match Hessian calibration hash: "
                    f"{payload['sha256']} != {expected.get('sha256')}"
                )
        return tokens

    quip_quant.utils.sample_token_blocks = sample_with_calibration_seed

    def quantize_linear_with_fingerprint(weights, save_path, hessian_path, cb, quant_args, device="cpu"):
        dtype = torch.float64 if bool(quant_args.use_fp64) else torch.float32
        scales = [weight.to(dtype).square().mean().sqrt() for weight in weights]
        hessian_payload = torch.load(hessian_path, map_location=torch.device("cpu"))
        hessian = quip_quant.utils.flat_to_sym(
            hessian_payload["flatH"],
            hessian_payload["n"],
        )
        mu = hessian_payload["mu"]
        hessian.add_(mu[None, :] * mu[:, None])
        weight_matrix = torch.vstack(
            [
                weights[index].to(dtype).cpu() / scales[index].cpu()
                for index in range(len(weights))
            ]
        )
        hessian = quip_quant.utils.regularize_H(
            hessian,
            hessian_payload["n"],
            quant_args.sigma_reg,
        )
        fingerprint = {
            "protocol": "paired_w_h_sha256_v1",
            "w_sha256": tensor_sha256(weight_matrix),
            "h_sha256": tensor_sha256(hessian),
            "w_shape": [int(dim) for dim in weight_matrix.shape],
            "h_shape": [int(dim) for dim in hessian.shape],
            "basis": "pre_incoherence",
        }
        result = original_quantize_linear(
            weights,
            save_path,
            hessian_path,
            cb,
            quant_args,
            device,
        )
        checkpoint_path = Path(save_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["input_fingerprint"] = fingerprint
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        return result

    quip_quant.quip.quantize_linear = quantize_linear_with_fingerprint
    quip_quant.finetune.quip.quantize_linear = quantize_linear_with_fingerprint
    quip_quant.main(args)


if __name__ == "__main__":
    main()
