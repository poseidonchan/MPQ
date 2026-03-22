from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _force_system_compilers() -> None:
    candidates = {
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    }
    for var, fallback in candidates.items():
        current = os.environ.get(var, "")
        if "x86_64-conda-linux-gnu" in current or current.endswith(("-cc", "-c++")):
            os.environ[var] = fallback


def _apply_patches() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "patches" / "apply_patches.py")],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        out = str(proc.stdout or "")
        if "submodule has local modifications" in out.lower():
            print("[patches] reusing current patched third-party worktree", flush=True)
            return
        raise RuntimeError(f"apply_patches.py failed\n{out}")


def _ensure_gptqmodel_importable() -> None:
    try:
        import gptqmodel  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "GPTQModel"))
    import gptqmodel  # noqa: F401


@torch.inference_mode()
def _untie_word_embeddings_if_needed(model) -> None:
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        model.config.tie_word_embeddings = False
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())


def _load_model(model_id: str, *, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.inference_mode()
def _capture_first_layer_stats(
    *,
    model_id: str,
    rotation: str,
    device: torch.device,
    calib_blocks: List[torch.Tensor],
    target_names: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    from gptqmodel.quantization import FORMAT
    from gptqmodel.quantization.config import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization.rotation.rotation import fuse_layer_norms, rotate_model
    from mpq.dense_decoder_adapter import resolve_dense_decoder_adapter_for_model

    model, _tokenizer = _load_model(model_id, device=device, dtype=torch.float16)
    adapter = resolve_dense_decoder_adapter_for_model(model)

    if rotation in {"model_hadamard", "model_random"}:
        _untie_word_embeddings_if_needed(model)
        model = fuse_layer_norms(
            model=model,
            pre_lm_head_norm_module_name="model.norm",
            layers_node="model.layers",
            lm_head_name="lm_head",
        )
        rot_device = device if device.type != "mps" else torch.device("cpu")
        model, _ = rotate_model(
            model=model,
            rotate_mode="hadamard" if rotation == "model_hadamard" else "random",
            device=rot_device,
            layers_node="model.layers",
            lm_head_name="lm_head",
        )
        model.to(device)
        model.eval()

    layer0 = adapter.get_layers(model)[0]
    handles = adapter.get_layer_handles(layer0)
    modules = {
        "q_proj": handles.q_proj,
        "up_proj": handles.up_proj,
        "down_proj": handles.down_proj,
    }
    selected = {name: modules[name] for name in target_names}

    collectors: Dict[str, Any] = {}
    hooks = []
    for name, module in selected.items():
        qcfg = QuantizeConfig(
            bits=4,
            group_size=128,
            sym=True,
            format=FORMAT.GPTQ_V2,
            device=str(device),
            offload_to_disk=False,
            rotation=None,
            desc_act=False,
        )
        collector = GPTQ(module, qcfg)
        collectors[name] = collector

        def _hook(_module, inp, out, *, _collector=collector):
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            _collector.add_batch(x.detach(), out.detach())

        hooks.append(module.register_forward_hook(_hook))

    for block in calib_blocks:
        if isinstance(block, dict):
            input_ids = block["input_ids"].to(device)
            attention_mask = block.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device)
            model(input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            input_ids = block.to(device)
            model(input_ids, use_cache=False)

    for hook in hooks:
        hook.remove()

    results: Dict[str, Dict[str, Any]] = {}
    for name, collector in collectors.items():
        H = collector.finalize_hessian(target_device=device).detach().clone()
        W = collector.clone_module(device=device).detach().clone()
        nsamples = int(collector.nsamples)
        results[name] = {
            "module_path": (
                "model.layers.0.self_attn.q_proj"
                if name == "q_proj"
                else ("model.layers.0.mlp.up_proj" if name == "up_proj" else "model.layers.0.mlp.down_proj")
            ),
            "W": W,
            "H": H,
            "nsamples": nsamples,
            "rows": int(W.shape[0]),
            "cols": int(W.shape[1]),
        }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _make_qcfg(
    *,
    bits: int,
    group_size: int,
    device: torch.device,
    desc_act: bool,
    mpq_enabled: bool,
    solver: str,
    polish_mode: str,
    rotation: str,
    iters: int,
    topk_k: int,
    polish_sweeps: int,
) -> Any:
    from gptqmodel.quantization import FORMAT
    from gptqmodel.quantization.config import QuantizeConfig

    if rotation in {"model_hadamard", "internal_hadamard"}:
        quant_rotation = "hadamard"
    elif rotation == "model_random":
        quant_rotation = "random"
    else:
        quant_rotation = None

    qcfg = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        sym=True,
        format=FORMAT.GPTQ_V2,
        device=str(device),
        offload_to_disk=False,
        rotation=quant_rotation,
        desc_act=bool(desc_act),
    )
    qcfg.act_group_aware = False
    qcfg.mpq.enabled = bool(mpq_enabled)
    qcfg.mpq.solver = str(solver)
    qcfg.mpq.polish_mode = str(polish_mode)
    qcfg.mpq.topk_k = int(topk_k)
    qcfg.mpq.iters = int(iters)
    qcfg.mpq.damping = 0.5
    qcfg.mpq.prior_tau = 0.2
    qcfg.mpq.polish_sweeps = int(polish_sweeps)
    qcfg.mpq.em_obs_precision = True
    qcfg.mpq.block_rows = 128
    qcfg.mpq.rotation_seed = 0
    qcfg.mpq.gamma_start = 1.0
    qcfg.mpq.gamma_retry_factor = 4.0
    qcfg.mpq.gamma_retry_steps = 3
    qcfg.mpq.obs_precision = 0.0
    qcfg.mpq.compile_block_cd = False
    qcfg.mpq.rotation_requested = str(rotation)
    qcfg.mpq.rotation_mode = str(rotation)
    return qcfg


@torch.inference_mode()
def _rankspace_proxy_mean(W: torch.Tensor, H: torch.Tensor, Q: torch.Tensor) -> float:
    from mpq.adapters.gptqmodel import _vamp_hessian_eigen
    from mpq.linear import vamp_rankspace_error

    H_work = 0.5 * H.to(dtype=torch.float32)
    lam, V, _ = _vamp_hessian_eigen(H_work)
    lam = lam.to(device=W.device, dtype=torch.float32)
    V = V.to(device=W.device, dtype=torch.float32)
    S = torch.sqrt(lam).contiguous()
    Vs = V * S.unsqueeze(0)
    Y_u = W.to(dtype=torch.float32).matmul(Vs)
    err = vamp_rankspace_error(Y_u=Y_u, x=Q.to(dtype=torch.float32), Vs=Vs)
    return float(err.mean().item()) if err.numel() else 0.0


@torch.inference_mode()
def _rel_frob(W: torch.Tensor, Q: torch.Tensor) -> float:
    denom = float(W.norm().item())
    if denom <= 0.0:
        return 0.0
    return float((W - Q).norm().item() / denom)


@torch.inference_mode()
def _run_gptq_row(
    *,
    payload: Dict[str, Any],
    bits: int,
    group_size: int,
    device: torch.device,
    desc_act: bool,
    polish_mode: str,
    rotation_label: str,
    topk_k: int,
) -> Dict[str, Any]:
    from gptqmodel.quantization.gptq import GPTQ

    W = payload["W"]
    H = payload["H"]
    rows, cols = W.shape
    module = torch.nn.Linear(cols, rows, bias=False, device=device, dtype=torch.float32)
    module.weight.data.copy_(W.to(device=device, dtype=torch.float32))
    qcfg = _make_qcfg(
        bits=bits,
        group_size=group_size,
        device=device,
        desc_act=desc_act,
        mpq_enabled=(polish_mode != "none" or str(rotation_label) == "internal_hadamard"),
        solver="gptq",
        polish_mode=polish_mode,
        rotation=rotation_label,
        iters=16,
        topk_k=topk_k,
        polish_sweeps=3,
    )
    g = GPTQ(module, qcfg)
    g.quantizer.configure(perchannel=True)
    g.H = H.to(device=device, dtype=torch.float32).clone()
    g.nsamples = int(payload["nsamples"])
    g._hessian_dirty = False
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = g.quantize(blocksize=128)
    return {
        "family": "gptq",
        "solver": "gptq",
        "desc_act": bool(desc_act),
        "rotation_requested": str(rotation_label),
        "rotation_effective": str(rotation_label),
        "rotation_mode": (
            "internal_rotation" if str(rotation_label) == "internal_hadamard"
            else ("model_rotation" if rotation_label != "none" else "none")
        ),
        "polish_mode": str(polish_mode),
        "rel_frob": _rel_frob(W, Q),
        "rankspace_proxy_mean": _rankspace_proxy_mean(W, H, Q),
        "quant_time_s": float(duration),
        "avg_loss": float(avg_loss) if isinstance(avg_loss, (int, float)) else str(avg_loss),
        "damp": float(damp),
        "nsamples": int(nsamples),
        "group_size": int(group_size),
        "scale_shape": list(scale.shape),
        "zero_shape": list(zero.shape),
        "g_idx_shape": list(g_idx.shape),
    }


@torch.inference_mode()
def _run_vamp_row(
    *,
    payload: Dict[str, Any],
    bits: int,
    group_size: int,
    device: torch.device,
    rotation: str,
    polish_mode: str,
    topk_k: int,
) -> Dict[str, Any]:
    from gptqmodel.quantization.gptq import GPTQ
    from mpq.adapters.gptqmodel import GPTQAdapterConfig, run_mpq_quantization
    from mpq.solver import UniformVampConfig

    W = payload["W"]
    H = payload["H"]
    rows, cols = W.shape
    module = torch.nn.Linear(cols, rows, bias=False, device=device, dtype=torch.float32)
    module.weight.data.copy_(W.to(device=device, dtype=torch.float32))
    qcfg = _make_qcfg(
        bits=bits,
        group_size=group_size,
        device=device,
        desc_act=False,
        mpq_enabled=True,
        solver="vamp",
        polish_mode=polish_mode,
        rotation=rotation,
        iters=16,
        topk_k=topk_k,
        polish_sweeps=3,
    )
    qcfg.rotation = "hadamard" if str(rotation) in {"model_hadamard", "internal_hadamard"} else ("random" if str(rotation) == "model_random" else None)
    tmp = GPTQ(module, qcfg)
    tmp.quantizer.configure(perchannel=True)
    quantizer = tmp.quantizer
    vamp_cfg = UniformVampConfig(
        iters=16,
        damping=0.5,
        final_decision="map",
        init_from="rtn",
        gamma_start=1.0,
        gamma_retry_factor=4.0,
        gamma_retry_steps=3,
        prior_tau=0.2,
        obs_precision=0.0,
        em_obs_precision=True,
        em_obs_precision_damping=0.3,
        em_obs_precision_min=1e-4,
        em_obs_precision_max=1e8,
        polish_sweeps=3,
        polish_mode=str(polish_mode),
        topk_k=int(topk_k),
        compile_block_cd=False,
    )
    Q, _q_int, _scale, _zero, _g_idx, duration, stats = run_mpq_quantization(
        W=W.to(device=device, dtype=torch.float32),
        H=H.to(device=device, dtype=torch.float32),
        quantizer=quantizer,
        qcfg=qcfg,
        module_name=str(payload["module_path"]),
        vamp_cfg=vamp_cfg,
        adapter_cfg=GPTQAdapterConfig(block_rows=128, rotation_seed=0),
    )
    return {
        "family": "vamp",
        "solver": "vamp",
        "desc_act": False,
        "rotation_requested": str(rotation),
        "rotation_effective": str(stats.get("rotation_effective", rotation)),
        "rotation_mode": (
            "mpq_internal"
            if str(rotation) == "internal_hadamard"
            else ("model_rotation" if str(rotation) in {"model_random", "model_hadamard"} else "none")
        ),
        "polish_mode": str(polish_mode),
        "rel_frob": _rel_frob(W, Q),
        "rankspace_proxy_mean": _rankspace_proxy_mean(W, H, Q),
        "quant_time_s": float(duration),
        "stats": stats,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _force_system_compilers()
    _apply_patches()
    _ensure_gptqmodel_importable()

    from mpq.data import load_or_build_calibration_blocks
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    calib_blocks = load_or_build_calibration_blocks(
        tokenizer=tokenizer,
        dataset_name=str(args.calibration_dataset),
        split=str(args.calibration_split),
        limit=int(args.calibration_limit),
        seqlen=int(args.calibration_seqlen),
        seed=int(args.seed),
        output_dir=str(output_dir),
        log_fn=lambda s: print(s, flush=True),
    )
    target_names = [str(x).strip() for x in args.targets]

    captures: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rotation in sorted(
        {
            *args.gptq_rotations,
            *[r for r in args.vamp_rotations if r in {"model_random", "model_hadamard"}],
        }
    ):
        if rotation == "none":
            continue
        captures[rotation] = _capture_first_layer_stats(
            model_id=str(args.model_id),
            rotation=str(rotation),
            device=device,
            calib_blocks=calib_blocks,
            target_names=target_names,
        )
    captures["none"] = _capture_first_layer_stats(
        model_id=str(args.model_id),
        rotation="none",
        device=device,
        calib_blocks=calib_blocks,
        target_names=target_names,
    )

    summary: Dict[str, Any] = {
        "model_id": str(args.model_id),
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "seed": int(args.seed),
        "calibration_dataset": str(args.calibration_dataset),
        "calibration_split": str(args.calibration_split),
        "calibration_limit": int(args.calibration_limit),
        "calibration_seqlen": int(args.calibration_seqlen),
        "targets": target_names,
        "results": {},
    }

    for target_name in target_names:
        rows: List[Dict[str, Any]] = []
        for rotation in args.gptq_rotations:
            payload = captures[str(rotation)][target_name]
            for desc_act in args.gptq_desc_act:
                for polish_mode in args.gptq_polish_modes:
                    rows.append(
                        _run_gptq_row(
                            payload=payload,
                            bits=int(args.bits),
                            group_size=int(args.group_size),
                            device=device,
                            desc_act=bool(desc_act),
                            polish_mode=str(polish_mode),
                            rotation_label=str(rotation),
                            topk_k=int(args.topk_k),
                        )
                    )
        for rotation in args.vamp_rotations:
            payload = captures["model_random" if str(rotation) == "model_random" else ("model_hadamard" if str(rotation) == "model_hadamard" else "none")][target_name]
            for polish_mode in args.vamp_polish_modes:
                rows.append(
                    _run_vamp_row(
                        payload=payload,
                        bits=int(args.bits),
                        group_size=int(args.group_size),
                        device=device,
                        rotation=str(rotation),
                        polish_mode=str(polish_mode),
                        topk_k=int(args.topk_k),
                    )
                )
        summary["results"][target_name] = {
            "module_path": captures["none"][target_name]["module_path"],
            "rows": int(captures["none"][target_name]["rows"]),
            "cols": int(captures["none"][target_name]["cols"]),
            "nsamples_none": int(captures["none"][target_name]["nsamples"]),
            "rows_out": rows,
        }

    out_path = output_dir / "smoke_first_layer_rotation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"json_out": str(out_path)}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First-layer GPTQ/VAMP rotation smoke for scalar grid.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-k", type=int, default=16)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=4)
    parser.add_argument("--calibration-seqlen", type=int, default=512)
    parser.add_argument("--targets", type=str, nargs="+", default=["q_proj", "down_proj"])
    parser.add_argument(
        "--gptq-rotations",
        type=str,
        nargs="+",
        choices=["none", "model_hadamard", "model_random", "internal_hadamard"],
        default=["none", "model_hadamard"],
    )
    parser.add_argument("--gptq-desc-act", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--gptq-polish-modes", type=str, nargs="+", default=["none", "block_cd"])
    parser.add_argument(
        "--vamp-rotations",
        type=str,
        nargs="+",
        choices=["none", "model_hadamard", "model_random", "internal_hadamard"],
        default=["none", "internal_hadamard", "model_random"],
    )
    parser.add_argument("--vamp-polish-modes", type=str, nargs="+", default=["none", "block_cd"])
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "runs" / "qwen3_4b_first_layer_smoke"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
