from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import fcntl
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download

# Monkey-patch transformers.GPTQConfig.to_dict_optimum to drop `act_group_aware`.
# Newer GPTQModel sets it True for GPTQ method; optimum's GPTQQuantizer re-instantiates
# QuantizeConfig with the saved dict and the resulting (desc_act=True, act_group_aware=True)
# combo fails QuantizeConfig validation. Dropping the key falls back to the default behavior.
try:
    from transformers.utils.quantization_config import GPTQConfig as _HFGPTQConfig
    _orig_to_dict_optimum = _HFGPTQConfig.to_dict_optimum
    def _patched_to_dict_optimum(self):
        d = _orig_to_dict_optimum(self)
        # GPTQQuantizer defaults act_group_aware=True; explicitly force False so
        # the resulting QuantizeConfig doesn't fail on (desc_act=True, act_group_aware=True).
        d["act_group_aware"] = False
        # Force torch backend; the triton matmul kernel (TRITON_V2) hits a fp16-pointer
        # type bug at compile time on this triton/GPTQModel combo.
        d["backend"] = "torch"
        return d
    _HFGPTQConfig.to_dict_optimum = _patched_to_dict_optimum
except Exception:
    pass

# NOTE: torch.compile no-op patches removed — they slowed quant 5x.
# The triton dequant kernel bug only hits at eval reload time, not during quant.

# cuSOLVER in the torch 2.11+cu128 wheel crashes with
# CUSOLVER_STATUS_INVALID_VALUE on eigh for matrices larger than ~24k
# (hit by Llama-2-70B mlp.down_proj, dim 28672) on H200. Fixed by torch
# nightly cu130 (verified 28k×28k eigh completes in ~21s). If running on
# the older cu128 stack and the matrix is too large, MAGMA is the only
# working option; cu130 resolves it at the cuSOLVER source.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.combo import QuantCombo, make_combo
from mpq.data import calibration_fingerprint, load_or_build_calibration_blocks
from mpq.eval_utils import (
    eval_ppl_from_tokens as shared_eval_ppl_from_tokens,
    eval_protocol_for_dataset,
    eval_token_fingerprint,
    load_or_build_c4_tokens as shared_load_or_build_c4_tokens,
    load_or_build_wikitext2_tokens as shared_load_or_build_wikitext2_tokens,
)


def _resolve_device_str(requested: Optional[str]) -> str:
    """Resolve requested device without probing CUDA availability."""

    if requested:
        return str(requested)
    # Default to CUDA for benchmarking; callers can set `--device cpu` explicitly.
    return "cuda:0"


def _tracked_cuda_devices() -> list[torch.device]:
    if not torch.cuda.is_available():
        return []
    return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]


def _cuda_synchronize_all(devices: list[torch.device]) -> None:
    for dev in devices:
        try:
            torch.cuda.synchronize(dev)
        except Exception:
            pass


def _reset_cuda_peak_memory_stats(devices: list[torch.device]) -> None:
    for dev in devices:
        try:
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                    torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass


def _snapshot_cuda_peak_memory(devices: list[torch.device]) -> Dict[str, Any]:
    per_device: Dict[str, Dict[str, float]] = {}
    allocated: list[float] = []
    reserved: list[float] = []
    for dev in devices:
        try:
            alloc_gb = float(torch.cuda.max_memory_allocated(dev) / (1024**3))
            reserved_gb = float(torch.cuda.max_memory_reserved(dev) / (1024**3))
        except Exception:
            continue
        per_device[str(dev)] = {
            "allocated_gb": alloc_gb,
            "reserved_gb": reserved_gb,
        }
        allocated.append(alloc_gb)
        reserved.append(reserved_gb)
    return {
        "devices": per_device,
        "allocated_gb_max": float(max(allocated)) if allocated else 0.0,
        "allocated_gb_sum": float(sum(allocated)) if allocated else 0.0,
        "reserved_gb_max": float(max(reserved)) if reserved else 0.0,
        "reserved_gb_sum": float(sum(reserved)) if reserved else 0.0,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_quant_log_summary(qmodel: Any) -> Dict[str, Any]:
    rows = getattr(qmodel, "quant_log", None)
    if not isinstance(rows, list):
        return {
            "modules": 0,
            "solver_module_sum_s": 0.0,
            "solver_module_max_s": 0.0,
        }
    durations = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        duration = _safe_float(row.get("time"))
        if duration is not None:
            durations.append(duration)
    return {
        "modules": int(len(rows)),
        "solver_module_sum_s": float(sum(durations)) if durations else 0.0,
        "solver_module_max_s": float(max(durations)) if durations else 0.0,
    }


def _collect_quant_region_totals(qmodel: Any) -> Dict[str, float]:
    timer = getattr(qmodel, "quant_region_timer", None)
    if timer is None or not hasattr(timer, "snapshot"):
        return {}
    try:
        snapshot = timer.snapshot()
    except Exception:
        return {}
    if not isinstance(snapshot, dict):
        return {}
    regions = [
        "capture_inputs",
        "pre_quant_forward",
        "process_quant",
        "post_quant_forward",
        "submodule_finalize",
        "submodule_finalize_create",
        "submodule_finalize_pack",
        "submodule_finalize_offload",
        "process_finalize",
        "model_load",
        "model_save",
    ]
    totals: Dict[str, float] = {}
    for region in regions:
        stat = snapshot.get(region, {})
        if isinstance(stat, dict):
            total = _safe_float(stat.get("total"))
            if total is not None:
                totals[f"{region}_s"] = float(total)
    return totals


def _extract_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(module_name))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _collect_mpq_mechanism(qcfg: Any) -> Dict[str, Any]:
    mpq_cfg = getattr(qcfg, "mpq", None)
    rows = getattr(mpq_cfg, "_mechanism_summaries", None)
    if not isinstance(rows, list):
        return {}

    per_layer: list[Dict[str, Any]] = []
    gamma_sum = 0.0
    gamma_count = 0
    changed_sum = 0
    changed_total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if item.get("layer_index") is None:
            item["layer_index"] = _extract_layer_index(str(item.get("module", "")))
        per_layer.append(item)
        try:
            weight = int(item.get("stage1_gamma1_count", 0) or 0)
            gamma_sum += float(item.get("stage1_gamma1_mean", 0.0) or 0.0) * weight
            gamma_count += weight
        except (TypeError, ValueError):
            pass
        try:
            changed_sum += int(item.get("cleanup_changed_count", 0) or 0)
            changed_total += int(item.get("cleanup_total_count", 0) or 0)
        except (TypeError, ValueError):
            pass

    per_layer.sort(key=lambda item: (item.get("layer_index") is None, item.get("layer_index"), str(item.get("module", ""))))
    if not per_layer:
        return {}
    return {
        "regime": str(per_layer[0].get("regime", "3bit_scalar")),
        "cleanup_unit": str(per_layer[0].get("cleanup_unit", "scalar_element")),
        "modules": int(len(per_layer)),
        "per_layer": per_layer,
        "stage1_gamma1_mean_global": float(gamma_sum / max(gamma_count, 1)),
        "cleanup_changed_count_global": int(changed_sum),
        "cleanup_total_count_global": int(changed_total),
        "cleanup_changed_fraction_global": float(changed_sum / max(changed_total, 1)),
    }


def _collect_matrix_fingerprints(qcfg: Any) -> Dict[str, Any]:
    mpq_cfg = getattr(qcfg, "mpq", None)
    rows = getattr(mpq_cfg, "_layer_stats", []) if mpq_cfg is not None else []
    modules: list[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        fingerprint = row.get("input_fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        if not fingerprint.get("w_sha256") or not fingerprint.get("h_sha256"):
            continue
        modules.append(
            {
                "module": str(row.get("module", "")),
                "w_sha256": str(fingerprint["w_sha256"]),
                "h_sha256": str(fingerprint["h_sha256"]),
                "w_shape": list(fingerprint.get("w_shape", [])),
                "h_shape": list(fingerprint.get("h_shape", [])),
                "basis": str(fingerprint.get("basis", "")),
            }
        )
    modules.sort(key=lambda item: item["module"])
    encoded = json.dumps(modules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "protocol": "paired_w_h_sha256_v1",
        "module_count": len(modules),
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "modules": modules,
    }


def _force_system_compilers() -> None:
    """Avoid conda cross-compiler wrappers that may be incompletely installed."""

    candidates = {
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    }
    for var, fallback in candidates.items():
        current = os.environ.get(var, "")
        if "x86_64-conda-linux-gnu" in current or current.endswith(("-cc", "-c++")):
            os.environ[var] = fallback


def _git_head(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_patches() -> None:
    if os.environ.get("MPQ_SKIP_PATCHES", "").strip() in {"1", "true", "TRUE", "yes"}:
        print("[patches] MPQ_SKIP_PATCHES set; assuming patched worktree", flush=True)
        return
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "patches" / "apply_patches.py"),
            "--gptq-only",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        out = str(proc.stdout or "")
        lower = out.lower()
        if any(phrase in lower for phrase in (
            "submodule has local modifications",
            "checkout has local modifications",
            "directory not empty",
            "failed to remove temporary worktree",
            "patch already applied",
        )):
            print("[patches] reusing current patched third-party worktree", flush=True)
            return
        raise RuntimeError(f"apply_patches.py failed\n{out}")


def _ensure_gptqmodel_importable(*, extra_site_packages: Optional[str]) -> None:
    """Ensure `import gptqmodel` works even if not installed in the active env.

    This is helpful when the active env has a working CUDA stack (e.g. QUIP)
    but the GPTQModel deps live in a different env. We always prefer the repo
    checkout for `gptqmodel` so patch application stays authoritative.
    """

    try:
        import gptqmodel  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "GPTQModel"))
    if extra_site_packages:
        sys.path.append(str(extra_site_packages))
    import gptqmodel  # noqa: F401


def _resolve_model_path(model_ref: str) -> str:
    path = Path(model_ref)
    if path.exists() and (path / "config.json").exists():
        return str(path)
    return str(
        snapshot_download(
            model_ref,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json"],
        )
    )


_AWQ_SCALE_LOG: list = []
_AWQ_CLIP_LOG: list = []
_AWQ_LAYER_MAP: Dict[int, int] = {}
_AWQ_HOOKS_INSTALLED = False


def _awq_install_recording_hooks() -> None:
    global _AWQ_HOOKS_INSTALLED
    if _AWQ_HOOKS_INSTALLED:
        return
    from gptqmodel.quantization.awq.quantize import scale as scale_mod

    _orig_apply_scale = scale_mod.apply_scale
    _orig_apply_clip = scale_mod.apply_clip

    def _wrapped_apply_scale(module, scales_list, input_feat_dict=None):
        layer_idx = _AWQ_LAYER_MAP.get(id(module), -1)
        saved = []
        for entry in scales_list:
            prev_op_name, layer_names, scales, loss = entry[:4]
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = 0.0
            scales_cpu = scales.detach().to(torch.float32).cpu().clone()
            saved.append((str(prev_op_name), list(layer_names), scales_cpu, loss_val))
        _AWQ_SCALE_LOG.append((int(layer_idx), saved))
        return _orig_apply_scale(module, scales_list, input_feat_dict=input_feat_dict)

    def _wrapped_apply_clip(module, clip_list):
        layer_idx = _AWQ_LAYER_MAP.get(id(module), -1)
        saved = [(str(name), max_val.detach().to(torch.float32).cpu().clone()) for name, max_val in clip_list]
        _AWQ_CLIP_LOG.append((int(layer_idx), saved))
        return _orig_apply_clip(module, clip_list)

    scale_mod.apply_scale = _wrapped_apply_scale
    scale_mod.apply_clip = _wrapped_apply_clip
    _AWQ_HOOKS_INSTALLED = True


def _awq_register_decoder_layers(hf_model) -> int:
    _AWQ_LAYER_MAP.clear()
    try:
        layers = hf_model.model.layers
    except AttributeError:
        return 0
    for idx, layer in enumerate(layers):
        _AWQ_LAYER_MAP[id(layer)] = int(idx)
    return len(_AWQ_LAYER_MAP)


def _awq_dump_state(path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({"scales": _AWQ_SCALE_LOG, "clips": _AWQ_CLIP_LOG}, temporary)
    temporary.replace(output)
    print(
        f"[awq-dump] wrote {len(_AWQ_SCALE_LOG)} scale-layer records, "
        f"{len(_AWQ_CLIP_LOG)} clip-layer records -> {path}",
        flush=True,
    )


def _awq_load_and_apply(hf_model, path: str, *, apply_clips: bool = True) -> None:
    from gptqmodel.quantization.awq.quantize.scale import apply_clip, apply_scale
    import torch.nn as nn

    data = torch.load(path, map_location="cpu", weights_only=False)
    try:
        layers = hf_model.model.layers
    except AttributeError as exc:
        raise RuntimeError("cannot replay AWQ scales: model has no .model.layers") from exc
    scales_records = data.get("scales", [])
    clips_records = data.get("clips", []) if apply_clips else []
    original_dtypes: Dict[str, torch.dtype] = {
        name: module.weight.dtype
        for name, module in hf_model.named_modules()
        if isinstance(module, nn.Linear) and hasattr(module, "weight")
    }
    n_scale = 0
    for layer_idx, saved in scales_records:
        if not 0 <= int(layer_idx) < len(layers):
            continue
        module = layers[int(layer_idx)]
        scales_list = [(p, list(l), s, (loss if loss is not None else 0.0)) for p, l, s, loss in saved]
        apply_scale(module, scales_list)
        n_scale += 1
    n_clip = 0
    for layer_idx, saved in clips_records:
        if not 0 <= int(layer_idx) < len(layers):
            continue
        module = layers[int(layer_idx)]
        apply_clip(module, saved)
        n_clip += 1
    for name, module in hf_model.named_modules():
        dtype = original_dtypes.get(name)
        if dtype is not None and hasattr(module, "weight") and module.weight.dtype != dtype:
            module.weight.data = module.weight.data.to(dtype=dtype)
    print(
        f"[awq-load] applied {n_scale}/{len(scales_records)} scale layers, "
        f"{n_clip} clip layers from {path} (apply_clips={apply_clips})",
        flush=True,
    )


def _apply_awq_preprocess(
    *,
    qmodel,
    calibration_blocks,
    bits: int,
    group_size: int,
    sym: bool,
    apply_clip: bool,
    batch_size: int,
    stop_after_layer: Optional[int] = None,
) -> None:
    """Apply AWQ scale/(optional clip) in-place without replacing linear modules."""
    from gptqmodel import BACKEND
    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.loop_processor import LoopProcessor
    from gptqmodel.looper.module_looper import ModuleLooper
    from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

    class _AWQPreprocessProcessor(AWQProcessor):
        def apply_quant(self, named_linears, scales_list):
            for named_module in named_linears.values():
                named_module.module.cpu()

        def submodule_finalize(self, module, model, **kwargs):
            return

        def finalize(self, model, **kwargs):
            LoopProcessor.finalize(self, model=model, **kwargs)

        @classmethod
        def name(cls) -> str:  # pragma: no cover
            return "awq-preprocess"

    device = getattr(qmodel.quantize_config, "device", None)
    awq_qcfg = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=False,
        act_group_aware=False,
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        offload_to_disk=False,
        device=device,
    )
    processor = _AWQPreprocessProcessor(
        tokenizer=qmodel.tokenizer,
        qcfg=awq_qcfg,
        calibration=calibration_blocks,
        prepare_dataset_func=qmodel.prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=int(batch_size),
        gptq_model=qmodel,
        model=qmodel.model,
        calculate_w_wq_diff=False,
    )
    processor.apply_clip = bool(apply_clip)

    looper = ModuleLooper(qmodel, processors=[processor])
    original_callback = getattr(qmodel, "layer_callback", None)
    if stop_after_layer is not None:
        qmodel.layer_callback = _StopAfterLayer(int(stop_after_layer))
    try:
        looper.loop(backend=BACKEND, failsafe=awq_qcfg.failsafe)
    finally:
        qmodel.layer_callback = original_callback


class _StopAfterLayer:
    def __init__(self, target: Optional[int]):
        self._target = target
        self._triggered = False

    def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
        if self._target is None or self._triggered:
            return None
        if layer_idx > self._target or (submodule_finalized and layer_idx >= self._target):
            self._triggered = True
            from gptqmodel.looper.module_looper import StopMainLoop

            raise StopMainLoop


def _materialize_inference_tensors(module) -> None:
    """Convert any inference-mode tensors in params/buffers to regular tensors.

    Some GPTQModel preprocessing paths (notably AWQ scale/clip) may leave
    `Tensor.is_inference()==True` tensors attached to modules. Subsequent
    forward passes can crash with:
      `RuntimeError: Inference tensors do not track version counter.`

    Cloning outside `torch.inference_mode()` materializes a regular tensor.
    """
    import torch.nn as nn

    # Ensure clones are materialized as *non-inference* tensors even if some
    # upstream code left inference mode enabled.
    with torch.inference_mode(False):
        for submodule in module.modules():
            for name, param in list(getattr(submodule, "_parameters", {}).items()):
                if param is None:
                    continue
                t = param.detach()
                if hasattr(t, "is_inference") and bool(t.is_inference()):
                    submodule._parameters[name] = nn.Parameter(t.clone(), requires_grad=False)

            for name, buf in list(getattr(submodule, "_buffers", {}).items()):
                if buf is None:
                    continue
                if hasattr(buf, "is_inference") and bool(buf.is_inference()):
                    submodule._buffers[name] = buf.clone()

            # Defensive: some preprocessing paths may stash tensors directly on the module
            # without registering them as params/buffers. Clone those as well.
            for name, value in list(vars(submodule).items()):
                if name in {"_parameters", "_buffers", "_modules"}:
                    continue
                if isinstance(value, torch.Tensor) and hasattr(value, "is_inference") and bool(value.is_inference()):
                    setattr(submodule, name, value.clone())


def _strip_gptqmodel_hooked_modules(model) -> None:
    """Replace GPTQModel's hooked wrappers with plain torch modules.

    GPTQModel may wrap `nn.Linear` with `HookedLinear` during quantization input
    capture. In some preprocessing combinations (notably AWQ+...), these wrappers
    can leak into the final model and break evaluation/inference.
    """
    try:
        from gptqmodel.nn_modules.hooked_linear import HookedLinear  # type: ignore
    except Exception:  # noqa: BLE001
        return

    import torch.nn as nn

    def _replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, HookedLinear):
                has_bias = getattr(child, "bias", None) is not None
                out_features = int(getattr(child, "out_features"))
                in_features = int(getattr(child, "in_features"))
                replacement = nn.Linear(in_features, out_features, bias=bool(has_bias))
                w = child.weight.detach().clone()
                replacement.weight = nn.Parameter(w, requires_grad=False)
                if has_bias:
                    b = child.bias.detach().clone()
                    replacement.bias = nn.Parameter(b, requires_grad=False)
                setattr(parent, name, replacement)
                continue
            _replace(child)

    _replace(model)


@torch.inference_mode()
def _eval_cache_path(output_dir: str, tokenizer, dataset: str, max_eval_tokens: int, seqlen: int, seed: int) -> str:
    cache_dir = os.path.join(output_dir, "_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(
        f"{tokenizer_id}|{dataset}|{max_eval_tokens}|{seqlen}|{seed}|eval_v2".encode("utf-8")
    ).hexdigest()[:16]
    tokenizer_label = Path(tokenizer_id).name.replace("/", "_") or "tokenizer"
    dataset_label = str(dataset).replace("/", "_")
    return os.path.join(cache_dir, f"{dataset_label}_{tokenizer_label}_{max_eval_tokens}_{digest}.pt")


def _load_or_build_wikitext2_tokens(tokenizer, *, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    return shared_load_or_build_wikitext2_tokens(
        tokenizer,
        max_eval_tokens=int(max_eval_tokens),
        seqlen=int(seqlen),
        seed=int(seed),
        output_dir=str(output_dir),
    )


def _load_or_build_c4_tokens(tokenizer, *, model_ref: str, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    return shared_load_or_build_c4_tokens(
        tokenizer,
        repo_root=str(REPO_ROOT),
        model_ref=str(model_ref),
        max_eval_tokens=int(max_eval_tokens),
        seqlen=int(seqlen),
        seed=int(seed),
        output_dir=str(output_dir),
    )


@torch.no_grad()
def eval_ppl_from_tokens(
    model,
    tokens: torch.Tensor,
    *,
    seqlen: int = 2048,
) -> float:
    return shared_eval_ppl_from_tokens(model, tokens, seqlen=int(seqlen))


def _resolve_combo_and_runtime(args) -> tuple[Optional[QuantCombo], Dict[str, Any]]:
    requested_method = str(getattr(args, "method", "") or "").strip().lower()
    runtime: Dict[str, Any] = {
        "request_source": "combo",
        "requested_method": requested_method or None,
        "awq_preprocess": False,
        "awq_apply_clip": False,
        "desc_act": True,
        "sym": bool(args.sym),
        "legacy_rtn": False,
        "ojbkq": False,
    }

    if requested_method:
        runtime["request_source"] = "method"
        if requested_method in {"gptq", "gptq-strong"}:
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="none",
                rotation="model_hadamard",
                solver="gptq",
                polish_mode="none",
            )
            return combo, runtime
        if requested_method == "mpq":
            runtime["desc_act"] = False
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="none",
                rotation="internal_hadamard",
                solver="vamp",
                polish_mode="block_cd" if int(args.mpq_polish_sweeps) > 0 else "none",
            )
            return combo, runtime
        if requested_method in {"cdquant", "quantease"}:
            runtime["desc_act"] = False
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="none",
                rotation="internal_hadamard",
                solver=requested_method,
                polish_mode="none",
            )
            return combo, runtime
        if requested_method == "rtn":
            runtime["desc_act"] = False
            runtime["legacy_rtn"] = True
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="none",
                rotation="internal_hadamard",
                solver="vamp",
                polish_mode="none",
            )
            return combo, runtime
        if requested_method == "awq+gptq":
            runtime["awq_preprocess"] = True
            runtime["awq_apply_clip"] = True
            runtime["sym"] = bool(args.sym)
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="awq",
                rotation="model_hadamard",
                solver="gptq",
                polish_mode="none",
            )
            _apply_awq_clip_override(runtime, args)
            return combo, runtime
        if requested_method == "awq+mpq":
            runtime["awq_preprocess"] = True
            runtime["awq_apply_clip"] = False
            runtime["sym"] = bool(args.sym)
            runtime["desc_act"] = False
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="awq",
                rotation="internal_hadamard",
                solver="vamp",
                polish_mode="block_cd" if int(args.mpq_polish_sweeps) > 0 else "none",
            )
            _apply_awq_clip_override(runtime, args)
            return combo, runtime
        if requested_method == "ojbkq":
            runtime["ojbkq"] = True
            runtime["desc_act"] = False
            return None, runtime
        raise ValueError(f"unsupported method={requested_method!r} for backend_gptq")

    requested_rotation = str(getattr(args, "quant_rotation", "auto") or "auto").strip().lower()
    if str(getattr(args, "quant_solver", "gptq")).strip().lower() == "ojbkq":
        if requested_rotation in {"", "auto"}:
            requested_rotation = "model_hadamard"
        combo = make_combo(
            backend=str(getattr(args, "quant_backend", "gptqmodel")),
            codebook=str(getattr(args, "quant_codebook", "grid_g128")),
            preprocess=str(getattr(args, "quant_preprocess", "none")),
            rotation=requested_rotation,
            solver="ojbkq",
            polish_mode=str(getattr(args, "quant_polish_mode", "none")),
            topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
        )
        runtime["ojbkq"] = True
        runtime["desc_act"] = False
        runtime["awq_preprocess"] = combo.preprocess == "awq"
        runtime["awq_apply_clip"] = False
        runtime["sym"] = bool(args.sym)
        _apply_awq_clip_override(runtime, args)
        return combo, runtime
    if requested_rotation in {"", "auto"}:
        requested_rotation = "model_hadamard" if str(getattr(args, "quant_solver", "gptq")) == "gptq" else "internal_hadamard"

    combo = make_combo(
        backend=str(getattr(args, "quant_backend", "gptqmodel")),
        codebook=str(getattr(args, "quant_codebook", "grid_g128")),
        preprocess=str(getattr(args, "quant_preprocess", "none")),
        rotation=requested_rotation,
        solver=str(getattr(args, "quant_solver", "gptq")),
        polish_mode=str(getattr(args, "quant_polish_mode", "none")),
        topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
    )
    runtime["awq_preprocess"] = combo.preprocess == "awq"
    runtime["awq_apply_clip"] = combo.preprocess == "awq" and combo.solver == "gptq"
    runtime["sym"] = bool(args.sym)
    runtime["desc_act"] = combo.solver == "gptq"
    _apply_awq_clip_override(runtime, args)
    return combo, runtime


def _apply_awq_clip_override(runtime: Dict[str, Any], args) -> None:
    override = getattr(args, "awq_apply_clip", None)
    if override is not None:
        runtime["awq_apply_clip"] = bool(override) and bool(runtime.get("awq_preprocess", False))


def _build_quantize_config(combo: Optional[QuantCombo], runtime: Dict[str, Any], args) -> Any:
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization import FORMAT

    device = _resolve_device_str(getattr(args, "device", None))
    rotation_name = combo.rotation if combo is not None else "none"
    if rotation_name == "model_hadamard":
        quant_rotation = "hadamard"
    elif rotation_name == "model_random":
        quant_rotation = "random"
    elif rotation_name == "internal_hadamard":
        quant_rotation = "hadamard"
    else:
        quant_rotation = None

    qcfg = QuantizeConfig(
        bits=int(args.bits),
        group_size=int(args.group_size),
        sym=bool(runtime["sym"]),
        # Default `FORMAT.GPTQ` (v1) is meant for checkpoint interchange; it forces a v2->v1 qzeros
        # conversion during packing which breaks in-process TorchQuantLinear inference. We always
        # evaluate in-process here, so prefer internal v2 format.
        format=FORMAT.GPTQ_V2,
        device=device,
        offload_to_disk=bool(args.offload_to_disk),
        rotation=quant_rotation,
        desc_act=bool(runtime["desc_act"]),
        act_group_aware=False,
    )

    use_repo_managed = combo is not None and (
        combo.solver in {"vamp", "cdquant", "quantease"} or combo.polish_mode != "none" or combo.rotation == "internal_hadamard"
    )
    if use_repo_managed:
        qcfg.mpq.enabled = True
        # Route every repo-managed solver through the shared adapter. In
        # particular, OJBKQ must pass through this path so paired-W/H auditing
        # and the requested post-OJBKQ CD budget are not bypassed by the native
        # GPTQModel OJBKQ early return.
        qcfg.mpq.solver = str(combo.solver)
        qcfg.mpq.dispatch_solver = str(combo.solver)
        qcfg.mpq.polish_mode = str(combo.polish_mode)
        qcfg.mpq.init = str(args.mpq_init)
        qcfg.mpq.rotation_mode = str(combo.rotation)
        qcfg.mpq.rotation_requested = str(combo.rotation)
        qcfg.mpq.rotation_effective = (
            "internal_block_hadamard" if combo.rotation == "internal_hadamard" else str(combo.rotation)
        )
        qcfg.mpq.rotation_internal = bool(combo.rotation == "internal_hadamard")
        qcfg.mpq.rotation_mode_effective = "internal_rotation" if combo.rotation == "internal_hadamard" else ("model_rotation" if combo.rotation in {"model_hadamard", "model_random"} else "none")
        qcfg.mpq.topk_source = str(combo.topk_source)
        qcfg.mpq.topk_k = int(getattr(args, "quant_topk_k", 16))
        qcfg.mpq.iters = 0 if bool(runtime["legacy_rtn"]) else int(args.mpq_iters)
        qcfg.mpq.damping = float(args.mpq_damping)
        qcfg.mpq.prior_tau = 0.0 if bool(runtime["legacy_rtn"]) else float(args.mpq_prior_tau)
        qcfg.mpq.vamp_backend = "eager" if bool(runtime["legacy_rtn"]) else str(args.mpq_vamp_backend)
        qcfg.mpq.polish_sweeps = 0 if bool(runtime["legacy_rtn"]) else int(args.mpq_polish_sweeps)
        if bool(runtime["legacy_rtn"]):
            qcfg.mpq.compile_block_cd = False
            qcfg.mpq.block_cd_backend = "eager"
        else:
            qcfg.mpq.compile_block_cd = bool(args.mpq_compile_block_cd)
            if args.mpq_block_cd_backend is not None:
                qcfg.mpq.block_cd_backend = str(args.mpq_block_cd_backend)
            else:
                qcfg.mpq.block_cd_backend = "compile" if bool(args.mpq_compile_block_cd) else "eager"
        qcfg.mpq.em_obs_precision = False if bool(runtime["legacy_rtn"]) else bool(args.mpq_em_obs_precision)
        qcfg.mpq.block_rows = int(args.mpq_block_rows)
        qcfg.mpq.rotation_seed = int(args.mpq_rotation_seed)
        if combo.rotation == "internal_hadamard" and combo.solver in {"vamp", "cdquant", "quantease"}:
            qcfg.meta = dict(qcfg.meta or {})
            qcfg.meta["mpq_input_rotation"] = {
                "mode": "internal_block_hadamard",
                "seed": int(args.mpq_rotation_seed),
            }
        qcfg.mpq.paired_matrix_cache_dir = (
            None
            if getattr(args, "paired_matrix_cache_dir", None) is None
            else str(args.paired_matrix_cache_dir)
        )
        qcfg.mpq.gamma_start = float(args.mpq_gamma_start)
        qcfg.mpq.gamma_retry_factor = float(args.mpq_gamma_retry_factor)
        qcfg.mpq.gamma_retry_steps = int(args.mpq_gamma_retry_steps)
        qcfg.mpq.obs_precision = float(args.mpq_obs_precision)
        qcfg.mpq._mechanism_summaries = []
    else:
        qcfg.mpq.enabled = False

    if bool(runtime["ojbkq"]):
        from mpq.baselines.ojbkq import OJBKQConfig

        qcfg.act_group_aware = False
        qcfg.ojbkq = OJBKQConfig(
            enabled=True,
            random_trials=int(args.ojbkq_random_trials),
            alpha_mode=str(args.ojbkq_alpha_mode),
            alpha_override=None if args.ojbkq_alpha_override is None else float(args.ojbkq_alpha_override),
            alpha_scale=float(args.ojbkq_alpha_scale),
            sample_topk=None if args.ojbkq_sample_topk is None else int(args.ojbkq_sample_topk),
        )

    return qcfg


def _run_lm_eval(hf_model, tokenizer, *, tasks, batch_size, num_fewshot, include_path=None):
    """Run lm-evaluation-harness tasks in-process."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size)
    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    nfs_kw = {} if num_fewshot is not None and num_fewshot < 0 else {"num_fewshot": num_fewshot}
    tm_kw = {"task_manager": TaskManager(include_path=include_path)} if include_path else {}
    eval_results = simple_evaluate(
        model=lm, tasks=tasks, batch_size=batch_size,
        confirm_run_unsafe_code=True, **nfs_kw, **tm_kw,
    )
    results = {}
    for task_name, task_metrics in eval_results.get("results", {}).items():
        results[task_name] = {
            k: v for k, v in task_metrics.items() if k != "alias"
        }
    return {"tasks": list(tasks), "results": results}


def run(args) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    calibration_seed = int(args.seed if args.calibration_seed is None else args.calibration_seed)
    calibration_draw = int(
        calibration_seed if args.calibration_draw is None else args.calibration_draw
    )
    eval_seed = int(args.seed if args.eval_seed is None else args.eval_seed)
    resolved_device = _resolve_device_str(getattr(args, "device", None))
    if args.device and str(args.device).startswith("cuda") and not resolved_device.startswith("cuda"):
        raise RuntimeError(f"requested device={args.device!r} but CUDA failed to initialize")
    args.device = resolved_device
    _force_system_compilers()

    _apply_patches()

    _ensure_gptqmodel_importable(extra_site_packages=getattr(args, "gptqmodel_extra_site_packages", None))
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    combo, runtime = _resolve_combo_and_runtime(args)
    if args.desc_act is not None:
        runtime["desc_act"] = bool(args.desc_act)
    if (
        combo is not None
        and combo.rotation == "internal_hadamard"
        and combo.solver not in {"vamp", "cdquant", "quantease"}
        and not bool(getattr(args, "skip_save", False))
    ):
        raise ValueError(
            "Packed internal_hadamard artifacts currently require the MPQ "
            "VAMP, CDQuant, or QuantEase solver. Use --skip-save for this "
            "research-only solver path."
        )

    device = torch.device(str(args.device))
    tracked_cuda_devices = _tracked_cuda_devices()
    awq_time_s = 0.0
    awq_peak_memory: Dict[str, Any] = {}
    awq_fingerprint: Dict[str, Any] = {}
    stop_after_layer = (
        None
        if getattr(args, "stop_after_layer", None) is None
        else int(getattr(args, "stop_after_layer"))
    )

    model_source = str(args.model_id)
    model_id = str(args.model_id_label or args.model_id)
    model_path = _resolve_model_path(model_source)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    calib_blocks = load_or_build_calibration_blocks(
        tokenizer=tokenizer,
        dataset_name=str(args.calibration_dataset),
        split=str(args.calibration_split),
        limit=int(args.calibration_limit),
        seqlen=int(args.calibration_seqlen),
        seed=calibration_seed,
        output_dir=str(args.calibration_cache_dir or args.output_dir),
        log_fn=lambda s: print(s, flush=True),
    )
    calibration_manifest = calibration_fingerprint(calib_blocks)

    qcfg = _build_quantize_config(combo, runtime, args)
    t_load = time.time()
    model = GPTQModel.load(model_path, qcfg)
    _cuda_synchronize_all(tracked_cuda_devices)
    load_time_s = float(time.time() - t_load)
    if bool(runtime["awq_preprocess"]):
        _reset_cuda_peak_memory_stats(tracked_cuda_devices)
        _cuda_synchronize_all(tracked_cuda_devices)
        t_awq = time.time()
        awq_load_path = getattr(args, "awq_load_scales", None)
        awq_dump_path = getattr(args, "awq_dump_scales", None)
        awq_cache_path = getattr(args, "awq_cache_scales", None)
        if awq_load_path:
            hf_model_for_awq = getattr(model, "model", model)
            _awq_load_and_apply(
                hf_model_for_awq,
                awq_load_path,
                apply_clips=bool(runtime["awq_apply_clip"]),
            )
        elif awq_cache_path:
            cache_path = Path(str(awq_cache_path))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
            with lock_path.open("w", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    hf_model_for_awq = getattr(model, "model", model)
                    _awq_load_and_apply(
                        hf_model_for_awq,
                        str(cache_path),
                        apply_clips=bool(runtime["awq_apply_clip"]),
                    )
                else:
                    hf_model_for_awq = getattr(model, "model", model)
                    _awq_register_decoder_layers(hf_model_for_awq)
                    _AWQ_SCALE_LOG.clear()
                    _AWQ_CLIP_LOG.clear()
                    _awq_install_recording_hooks()
                    _apply_awq_preprocess(
                        qmodel=model,
                        calibration_blocks=calib_blocks,
                        bits=int(args.bits),
                        group_size=int(args.group_size),
                        sym=bool(runtime["sym"]),
                        apply_clip=bool(runtime["awq_apply_clip"]),
                        batch_size=int(args.awq_batch_size),
                        stop_after_layer=stop_after_layer,
                    )
                    _awq_dump_state(str(cache_path))
        else:
            if awq_dump_path:
                hf_model_for_awq = getattr(model, "model", model)
                _awq_register_decoder_layers(hf_model_for_awq)
                _AWQ_SCALE_LOG.clear()
                _AWQ_CLIP_LOG.clear()
                _awq_install_recording_hooks()
            _apply_awq_preprocess(
                qmodel=model,
                calibration_blocks=calib_blocks,
                bits=int(args.bits),
                group_size=int(args.group_size),
                sym=bool(runtime["sym"]),
                apply_clip=bool(runtime["awq_apply_clip"]),
                batch_size=int(args.awq_batch_size),
                stop_after_layer=stop_after_layer,
            )
            if awq_dump_path:
                _awq_dump_state(awq_dump_path)
        _cuda_synchronize_all(tracked_cuda_devices)
        awq_time_s = float(time.time() - t_awq)
        awq_peak_memory = _snapshot_cuda_peak_memory(tracked_cuda_devices)
        awq_artifact = getattr(args, "awq_cache_scales", None) or getattr(args, "awq_load_scales", None)
        if awq_artifact and Path(str(awq_artifact)).exists():
            awq_path = Path(str(awq_artifact))
            awq_fingerprint = {
                "protocol": "file_bytes_sha256_v1",
                "sha256": _file_sha256(awq_path),
                "bytes": awq_path.stat().st_size,
            }
    gc.collect()
    if tracked_cuda_devices:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _reset_cuda_peak_memory_stats(tracked_cuda_devices)
    _cuda_synchronize_all(tracked_cuda_devices)
    t_quant = time.time()
    original_callback = getattr(model, "layer_callback", None)
    if stop_after_layer is not None:
        model.layer_callback = _StopAfterLayer(stop_after_layer)
    try:
        model.quantize(calib_blocks)
    finally:
        model.layer_callback = original_callback
    _cuda_synchronize_all(tracked_cuda_devices)
    quant_time = float(time.time() - t_quant)
    quant_log_summary = _collect_quant_log_summary(model)
    quant_region_totals = _collect_quant_region_totals(model)
    mechanism_summary = _collect_mpq_mechanism(qcfg)
    matrix_fingerprints = _collect_matrix_fingerprints(qcfg)
    quant_peak_memory = _snapshot_cuda_peak_memory(tracked_cuda_devices)

    skip_save = bool(getattr(args, "skip_save", False))
    export_dense_hf = bool(getattr(args, "export_dense_hf", False))
    quantized_dir = Path(args.output_dir) / "quantized"
    hf_dir = Path(args.output_dir) / "hf"
    save_quantized_s = 0.0
    hfize_s = 0.0
    # When offload_to_disk was used during quant, the in-memory model has meta
    # tensors. accelerate.dispatch_model can't always materialize them from
    # GPTQModel's offload format, so when we're going to evaluate we force a
    # save+reload cycle.
    _used_offload = bool(getattr(qcfg, "offload_to_disk", False))
    _force_save_for_eval = (
        _used_offload
        and not bool(getattr(args, "skip_eval", False))
    )
    if not skip_save or _force_save_for_eval:
        quantized_dir.mkdir(parents=True, exist_ok=True)
        if export_dense_hf:
            hf_dir.mkdir(parents=True, exist_ok=True)
        t_save = time.time()
        model.save(str(quantized_dir))
        tokenizer.save_pretrained(str(quantized_dir))
        save_quantized_s = float(time.time() - t_save)

    # Release large quantization intermediates before evaluation.
    del calib_blocks

    # Best-effort: clear optional workspace caches (may retain large buffers).
    if importlib.util.find_spec("gptqmodel.quantization.gptq") is not None:
        gptq_mod = importlib.import_module("gptqmodel.quantization.gptq")
        workspace_cache = getattr(gptq_mod, "_WORKSPACE_CACHE", None)
        if hasattr(workspace_cache, "clear"):
            workspace_cache.clear()

    skip_eval = bool(getattr(args, "skip_eval", False))
    eval_datasets = list(getattr(args, "eval_datasets", ["wikitext2", "c4"]))
    ppls: Dict[str, float] = {}
    eval_times: Dict[str, float] = {}
    eval_protocol: Dict[str, Any] = {}
    eval_token_hashes: Dict[str, Any] = {}
    eval_cache_dir = str(args.eval_cache_dir or args.output_dir)

    hf_model = getattr(model, "model", model)
    hf_model.eval()
    # If we forced a save (offload_to_disk path), materialize via reload from
    # disk. The original `model` object's meta tensors can't be `.to(device)`'d
    # cleanly. Reload with device_map="auto" to spread quantized weights across
    # available GPUs.
    if _force_save_for_eval:
        from transformers import AutoModelForCausalLM
        del hf_model
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Free the offload dir immediately to make room for the reload.
        offload_path_str = getattr(qcfg, "offload_to_disk_path", None)
        if offload_path_str:
            import shutil
            shutil.rmtree(offload_path_str, ignore_errors=True)
        # Patch the saved quantize_config.json to disable broken triton kernel and avoid
        # act_group_aware/desc_act conflict before HF loads it.
        try:
            import json as _json
            qcfg_path = Path(str(quantized_dir)) / "quantize_config.json"
            if qcfg_path.exists():
                _saved = _json.loads(qcfg_path.read_text())
                _saved["act_group_aware"] = False
                _saved["backend"] = "torch"
                qcfg_path.write_text(_json.dumps(_saved, indent=2))
                print(f"[eval] patched {qcfg_path} (act_group_aware=False, backend=torch)", flush=True)
        except Exception as _e:
            print(f"[eval] config patch failed: {_e}", flush=True)
        from transformers import GPTQConfig as _HFGPTQ
        _override_qc = _HFGPTQ(
            bits=int(args.bits),
            group_size=int(args.group_size),
            desc_act=True,
            backend="torch",
        )
        try:
            _override_qc.act_group_aware = False
        except Exception:
            pass
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(quantized_dir),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=_override_qc,
        )
        hf_model.eval()
        print(f"[eval] reloaded quantized model from {quantized_dir} with device_map='auto' (backend=torch)", flush=True)
    else:
        hf_model.to(device)
        _strip_gptqmodel_hooked_modules(hf_model)
        hf_model.to(device)
        _materialize_inference_tensors(hf_model)

    if not skip_eval:
        for dataset in eval_datasets:
            dataset_norm = str(dataset).strip().lower()
            eval_protocol[dataset_norm] = eval_protocol_for_dataset(
                dataset_norm,
                seqlen=int(args.eval_seqlen),
                requested_max_eval_tokens=int(args.eval_max_eval_tokens),
            )
            eval_json = Path(args.output_dir) / f"eval_{dataset_norm}.json"
            t_eval = time.time()
            if dataset_norm == "wikitext2":
                tokens = _load_or_build_wikitext2_tokens(
                    tokenizer,
                    max_eval_tokens=int(args.eval_max_eval_tokens),
                    seqlen=int(args.eval_seqlen),
                    seed=eval_seed,
                    output_dir=eval_cache_dir,
                )
            elif dataset_norm == "c4":
                tokens = _load_or_build_c4_tokens(
                    tokenizer,
                    model_ref=str(model_path),
                    max_eval_tokens=int(args.eval_max_eval_tokens),
                    seqlen=int(args.eval_seqlen),
                    seed=eval_seed,
                    output_dir=eval_cache_dir,
                )
            else:
                raise ValueError(f"unsupported eval dataset: {dataset_norm!r}")
            eval_token_hashes[dataset_norm] = eval_token_fingerprint(tokens)
            dataset_ppl = float(eval_ppl_from_tokens(hf_model, tokens, seqlen=int(args.eval_seqlen)))
            ppls[dataset_norm] = dataset_ppl
            eval_times[f"eval_{dataset_norm}_s"] = float(time.time() - t_eval)
            eval_payload = {
                "datasets": [dataset_norm],
                "eval_protocol": {dataset_norm: eval_protocol[dataset_norm]},
                "max_eval_tokens": int(args.eval_max_eval_tokens),
                "ppl": {dataset_norm: dataset_ppl},
                "seed": eval_seed,
                "seqlen": int(args.eval_seqlen),
                "token_fingerprint": eval_token_hashes[dataset_norm],
            }
            eval_json.write_text(json.dumps(eval_payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        eval_datasets = []

    lm_eval_tasks = list(getattr(args, "lm_eval_tasks", None) or [])
    lm_eval_result = {}
    lm_eval_time_s = 0.0
    if lm_eval_tasks and not skip_eval:
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        t_lm = time.time()
        lm_eval_result = _run_lm_eval(
            hf_model, tokenizer,
            tasks=lm_eval_tasks,
            batch_size=getattr(args, "lm_eval_batch_size", "1"),
            num_fewshot=int(getattr(args, "lm_eval_num_fewshot", 0)),
            include_path=getattr(args, "lm_eval_include_path", None),
        )
        lm_eval_time_s = float(time.time() - t_lm)

    if not skip_save and export_dense_hf:
        t_hfize = time.time()
        GPTQModel.export(str(quantized_dir), str(hf_dir), format="hf")
        hfize_s = float(time.time() - t_hfize)

    if "model" in locals():
        del model
    if "hf_model" in locals():
        del hf_model
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "backend": "gptqmodel",
        "requested_method": runtime["requested_method"],
        "request_source": str(runtime["request_source"]),
        "resolved_method": "ojbkq" if bool(runtime["ojbkq"]) else (combo.solver if combo is not None else "gptq"),
        "baseline_family": "ojbkq" if bool(runtime["ojbkq"]) else ("gptq" if combo is None or combo.solver == "gptq" else ("mpq" if combo.solver == "vamp" else combo.solver)),
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "sym": bool(runtime["sym"]),
        "codebook": combo.codebook if combo is not None else "grid_g128",
        "preprocess": combo.preprocess if combo is not None else "none",
        "rotation": combo.rotation if combo is not None else "none",
        "rotation_mode": (
            "internal_rotation"
            if bool(getattr(getattr(qcfg, "mpq", None), "rotation_internal", False))
            else ("model_rotation" if (combo is not None and combo.rotation in {"model_hadamard", "model_random"}) else "none")
        ),
        "rotation_requested": str(getattr(getattr(qcfg, "mpq", None), "rotation_requested", combo.rotation if combo is not None else "none")),
        "rotation_effective": str(getattr(getattr(qcfg, "mpq", None), "rotation_effective", combo.rotation if combo is not None else "none")),
        "solver": combo.solver if combo is not None else "gptq",
        "polish_mode": combo.polish_mode if combo is not None else "none",
        "topk_source": combo.topk_source if combo is not None else "aux_vamp",
        "desc_act": bool(getattr(qcfg, "desc_act", runtime["desc_act"])),
        "seed": int(args.seed),
        "seeds": {
            "solver": int(args.seed),
            "calibration": calibration_seed,
            "rotation": int(args.mpq_rotation_seed),
            "eval": eval_seed,
        },
        "model_id": model_id,
        "model_source": model_source,
        "model_resolved_path": model_path,
        "run_scope": {
            "partial": bool(stop_after_layer is not None),
            "stop_after_layer": stop_after_layer,
        },
        "awq_preprocess": {
            "enabled": bool(runtime["awq_preprocess"]),
            "apply_clip": bool(runtime["awq_apply_clip"]),
            "batch_size": int(args.awq_batch_size),
            "time_s": float(awq_time_s),
            "cache_scales": str(getattr(args, "awq_cache_scales", "") or ""),
            "fingerprint": awq_fingerprint,
        },
        "calibration": {
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "limit": int(args.calibration_limit),
            "seqlen": int(args.calibration_seqlen),
            "draw": calibration_draw,
            "seed": calibration_seed,
            "fingerprint": calibration_manifest,
            "cache_dir": str(args.calibration_cache_dir or args.output_dir),
        },
        "eval": {
            "datasets": eval_datasets,
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(args.eval_max_eval_tokens),
            "eval_protocol": eval_protocol,
            "seed": eval_seed,
            "token_fingerprints": eval_token_hashes,
            "ppl": ppls,
        },
        "lm_eval": lm_eval_result,
        "timing": {
            "awq_preprocess_s": float(awq_time_s),
            "load_s": float(load_time_s),
            "quantize_s": float(quant_time),
            "solver_module_sum_s": float(quant_log_summary.get("solver_module_sum_s", 0.0)),
            "solver_module_max_s": float(quant_log_summary.get("solver_module_max_s", 0.0)),
            "save_quantized_s": float(save_quantized_s),
            "hfize_s": float(hfize_s),
            **quant_region_totals,
            **eval_times,
            "lm_eval_s": float(lm_eval_time_s),
            "eval_total_s": float(sum(eval_times.values()) + lm_eval_time_s),
            "quantize_total_s": float(load_time_s + awq_time_s + quant_time + save_quantized_s + hfize_s),
            "total_s": float(load_time_s + awq_time_s + quant_time + save_quantized_s + hfize_s + sum(eval_times.values()) + lm_eval_time_s),
        },
        "peak_memory": {
            "awq_preprocess": awq_peak_memory,
            "quantize": quant_peak_memory,
        },
        "solver_profile": {
            "modules": int(quant_log_summary.get("modules", 0)),
        },
        "mechanism": mechanism_summary,
        "matrix_fingerprints": matrix_fingerprints,
        "paths": {
            "native_dir": str(quantized_dir) if not skip_save else "",
            "hf_dir": str(hf_dir if export_dense_hf else quantized_dir) if not skip_save else "",
            "dense_hf_dir": str(hf_dir) if not skip_save and export_dense_hf else "",
            "quantized_dir": str(quantized_dir) if not skip_save else "",
        },
        "artifact_format": {
            "native": "gptqmodel",
            "hf": "huggingface_dense" if export_dense_hf else "gptqmodel",
        },
        "quantization_path": {
            "dense": bool(
                getattr(qcfg, "mpq", None)
                and bool(getattr(qcfg.mpq, "enabled", False))
                and bool(getattr(qcfg.mpq, "rotation_internal", False))
                and skip_save
            ),
            "packed": bool(not skip_save),
        },
        "meta": {
            "protocol_version": str(args.protocol_version),
            "repo_sha": _git_head(REPO_ROOT),
            "third_party": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "fast-hadamard-transform": _git_head(
                    REPO_ROOT / "third_party" / "fast-hadamard-transform"
                ),
            },
        },
        "mpq": {
            "enabled": bool(getattr(qcfg, "mpq", None) and getattr(qcfg.mpq, "enabled", False)),
            "solver": str(getattr(qcfg.mpq, "solver", "")) if getattr(qcfg, "mpq", None) else "",
            "init": str(getattr(qcfg.mpq, "init", "")) if getattr(qcfg, "mpq", None) else "",
            "polish_mode": str(getattr(qcfg.mpq, "polish_mode", "")) if getattr(qcfg, "mpq", None) else "",
            "topk_source": str(getattr(qcfg.mpq, "topk_source", "")) if getattr(qcfg, "mpq", None) else "",
            "topk_k": int(getattr(qcfg.mpq, "topk_k", 0)) if getattr(qcfg, "mpq", None) else 0,
            "iters": int(getattr(qcfg.mpq, "iters", 0)) if getattr(qcfg, "mpq", None) else 0,
            "prior_tau": float(getattr(qcfg.mpq, "prior_tau", 0.0)) if getattr(qcfg, "mpq", None) else 0.0,
            "damping": float(getattr(qcfg.mpq, "damping", 0.0)) if getattr(qcfg, "mpq", None) else 0.0,
            "vamp_backend": str(getattr(qcfg.mpq, "vamp_backend", "")) if getattr(qcfg, "mpq", None) else "",
            "polish_sweeps": int(getattr(qcfg.mpq, "polish_sweeps", 0)) if getattr(qcfg, "mpq", None) else 0,
            "compile_block_cd": bool(getattr(qcfg.mpq, "compile_block_cd", False)) if getattr(qcfg, "mpq", None) else False,
            "block_cd_backend": str(getattr(qcfg.mpq, "block_cd_backend", "")) if getattr(qcfg, "mpq", None) else "",
            "em_obs_precision": bool(getattr(qcfg.mpq, "em_obs_precision", False)) if getattr(qcfg, "mpq", None) else False,
            "block_rows": int(getattr(qcfg.mpq, "block_rows", 0)) if getattr(qcfg, "mpq", None) else 0,
            "rotation_seed": int(getattr(qcfg.mpq, "rotation_seed", 0)) if getattr(qcfg, "mpq", None) else 0,
            "rotation_mode": str(getattr(qcfg.mpq, "rotation_mode", "")) if getattr(qcfg, "mpq", None) else "",
            "rotation_requested": str(getattr(qcfg.mpq, "rotation_requested", "")) if getattr(qcfg, "mpq", None) else "",
            "rotation_effective": str(getattr(qcfg.mpq, "rotation_effective", "")) if getattr(qcfg, "mpq", None) else "",
            "rotation_internal": bool(getattr(qcfg.mpq, "rotation_internal", False)) if getattr(qcfg, "mpq", None) else False,
            "rotation_fallback_reason": str(getattr(qcfg.mpq, "rotation_fallback_reason", "")) if getattr(qcfg, "mpq", None) else "",
            "rotation_block_dim": int(getattr(qcfg.mpq, "rotation_block_dim", 0)) if getattr(qcfg, "mpq", None) else 0,
            "paired_matrix_cache_dir": str(getattr(qcfg.mpq, "paired_matrix_cache_dir", "") or "") if getattr(qcfg, "mpq", None) else "",
            "layer_stats": getattr(qcfg.mpq, "_layer_stats", []) if getattr(qcfg, "mpq", None) else [],
        },
        "ojbkq": {
            "enabled": bool(getattr(qcfg, "ojbkq", None) is not None and bool(getattr(qcfg.ojbkq, "enabled", False))),
            "random_trials": int(getattr(qcfg.ojbkq, "random_trials", 0)) if getattr(qcfg, "ojbkq", None) else 0,
            "total_candidates": 1 + int(getattr(qcfg.ojbkq, "random_trials", 0)) if getattr(qcfg, "ojbkq", None) else 0,
            "paper_ojbkq": bool(
                getattr(qcfg, "ojbkq", None)
                and int(getattr(qcfg.ojbkq, "random_trials", 0)) == 5
                and str(getattr(qcfg.ojbkq, "alpha_mode", "")) == "paper"
            ),
            "greedy_initialization": "gptq",
            "alpha_mode": str(getattr(qcfg.ojbkq, "alpha_mode", "")) if getattr(qcfg, "ojbkq", None) else "",
            "alpha_override": getattr(qcfg.ojbkq, "alpha_override", None) if getattr(qcfg, "ojbkq", None) else None,
            "alpha_scale": float(getattr(qcfg.ojbkq, "alpha_scale", 0.0)) if getattr(qcfg, "ojbkq", None) else 0.0,
            "sample_topk": getattr(qcfg.ojbkq, "sample_topk", None) if getattr(qcfg, "ojbkq", None) else None,
            "layer_stats": getattr(qcfg.ojbkq, "_layer_stats", []) if getattr(qcfg, "ojbkq", None) else [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default=None,
        choices=["gptq", "gptq-strong", "mpq", "rtn", "cdquant", "quantease", "ojbkq", "awq+gptq", "awq+mpq"],
    )
    parser.add_argument("--quant-backend", type=str, default="gptqmodel")
    parser.add_argument("--quant-codebook", type=str, default="grid_g128")
    parser.add_argument("--quant-preprocess", type=str, default="none")
    parser.add_argument(
        "--quant-rotation",
        type=str,
        default="auto",
        help="GPTQModel rotation mode: auto|none|model_hadamard|model_random|internal_hadamard.",
    )
    parser.add_argument("--quant-solver", type=str, default="gptq", choices=["gptq", "vamp", "cdquant", "quantease", "ojbkq"])
    parser.add_argument("--quant-polish-mode", type=str, default="none", choices=["none", "block_cd", "topk_block_cd"])
    parser.add_argument("--quant-topk-source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", type=int, default=16)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--model-id-label",
        default=None,
        help="Canonical model id recorded in results when --model-id is a local path or mirror.",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--desc-act",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override activation ordering for protocol-matched solver comparisons.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # Calibration (Gate A defaults)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--calibration-seed", type=int, default=None)
    parser.add_argument("--calibration-draw", type=int, default=None)
    parser.add_argument("--calibration-cache-dir", type=str, default=None)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=0, help="0 => paper-style defaults: full Wikitext2, C4 = 256*seqlen")
    parser.add_argument("--eval-datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-cache-dir", type=str, default=None)
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--export-dense-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export a dequantized Hugging Face copy; the native GPTQModel artifact is saved by default.",
    )

    # lm-evaluation-harness
    parser.add_argument("--lm-eval-tasks", nargs="*", default=None)
    parser.add_argument("--lm-eval-batch-size", default="1")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=0)
    parser.add_argument("--lm-eval-include-path", default=None,
                        help="extra task yaml dir for lm-eval TaskManager (e.g. custom gpqa_main_mc_local)")

    # AWQ preprocess
    parser.add_argument("--awq-batch-size", type=int, default=4)
    parser.add_argument("--awq-apply-clip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--awq-dump-scales",
        type=str,
        default=None,
        help="Dump AWQ apply_scale/apply_clip args (per decoder layer) to this .pt file after AWQ preprocess runs.",
    )
    parser.add_argument(
        "--awq-load-scales",
        type=str,
        default=None,
        help="Skip AWQ scale search. Instead load saved scales/clips from this .pt file and apply them directly.",
    )
    parser.add_argument(
        "--awq-cache-scales",
        type=str,
        default=None,
        help="Shared AWQ scale/cache file. If missing, compute and dump under a file lock; otherwise load it.",
    )

    # MPQ hyperparams (Gate A defaults)
    parser.add_argument("--mpq-iters", type=int, default=10)
    parser.add_argument("--mpq-damping", type=float, default=0.5)
    parser.add_argument("--mpq-prior-tau", type=float, default=0.2)
    parser.add_argument(
        "--mpq-vamp-backend",
        type=str,
        default="auto",
        choices=["eager", "compile", "triton", "auto"],
    )
    parser.add_argument("--mpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpq-compile-block-cd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mpq-block-cd-backend",
        type=str,
        default=None,
        choices=["eager", "compile", "triton", "auto"],
    )
    parser.add_argument("--mpq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-block-rows", type=int, default=128)
    parser.add_argument("--mpq-rotation-seed", type=int, default=0)
    parser.add_argument("--paired-matrix-cache-dir", type=str, default=None)
    parser.add_argument("--mpq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpq-obs-precision", type=float, default=0.0)
    parser.add_argument("--mpq-init", type=str, default="float", choices=["float", "random", "rtn", "ldlq"])

    # OJBKQ baseline hyperparams
    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, required=True)
    parser.add_argument(
        "--gptqmodel-extra-site-packages",
        type=str,
        default=None,
        help="Optional extra site-packages path to satisfy GPTQModel deps when running without installing GPTQModel.",
    )
    parser.add_argument("--awq-source", type=str, default="gptqmodel")
    parser.add_argument("--gptq-env", type=str, default="mpq-gptq")
    parser.add_argument("--protocol-version", type=str, default="legacy")
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        default=None,
        help="Inclusive layer index for early-stop diagnostic runs. Applies to AWQ preprocess and quantization.",
    )

    args = parser.parse_args()

    result = run(args)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
