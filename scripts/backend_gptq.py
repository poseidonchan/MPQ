from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.combo import QuantCombo, make_combo


def _resolve_device_str(requested: Optional[str]) -> str:
    """Resolve requested device without probing CUDA availability."""

    if requested:
        return str(requested)
    # Default to CUDA for benchmarking; callers can set `--device cpu` explicitly.
    return "cuda:0"


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
        raise RuntimeError(proc.stdout)
    return proc.stdout.strip()


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


def _ensure_gptqmodel_importable(*, extra_site_packages: Optional[str]) -> None:
    """Ensure `import gptqmodel` works even if not installed in the active env.

    This is helpful when the active env has a working CUDA stack (e.g. QUIP)
    but the GPTQModel deps live in a different env. We always prefer the repo
    submodule for `gptqmodel` so patch application stays authoritative.
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


def _apply_awq_preprocess(
    *,
    qmodel,
    calibration_blocks,
    bits: int,
    group_size: int,
    sym: bool,
    apply_clip: bool,
    batch_size: int,
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
    looper.loop(backend=BACKEND, failsafe=awq_qcfg.failsafe)


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
    cache_path = _eval_cache_path(output_dir, tokenizer, "wikitext2", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[:, :max_eval_tokens].cpu().contiguous()
    torch.save(tokens, cache_path)
    return tokens


def _load_or_build_c4_tokens(tokenizer, *, model_ref: str, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    cache_path = _eval_cache_path(output_dir, tokenizer, "c4", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "gptq_data_utils",
        REPO_ROOT / "third_party" / "quip-sharp" / "lib" / "utils" / "gptq_data_utils.py",
    )
    gptq_data_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(gptq_data_utils)

    tokens = gptq_data_utils.get_test_tokens("c4", seed=int(seed), seqlen=int(seqlen), model=str(model_ref))
    tokens = tokens[:, :max_eval_tokens].cpu().contiguous()
    torch.save(tokens, cache_path)
    return tokens


@torch.no_grad()
def eval_ppl_from_tokens(
    model,
    tokens: torch.Tensor,
    *,
    seqlen: int = 2048,
) -> float:
    runtime_device = next(model.parameters()).device
    tokens = tokens.to(runtime_device)

    nlls = []
    nsamples = tokens.shape[1] // int(seqlen)
    for i in range(nsamples):
        batch = tokens[:, i * seqlen : (i + 1) * seqlen]
        # Disable KV cache to keep peak memory bounded during long-run eval.
        outputs = model(batch, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * (int(seqlen) - 1))

    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * (int(seqlen) - 1)
    return float(torch.exp(total_nll / total_tokens).item())


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
                rotation="hadamard",
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
                rotation="hadamard",
                solver="vamp",
                polish_mode="block_cd" if int(args.mpq_polish_sweeps) > 0 else "none",
            )
            return combo, runtime
        if requested_method == "rtn":
            runtime["desc_act"] = False
            runtime["legacy_rtn"] = True
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="none",
                rotation="hadamard",
                solver="vamp",
                polish_mode="none",
            )
            return combo, runtime
        if requested_method == "awq+gptq":
            runtime["awq_preprocess"] = True
            runtime["awq_apply_clip"] = True
            runtime["sym"] = False
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="awq",
                rotation="hadamard",
                solver="gptq",
                polish_mode="none",
            )
            return combo, runtime
        if requested_method == "awq+mpq":
            runtime["awq_preprocess"] = True
            runtime["awq_apply_clip"] = False
            runtime["sym"] = False
            runtime["desc_act"] = False
            combo = make_combo(
                backend="gptqmodel",
                codebook="grid_g128",
                preprocess="awq",
                rotation="hadamard",
                solver="vamp",
                polish_mode="block_cd" if int(args.mpq_polish_sweeps) > 0 else "none",
            )
            return combo, runtime
        if requested_method == "ojbkq":
            runtime["ojbkq"] = True
            runtime["desc_act"] = False
            return None, runtime
        raise ValueError(f"unsupported method={requested_method!r} for backend_gptq")

    combo = make_combo(
        backend=str(getattr(args, "quant_backend", "gptqmodel")),
        codebook=str(getattr(args, "quant_codebook", "grid_g128")),
        preprocess=str(getattr(args, "quant_preprocess", "none")),
        rotation=str(getattr(args, "quant_rotation", "hadamard")),
        solver=str(getattr(args, "quant_solver", "gptq")),
        polish_mode=str(getattr(args, "quant_polish_mode", "none")),
        topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
    )
    runtime["awq_preprocess"] = combo.preprocess == "awq"
    runtime["awq_apply_clip"] = combo.preprocess == "awq" and combo.solver == "gptq"
    runtime["sym"] = False if combo.preprocess == "awq" else bool(args.sym)
    runtime["desc_act"] = combo.solver == "gptq"
    return combo, runtime


def _build_quantize_config(combo: Optional[QuantCombo], runtime: Dict[str, Any], args) -> Any:
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization import FORMAT

    device = _resolve_device_str(getattr(args, "device", None))
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
        rotation=None if combo is None or combo.rotation in {"none", ""} else str(combo.rotation),
        desc_act=bool(runtime["desc_act"]),
    )

    use_repo_managed = combo is not None and (combo.solver == "vamp" or combo.polish_mode != "none")
    if use_repo_managed:
        qcfg.mpq.enabled = True
        qcfg.mpq.solver = str(combo.solver)
        qcfg.mpq.polish_mode = str(combo.polish_mode)
        qcfg.mpq.topk_source = str(combo.topk_source)
        qcfg.mpq.topk_k = int(getattr(args, "quant_topk_k", 16))
        qcfg.mpq.iters = 0 if bool(runtime["legacy_rtn"]) else int(args.mpq_iters)
        qcfg.mpq.damping = float(args.mpq_damping)
        qcfg.mpq.prior_tau = 0.0 if bool(runtime["legacy_rtn"]) else float(args.mpq_prior_tau)
        qcfg.mpq.polish_sweeps = 0 if bool(runtime["legacy_rtn"]) else int(args.mpq_polish_sweeps)
        qcfg.mpq.em_obs_precision = False if bool(runtime["legacy_rtn"]) else bool(args.mpq_em_obs_precision)
        qcfg.mpq.block_rows = int(args.mpq_block_rows)
        qcfg.mpq.rotation_seed = int(args.mpq_rotation_seed)
        qcfg.mpq.gamma_start = float(args.mpq_gamma_start)
        qcfg.mpq.gamma_retry_factor = float(args.mpq_gamma_retry_factor)
        qcfg.mpq.gamma_retry_steps = int(args.mpq_gamma_retry_steps)
        qcfg.mpq.obs_precision = float(args.mpq_obs_precision)
    else:
        qcfg.mpq.enabled = False

    if bool(runtime["ojbkq"]):
        from mpq.baselines.ojbkq import OJBKQConfig

        qcfg.desc_act = False
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


def _run_lm_eval(hf_model, tokenizer, *, tasks, batch_size, num_fewshot):
    """Run lm-evaluation-harness tasks in-process."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size)
    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    eval_results = simple_evaluate(
        model=lm, tasks=tasks, batch_size=batch_size, num_fewshot=num_fewshot,
        confirm_run_unsafe_code=True,
    )
    results = {}
    for task_name, task_metrics in eval_results.get("results", {}).items():
        results[task_name] = {
            k: v for k, v in task_metrics.items() if k != "alias"
        }
    return {"tasks": list(tasks), "results": results}


def run(args) -> Dict[str, Any]:
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

    device = torch.device(str(args.device))
    awq_time_s = 0.0

    model_id = str(args.model_id)
    model_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    from mpq.data import load_or_build_calibration_blocks

    calib_blocks = load_or_build_calibration_blocks(
        tokenizer=tokenizer,
        dataset_name=str(args.calibration_dataset),
        split=str(args.calibration_split),
        limit=int(args.calibration_limit),
        seqlen=int(args.calibration_seqlen),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        log_fn=lambda s: print(s, flush=True),
    )

    qcfg = _build_quantize_config(combo, runtime, args)
    t_load = time.time()
    model = GPTQModel.load(model_path, qcfg)
    load_time_s = float(time.time() - t_load)
    if bool(runtime["awq_preprocess"]):
        t_awq = time.time()
        _apply_awq_preprocess(
            qmodel=model,
            calibration_blocks=calib_blocks,
            bits=int(args.bits),
            group_size=int(args.group_size),
            sym=bool(runtime["sym"]),
            apply_clip=bool(runtime["awq_apply_clip"]),
            batch_size=int(args.awq_batch_size),
        )
        awq_time_s = float(time.time() - t_awq)
    t_quant = time.time()
    model.quantize(calib_blocks)
    quant_time = float(time.time() - t_quant)

    quantized_dir = Path(args.output_dir) / "quantized"
    hf_dir = Path(args.output_dir) / "hf"
    quantized_dir.mkdir(parents=True, exist_ok=True)
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
    if not skip_eval:
        # Evaluate in-process.
        hf_model = getattr(model, "model", model)
        # Drop wrapper refs (may retain large buffers).
        if hf_model is not model:
            del model
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        hf_model.eval()
        hf_model.to(device)
        _strip_gptqmodel_hooked_modules(hf_model)
        hf_model.to(device)
        _materialize_inference_tensors(hf_model)
        for dataset in eval_datasets:
            dataset_norm = str(dataset).strip().lower()
            t_eval = time.time()
            if dataset_norm == "wikitext2":
                tokens = _load_or_build_wikitext2_tokens(
                    tokenizer,
                    max_eval_tokens=int(args.eval_max_eval_tokens),
                    seqlen=int(args.eval_seqlen),
                    seed=int(args.seed),
                    output_dir=str(args.output_dir),
                )
            elif dataset_norm == "c4":
                tokens = _load_or_build_c4_tokens(
                    tokenizer,
                    model_ref=str(model_path),
                    max_eval_tokens=int(args.eval_max_eval_tokens),
                    seqlen=int(args.eval_seqlen),
                    seed=int(args.seed),
                    output_dir=str(args.output_dir),
                )
            else:
                raise ValueError(f"unsupported eval dataset for backend_gptq: {dataset!r}")
            ppls[dataset_norm] = eval_ppl_from_tokens(
                hf_model,
                tokens,
                seqlen=int(args.eval_seqlen),
            )
            eval_times[f"eval_{dataset_norm}_s"] = float(time.time() - t_eval)
    else:
        eval_datasets = []

    lm_eval_tasks = list(getattr(args, "lm_eval_tasks", None) or [])
    lm_eval_result = {}
    lm_eval_time_s = 0.0
    if lm_eval_tasks and not skip_eval:
        t_lm = time.time()
        lm_eval_result = _run_lm_eval(
            hf_model, tokenizer,
            tasks=lm_eval_tasks,
            batch_size=getattr(args, "lm_eval_batch_size", "1"),
            num_fewshot=int(getattr(args, "lm_eval_num_fewshot", 0)),
        )
        lm_eval_time_s = float(time.time() - t_lm)

    if "hf_model" in locals():
        del hf_model
    if "model" in locals():
        del model
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    t_hfize = time.time()
    GPTQModel.export(str(quantized_dir), str(hf_dir), format="hf")
    hfize_s = float(time.time() - t_hfize)

    return {
        "backend": "gptqmodel",
        "requested_method": runtime["requested_method"],
        "request_source": str(runtime["request_source"]),
        "resolved_method": "ojbkq" if bool(runtime["ojbkq"]) else (combo.solver if combo is not None else "gptq"),
        "baseline_family": "ojbkq" if bool(runtime["ojbkq"]) else ("gptq" if combo is None or combo.solver == "gptq" else "mpq"),
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "sym": bool(runtime["sym"]),
        "codebook": combo.codebook if combo is not None else "grid_g128",
        "preprocess": combo.preprocess if combo is not None else "none",
        "rotation": combo.rotation if combo is not None else "none",
        "solver": combo.solver if combo is not None else "gptq",
        "polish_mode": combo.polish_mode if combo is not None else "none",
        "topk_source": combo.topk_source if combo is not None else "aux_vamp",
        "desc_act": bool(getattr(qcfg, "desc_act", runtime["desc_act"])),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "awq_preprocess": {
            "enabled": bool(runtime["awq_preprocess"]),
            "apply_clip": bool(runtime["awq_apply_clip"]),
            "batch_size": int(args.awq_batch_size),
            "time_s": float(awq_time_s),
        },
        "calibration": {
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "limit": int(args.calibration_limit),
            "seqlen": int(args.calibration_seqlen),
        },
        "eval": {
            "datasets": eval_datasets,
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(args.eval_max_eval_tokens),
            "ppl": ppls,
        },
        "lm_eval": lm_eval_result,
        "timing": {
            "awq_preprocess_s": float(awq_time_s),
            "load_s": float(load_time_s),
            "quantize_s": float(quant_time),
            "save_quantized_s": float(save_quantized_s),
            "hfize_s": float(hfize_s),
            **eval_times,
            "lm_eval_s": float(lm_eval_time_s),
            "eval_total_s": float(sum(eval_times.values()) + lm_eval_time_s),
            "quantize_total_s": float(load_time_s + awq_time_s + quant_time + save_quantized_s + hfize_s),
            "total_s": float(load_time_s + awq_time_s + quant_time + save_quantized_s + hfize_s + sum(eval_times.values()) + lm_eval_time_s),
        },
        "paths": {
            "native_dir": str(quantized_dir),
            "hf_dir": str(hf_dir),
            "quantized_dir": str(quantized_dir),
        },
        "artifact_format": {
            "native": "gptqmodel",
            "hf": "huggingface",
        },
        "quantization_path": {
            "dense": bool(
                getattr(qcfg, "mpq", None)
                and bool(getattr(qcfg.mpq, "enabled", False))
                and str(getattr(qcfg.mpq, "solver", "")) == "vamp"
                and str(getattr(qcfg, "rotation", None)) == "hadamard"
            ),
            "packed": not bool(
                getattr(qcfg, "mpq", None)
                and bool(getattr(qcfg.mpq, "enabled", False))
                and str(getattr(qcfg.mpq, "solver", "")) == "vamp"
                and str(getattr(qcfg, "rotation", None)) == "hadamard"
            ),
        },
        "meta": {
            "repo_sha": _git_head(REPO_ROOT),
            "submodules": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
        },
        "mpq": {
            "enabled": bool(getattr(qcfg, "mpq", None) and getattr(qcfg.mpq, "enabled", False)),
            "solver": str(getattr(qcfg.mpq, "solver", "")) if getattr(qcfg, "mpq", None) else "",
            "polish_mode": str(getattr(qcfg.mpq, "polish_mode", "")) if getattr(qcfg, "mpq", None) else "",
            "topk_source": str(getattr(qcfg.mpq, "topk_source", "")) if getattr(qcfg, "mpq", None) else "",
            "topk_k": int(getattr(qcfg.mpq, "topk_k", 0)) if getattr(qcfg, "mpq", None) else 0,
            "iters": int(getattr(qcfg.mpq, "iters", 0)) if getattr(qcfg, "mpq", None) else 0,
            "prior_tau": float(getattr(qcfg.mpq, "prior_tau", 0.0)) if getattr(qcfg, "mpq", None) else 0.0,
            "damping": float(getattr(qcfg.mpq, "damping", 0.0)) if getattr(qcfg, "mpq", None) else 0.0,
            "polish_sweeps": int(getattr(qcfg.mpq, "polish_sweeps", 0)) if getattr(qcfg, "mpq", None) else 0,
            "em_obs_precision": bool(getattr(qcfg.mpq, "em_obs_precision", False)) if getattr(qcfg, "mpq", None) else False,
            "block_rows": int(getattr(qcfg.mpq, "block_rows", 0)) if getattr(qcfg, "mpq", None) else 0,
            "rotation_seed": int(getattr(qcfg.mpq, "rotation_seed", 0)) if getattr(qcfg, "mpq", None) else 0,
        },
        "ojbkq": {
            "enabled": bool(getattr(qcfg, "ojbkq", None) is not None and bool(getattr(qcfg.ojbkq, "enabled", False))),
            "random_trials": int(getattr(qcfg.ojbkq, "random_trials", 0)) if getattr(qcfg, "ojbkq", None) else 0,
            "alpha_mode": str(getattr(qcfg.ojbkq, "alpha_mode", "")) if getattr(qcfg, "ojbkq", None) else "",
            "alpha_override": getattr(qcfg.ojbkq, "alpha_override", None) if getattr(qcfg, "ojbkq", None) else None,
            "alpha_scale": float(getattr(qcfg.ojbkq, "alpha_scale", 0.0)) if getattr(qcfg, "ojbkq", None) else 0.0,
            "sample_topk": getattr(qcfg.ojbkq, "sample_topk", None) if getattr(qcfg, "ojbkq", None) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default=None,
        choices=["gptq", "gptq-strong", "mpq", "rtn", "ojbkq", "awq+gptq", "awq+mpq"],
    )
    parser.add_argument("--quant-backend", type=str, default="gptqmodel")
    parser.add_argument("--quant-codebook", type=str, default="grid_g128")
    parser.add_argument("--quant-preprocess", type=str, default="none")
    parser.add_argument("--quant-rotation", type=str, default="hadamard")
    parser.add_argument("--quant-solver", type=str, default="gptq", choices=["gptq", "vamp"])
    parser.add_argument("--quant-polish-mode", type=str, default="none", choices=["none", "block_cd", "topk_block_cd"])
    parser.add_argument("--quant-topk-source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", type=int, default=16)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # Calibration (Gate A defaults)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)
    parser.add_argument("--eval-datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)

    # lm-evaluation-harness
    parser.add_argument("--lm-eval-tasks", nargs="*", default=None)
    parser.add_argument("--lm-eval-batch-size", default="1")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=0)

    # AWQ preprocess
    parser.add_argument("--awq-batch-size", type=int, default=1)

    # MPQ hyperparams (Gate A defaults)
    parser.add_argument("--mpq-iters", type=int, default=10)
    parser.add_argument("--mpq-damping", type=float, default=0.5)
    parser.add_argument("--mpq-prior-tau", type=float, default=0.2)
    parser.add_argument("--mpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-block-rows", type=int, default=128)
    parser.add_argument("--mpq-rotation-seed", type=int, default=0)
    parser.add_argument("--mpq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpq-obs-precision", type=float, default=0.0)

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

    args = parser.parse_args()

    result = run(args)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
