from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_via_runpy() -> None:
    """Re-exec via `python -c` + `runpy` to avoid CUDA init flakiness.

    In this environment, executing a Python *file* directly via `conda run ... python path/to/script.py`
    intermittently fails CUDA initialization with `cudaGetDeviceCount` error 304.
    Running the same entrypoint via `python -c` is significantly more reliable.

    We therefore re-exec ourselves into a `-c` launcher when `sys.path[0]` is a
    non-empty script directory (i.e. the direct-file execution path).
    """

    if sys.path and sys.path[0] == "":
        return

    script = str(Path(__file__).resolve())
    argv = [script, *sys.argv[1:]]
    code = (
        "import runpy, sys; "
        f"sys.argv = {json.dumps(argv)}; "
        f"runpy.run_path({json.dumps(script)}, run_name='__main__')"
    )
    os.execvpe(sys.executable, [sys.executable, "-c", code], os.environ)


def _resolve_device_str(requested: Optional[str]) -> str:
    """Resolve requested device without probing CUDA availability."""

    if requested:
        return str(requested)
    # Default to CUDA for benchmarking; callers can set `--device cpu` explicitly.
    return "cuda:0"


def _is_transient_cuda_init_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cudagetdevicecount" in msg and "304" in msg


def _preinit_cuda(device_str: str, *, retries: int = 60, sleep_s: float = 1.0) -> None:
    """Initialize CUDA before importing heavy deps (e.g. transformers).

    Some downstream imports probe CUDA; doing a tiny CUDA allocation up-front
    keeps those probes from putting CUDA into an unrecoverable error state.
    """

    if not str(device_str).startswith("cuda"):
        return

    last_exc: Optional[BaseException] = None
    for attempt in range(max(int(retries), 0) + 1):
        try:
            _ = torch.empty((0,), device=str(device_str))
            torch.cuda.synchronize()
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient_cuda_init_error(exc):
                raise
            if attempt >= max(int(retries), 0):
                break
            time.sleep(float(sleep_s))

    assert last_exc is not None
    raise RuntimeError("CUDA initialization failed after retries") from last_exc


def _setup_hf_offline_env() -> None:
    # This repo is frequently run in offline environments (no HF Hub access).
    # Force cache-only behavior unless a caller explicitly overrides these env vars.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # Redirect HF caches to a writable location (the default HF cache dir may be read-only in sandboxes).
    hf_home = os.environ.get("MPGQ_HF_HOME") or "/tmp/mpgq_hf"
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets"))


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
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("apply_patches.py failed")


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
def _eval_cache_path(output_dir: str, tokenizer, max_eval_tokens: int) -> str:
    cache_dir = os.path.join(output_dir, "_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(f"{tokenizer_id}|wikitext2|{max_eval_tokens}|eval_v1".encode("utf-8")).hexdigest()[:16]
    tokenizer_label = Path(tokenizer_id).name.replace("/", "_") or "tokenizer"
    return os.path.join(cache_dir, f"wikitext2_{tokenizer_label}_{max_eval_tokens}_{digest}.pt")


def _load_or_build_wikitext2_tokens(tokenizer, *, max_eval_tokens: int, output_dir: str) -> torch.Tensor:
    cache_path = _eval_cache_path(output_dir, tokenizer, max_eval_tokens)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[:, :max_eval_tokens].cpu().contiguous()
    torch.save(tokens, cache_path)
    return tokens


@torch.no_grad()
def eval_ppl_wikitext2(
    model,
    tokenizer,
    *,
    seqlen: int = 2048,
    max_eval_tokens: int = 131072,
    output_dir: str,
) -> float:
    runtime_device = next(model.parameters()).device
    tokens = _load_or_build_wikitext2_tokens(tokenizer, max_eval_tokens=int(max_eval_tokens), output_dir=str(output_dir))
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


def _setup_hf_cache(*, model_id: str, calibration_dataset: str) -> None:
    """Reuse existing read-only HF caches by mirroring them into the writable cache.

    In this sandbox we can *read* from `~/.cache/huggingface` but may not be able to
    *write* lock files there. We therefore:
    - Use `HF_*_CACHE` under `MPGQ_HF_HOME` (defaults to `/tmp/mpgq_hf`).
    - Reuse existing cached artifacts (from `~/.cache/huggingface`) by:
        - symlinking large Hub repos (models / large datasets) from the read-only cache
        - copying small datasets cache dirs (e.g. wikitext) into the writable cache
    """

    hub_cache = Path(os.environ["HF_HUB_CACHE"])
    datasets_cache = Path(os.environ["HF_DATASETS_CACHE"])
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    ro_root = Path.home() / ".cache" / "huggingface"
    ro_hub = ro_root / "hub"
    ro_datasets = ro_root / "datasets"

    def _mirror_hub_repo(repo_dirname: str) -> None:
        src = ro_hub / repo_dirname
        dst = hub_cache / repo_dirname
        if dst.exists():
            return
        if src.exists():
            dst.symlink_to(src, target_is_directory=True)

    def _copy_datasets_dir(dataset_dirname: str) -> None:
        src = ro_datasets / dataset_dirname
        dst = datasets_cache / dataset_dirname
        if dst.exists():
            return
        if src.exists():
            shutil.copytree(src, dst, symlinks=True)

    # Models: `org/name` -> `models--org--name` in Hub cache.
    mid = str(model_id)
    if "/" in mid and not Path(mid).exists():
        org, name = mid.split("/", 1)
        _mirror_hub_repo(f"models--{org}--{name}")

    # Eval always uses Wikitext2 in-process; ensure it is available offline.
    _mirror_hub_repo("datasets--wikitext")
    _copy_datasets_dir("wikitext")

    # Calibration datasets.
    ds = str(calibration_dataset)
    if ds == "c4":
        _mirror_hub_repo("datasets--allenai--c4")
    elif ds in {"wikitext2", "wikitext-2-raw-v1"}:
        pass
    else:
        # Other datasets may still work if already cached in the writable location.
        pass


def _resolve_local_hf_snapshot(model_id_or_path: str) -> Optional[str]:
    """Resolve an HF repo id to a local snapshot dir in the Hub cache.

    GPTQModel may call Hub APIs like `list_repo_files()` for remote ids, which
    fails in offline sandboxes. Passing a local snapshot directory avoids all
    Hub network calls.
    """

    mid = str(model_id_or_path).strip()
    if not mid:
        return None
    if Path(mid).exists():
        return mid
    if "/" not in mid:
        return None

    org, name = mid.split("/", 1)
    repo = Path(os.environ["HF_HUB_CACHE"]) / f"models--{org}--{name}"
    snapshots = repo / "snapshots"
    if not snapshots.exists():
        return None

    # Prefer pinned `main` ref if present.
    ref = repo / "refs" / "main"
    if ref.exists():
        snap_id = ref.read_text(encoding="utf-8", errors="replace").strip()
        cand = snapshots / snap_id
        if cand.exists():
            return str(cand)

    snap_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not snap_dirs:
        return None
    if len(snap_dirs) == 1:
        return str(snap_dirs[0])
    snap_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snap_dirs[0])


def _build_quantize_config(method: str, args) -> Any:
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization import FORMAT

    device = _resolve_device_str(getattr(args, "device", None))
    qcfg = QuantizeConfig(
        bits=int(args.bits),
        group_size=int(args.group_size),
        sym=bool(args.sym),
        # Default `FORMAT.GPTQ` (v1) is meant for checkpoint interchange; it forces a v2->v1 qzeros
        # conversion during packing which breaks in-process TorchQuantLinear inference. We always
        # evaluate in-process here, so prefer internal v2 format.
        format=FORMAT.GPTQ_V2,
        device=device,
        offload_to_disk=bool(args.offload_to_disk),
        rotation=str(args.rotation) if args.rotation else None,
        desc_act=bool(args.desc_act),
    )

    if method in {"mpgq", "rtn"}:
        qcfg.mpgq.enabled = True
        qcfg.mpgq.iters = 0 if method == "rtn" else int(args.mpgq_iters)
        qcfg.mpgq.damping = float(args.mpgq_damping)
        qcfg.mpgq.prior_tau = 0.0 if method == "rtn" else float(args.mpgq_prior_tau)
        qcfg.mpgq.polish_sweeps = 0 if method == "rtn" else int(args.mpgq_polish_sweeps)
        qcfg.mpgq.em_obs_precision = False if method == "rtn" else bool(args.mpgq_em_obs_precision)
        qcfg.mpgq.block_rows = int(args.mpgq_block_rows)
        qcfg.mpgq.rotation_seed = int(args.mpgq_rotation_seed)
    else:
        qcfg.mpgq.enabled = False

    if method == "ojbkq":
        from mpgq.baselines.ojbkq import OJBKQConfig

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


def run(args) -> Dict[str, Any]:
    resolved_device = _resolve_device_str(getattr(args, "device", None))
    if args.device and str(args.device).startswith("cuda") and not resolved_device.startswith("cuda"):
        raise RuntimeError(f"requested device={args.device!r} but CUDA failed to initialize")
    args.device = resolved_device
    _preinit_cuda(str(resolved_device))
    _setup_hf_offline_env()
    _force_system_compilers()

    _apply_patches()

    _ensure_gptqmodel_importable(extra_site_packages=getattr(args, "gptqmodel_extra_site_packages", None))
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    _setup_hf_cache(model_id=str(args.model_id), calibration_dataset=str(args.calibration_dataset))

    requested_method = str(args.method).strip().lower()
    supported = {"gptq", "gptq-strong", "mpgq", "rtn", "ojbkq", "awq+gptq", "awq+mpgq"}
    if requested_method not in supported:
        raise ValueError(f"unsupported method={requested_method!r} for backend_gptq")

    device = torch.device(str(args.device))

    awq_preprocess = False
    awq_apply_clip = False
    awq_time_s = 0.0

    # Resolve method -> base solver + canonical defaults.
    if requested_method in {"gptq", "gptq-strong"}:
        method = "gptq"
        args.desc_act = True
        args.rotation = "hadamard"
    elif requested_method == "mpgq":
        method = "mpgq"
        args.desc_act = False
        args.rotation = "hadamard"
    elif requested_method == "rtn":
        method = "rtn"
        args.desc_act = False
        args.rotation = "hadamard"
    elif requested_method == "ojbkq":
        method = "ojbkq"
        args.desc_act = False
        args.rotation = None
    elif requested_method == "awq+gptq":
        method = "gptq"
        awq_preprocess = True
        awq_apply_clip = True
        args.sym = False
        args.desc_act = True
        args.rotation = "hadamard"
    elif requested_method == "awq+mpgq":
        method = "mpgq"
        awq_preprocess = True
        awq_apply_clip = False
        args.sym = False
        args.desc_act = False
        args.rotation = "hadamard"
    else:
        raise RuntimeError("unreachable")

    model_id = str(args.model_id)
    model_path = _resolve_local_hf_snapshot(model_id) or model_id

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    from mpgq.data import load_or_build_calibration_blocks

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

    qcfg = _build_quantize_config(method, args)
    t_load = time.time()
    model = GPTQModel.load(model_path, qcfg)
    load_time_s = float(time.time() - t_load)
    if awq_preprocess:
        t_awq = time.time()
        _apply_awq_preprocess(
            qmodel=model,
            calibration_blocks=calib_blocks,
            bits=int(args.bits),
            group_size=int(args.group_size),
            sym=bool(args.sym),
            apply_clip=bool(awq_apply_clip),
            batch_size=int(args.awq_batch_size),
        )
        awq_time_s = float(time.time() - t_awq)
    t_quant = time.time()
    model.quantize(calib_blocks)
    quant_time = float(time.time() - t_quant)

    # Release large quantization intermediates before evaluation.
    del calib_blocks

    # Best-effort: clear optional workspace caches (may retain large buffers).
    if importlib.util.find_spec("gptqmodel.quantization.gptq") is not None:
        gptq_mod = importlib.import_module("gptqmodel.quantization.gptq")
        workspace_cache = getattr(gptq_mod, "_WORKSPACE_CACHE", None)
        if hasattr(workspace_cache, "clear"):
            workspace_cache.clear()

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
    ppl = eval_ppl_wikitext2(
        hf_model,
        tokenizer,
        seqlen=int(args.eval_seqlen),
        max_eval_tokens=int(args.eval_max_eval_tokens),
        output_dir=str(args.output_dir),
    )

    return {
        "backend": "gptqmodel",
        "requested_method": requested_method,
        "resolved_method": method,
        "baseline_family": "gptq" if requested_method in {"gptq", "gptq-strong"} else requested_method,
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "sym": bool(args.sym),
        "rotation": "none" if args.rotation in {None, "", "none"} else str(args.rotation),
        "desc_act": bool(args.desc_act),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "awq_preprocess": {
            "enabled": bool(awq_preprocess),
            "apply_clip": bool(awq_apply_clip),
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
            "dataset": "wikitext2",
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(args.eval_max_eval_tokens),
            "ppl": float(ppl),
        },
        "timing": {
            "awq_preprocess_s": float(awq_time_s),
            "load_s": float(load_time_s),
            "quantize_s": float(quant_time),
        },
        "quantization_path": {
            "dense": bool(getattr(qcfg, "mpgq", None) and bool(getattr(qcfg.mpgq, "enabled", False)) and str(getattr(qcfg, "rotation", None)) == "hadamard"),
            "packed": not bool(getattr(qcfg, "mpgq", None) and bool(getattr(qcfg.mpgq, "enabled", False)) and str(getattr(qcfg, "rotation", None)) == "hadamard"),
        },
        "meta": {
            "repo_sha": _git_head(REPO_ROOT),
            "submodules": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
        },
        "mpgq": {
            "enabled": bool(getattr(qcfg, "mpgq", None) and getattr(qcfg.mpgq, "enabled", False)),
            "iters": int(getattr(qcfg.mpgq, "iters", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "prior_tau": float(getattr(qcfg.mpgq, "prior_tau", 0.0)) if getattr(qcfg, "mpgq", None) else 0.0,
            "damping": float(getattr(qcfg.mpgq, "damping", 0.0)) if getattr(qcfg, "mpgq", None) else 0.0,
            "polish_sweeps": int(getattr(qcfg.mpgq, "polish_sweeps", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "em_obs_precision": bool(getattr(qcfg.mpgq, "em_obs_precision", False)) if getattr(qcfg, "mpgq", None) else False,
            "block_rows": int(getattr(qcfg.mpgq, "block_rows", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "rotation_seed": int(getattr(qcfg.mpgq, "rotation_seed", 0)) if getattr(qcfg, "mpgq", None) else 0,
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
    _maybe_reexec_via_runpy()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["gptq", "gptq-strong", "mpgq", "rtn", "ojbkq", "awq+gptq", "awq+mpgq"],
    )
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

    # AWQ preprocess
    parser.add_argument("--awq-batch-size", type=int, default=1)

    # MPGQ hyperparams (Gate A defaults)
    parser.add_argument("--mpgq-iters", type=int, default=10)
    parser.add_argument("--mpgq-damping", type=float, default=0.5)
    parser.add_argument("--mpgq-prior-tau", type=float, default=0.2)
    parser.add_argument("--mpgq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpgq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)

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
