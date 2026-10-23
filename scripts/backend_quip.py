from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import snapshot_download
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.combo import QuantCombo, make_combo
from mpq.dense_decoder_adapter import resolve_dense_decoder_adapter_for_model_id
from mpq.eval_utils import eval_protocol_for_dataset


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _calibration_seed(args) -> int:
    value = getattr(args, "calibration_seed", None)
    return int(args.seed if value is None else value)


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


def _resolve_model_path(model_ref: str) -> str:
    """Resolve a HF model id to a local snapshot path with config.json.

    This avoids accidental shadowing when a repo-relative directory exists with
    the same name as the HF id (e.g. `Qwen/Qwen3-0.6B/` used for eval cache).
    """

    path = Path(model_ref)
    if path.exists() and (path / "config.json").exists():
        return str(path)
    return str(
        snapshot_download(
            model_ref,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json", "*.model"],
        )
    )


def _apply_patches() -> None:
    if os.environ.get("MPQ_SKIP_PATCHES", "").strip() in {"1", "true", "TRUE", "yes"}:
        print("[patches] MPQ_SKIP_PATCHES set; assuming patched worktree", flush=True)
        return
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "patches" / "apply_patches.py"),
            "--quip-only",
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


def _ensure_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    # Ensure patched quip-sharp can import `mpq.*` while running under QUIP_ROOT.
    existing = env.get("PYTHONPATH", "")
    prefix = str(REPO_ROOT)
    if existing:
        env["PYTHONPATH"] = f"{prefix}:{existing}"
    else:
        env["PYTHONPATH"] = prefix
    return env


def _ensure_ld_library_path(env: Dict[str, str]) -> Dict[str, str]:
    """Ensure CUDA extensions can load conda-provided runtime libraries.

    `conda run` does not reliably set `LD_LIBRARY_PATH`, and some editable CUDA
    extensions (e.g. quiptools_cuda) may link against newer `libstdc++` from
    the conda env.
    """

    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return env
    lib_dir = str(Path(conda_prefix) / "lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}"
    else:
        env["LD_LIBRARY_PATH"] = lib_dir
    return env


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Optional[Path],
) -> int:
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("a", encoding="utf-8")
        log_fh.write(f"\n# --- run {time.strftime('%Y-%m-%d %H:%M:%S')} cmd: {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if log_fh is not None:
                log_fh.write(line)
            # Keep stdout for interactive debugging.
            print(line, end="", flush=True)
    finally:
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()

    return int(proc.wait())


def _prepare_awq_base_model(
    *,
    args,
    combo: QuantCombo,
    exp_dir: Path,
    env: Dict[str, str],
    log_dir: Path,
) -> float:
    if combo.preprocess != "awq":
        return 0.0
    awq_cache_dir = getattr(args, "awq_cache_dir", None)
    awq_dir = Path(str(awq_cache_dir)).resolve() if awq_cache_dir else exp_dir / "awq_model"
    awq_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = awq_dir.parent / f".{awq_dir.name}.lock"
    cmd = [
        "conda",
        "run",
        "-n",
        str(getattr(args, "gptq_env", "mpq-gptq")),
        "python",
        str(REPO_ROOT / "scripts" / "prepare_awq_model.py"),
        "--model-id",
        str(args.base_model),
        "--output-dir",
        str(awq_dir),
        "--bits",
        "2",
        "--group-size",
        str(int(getattr(args, "awq_group_size", 128))),
        "--sym" if bool(getattr(args, "awq_sym", True)) else "--no-sym",
        "--calibration-dataset",
        str(args.calibration_dataset),
        "--calibration-split",
        str(args.calibration_split),
        "--calibration-limit",
        str(int(args.devset_size)),
        "--calibration-seqlen",
        str(int(args.ctx_size)),
        "--seed",
        str(_calibration_seed(args)),
        "--awq-batch-size",
        str(int(getattr(args, "awq_batch_size", 1))),
        "--no-apply-clip",
    ]
    extra_site = getattr(args, "gptqmodel_extra_site_packages", None)
    if extra_site:
        cmd += ["--gptqmodel-extra-site-packages", str(extra_site)]
    t0 = time.time()
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        rc = _run_logged(cmd, cwd=REPO_ROOT, env=env, log_path=log_dir / "awq_preprocess.log")
        if rc != 0:
            raise RuntimeError(f"prepare_awq_model.py failed with code {rc}")
    args.base_model = str(awq_dir)
    return float(time.time() - t0)


_PPL_RE = re.compile(r"(wikitext2|c4) perplexity:\\s*([0-9]+(?:\\.[0-9]+)?)", flags=re.IGNORECASE)


def _parse_ppl_from_log(log_path: Path) -> Dict[str, float]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}
    for match in _PPL_RE.finditer(text):
        out[match.group(1).lower()] = float(match.group(2))
    return out


def _load_ppl_from_json(json_path: Path) -> Dict[str, float]:
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    ppl = payload.get("ppl")
    if not isinstance(ppl, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in ppl.items():
        try:
            out[str(k).lower()] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _load_eval_fingerprints(json_path: Path) -> Dict[str, Dict[str, Any]]:
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    fingerprints = payload.get("token_fingerprints")
    if not isinstance(fingerprints, dict):
        return {}
    return {
        str(dataset).lower(): dict(fingerprint)
        for dataset, fingerprint in fingerprints.items()
        if isinstance(fingerprint, dict) and fingerprint.get("sha256")
    }


def _collect_mpq_layer_stats(ckpt_dir: Path) -> Dict[str, Any]:
    """Collect lightweight surrogate-gate stats from per-layer checkpoints."""
    stats: Dict[str, Any] = {
        "modules": [],
        "violations": [],
        "matrix_fingerprints": [],
        "snapshot_modules": [],
    }
    if not ckpt_dir.exists():
        return stats
    mechanism_rows: list[Dict[str, Any]] = []
    gamma_sum = 0.0
    gamma_count = 0
    changed_sum = 0
    changed_total = 0

    # QUIP# saves a small number of files per layer.
    ckpt_files = sorted(ckpt_dir.glob("*_qkv.pt")) + sorted(ckpt_dir.glob("*_o.pt")) + sorted(ckpt_dir.glob("*_up.pt")) + sorted(ckpt_dir.glob("*_down.pt"))
    for path in ckpt_files:
        try:
            payload = torch.load(path, map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            stats.setdefault("load_errors", []).append({"file": str(path), "error": str(exc)})
            continue

        fingerprint = payload.get("input_fingerprint")
        if isinstance(fingerprint, dict) and fingerprint.get("w_sha256") and fingerprint.get("h_sha256"):
            stats["matrix_fingerprints"].append(
                {
                    "module": path.stem,
                    "w_sha256": str(fingerprint["w_sha256"]),
                    "h_sha256": str(fingerprint["h_sha256"]),
                    "w_shape": list(fingerprint.get("w_shape", [])),
                    "h_shape": list(fingerprint.get("h_shape", [])),
                    "basis": str(fingerprint.get("basis", "")),
                }
            )
        mpq_stats = payload.get("mpq_stats")
        if not isinstance(mpq_stats, dict):
            continue
        snapshot_budgets = [
            int(budget)
            for budget in payload.get("cd_snapshot_budgets", [])
        ]
        if snapshot_budgets:
            stats["snapshot_modules"].append(
                {
                    "module": path.stem,
                    "budgets": snapshot_budgets,
                }
            )

        module_name = str(mpq_stats.get("module", path.name))
        violation_count = int(mpq_stats.get("surrogate_chunk_violation_count", 0) or 0)
        max_delta = float(mpq_stats.get("surrogate_chunk_violation_max_delta", 0.0) or 0.0)
        max_ratio = float(mpq_stats.get("surrogate_chunk_violation_max_ratio", 0.0) or 0.0)

        stats["modules"].append(
            {
                "module": module_name,
                "exact_surrogate_ldlq_mean": float(mpq_stats.get("exact_surrogate_ldlq_mean", float("nan"))),
                "exact_surrogate_after_mean": float(mpq_stats.get("exact_surrogate_after_mean", float("nan"))),
                "surrogate_chunk_violation_count": violation_count,
                "surrogate_chunk_violation_max_delta": max_delta,
                "surrogate_chunk_violation_max_ratio": max_ratio,
                "solver": str(mpq_stats.get("solver", "")),
                "solver_stats": mpq_stats.get("solver_stats", {}),
                "cd_trace": mpq_stats.get("cd_trace", []),
                "cd_snapshot_budgets": snapshot_budgets,
            }
        )
        if violation_count > 0:
            stats["violations"].append(
                {
                    "module": module_name,
                    "count": violation_count,
                    "max_delta": max_delta,
                    "max_ratio": max_ratio,
                }
            )

        mechanism = mpq_stats.get("mechanism")
        if isinstance(mechanism, dict):
            item = dict(mechanism)
            if item.get("layer_index") is None:
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(item.get("module", module_name)))
                if match is not None:
                    try:
                        item["layer_index"] = int(match.group(1))
                    except (TypeError, ValueError):
                        item["layer_index"] = None
            mechanism_rows.append(item)
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

    stats["violation_modules"] = len(stats["violations"])
    stats["violation_chunks_total"] = int(sum(v.get("count", 0) for v in stats["violations"]))
    stats["snapshot_module_count"] = len(stats["snapshot_modules"])
    if mechanism_rows:
        mechanism_rows.sort(key=lambda item: (item.get("layer_index") is None, item.get("layer_index"), str(item.get("module", ""))))
        stats["mechanism"] = {
            "regime": str(mechanism_rows[0].get("regime", "2bit_e8p12")),
            "cleanup_unit": str(mechanism_rows[0].get("cleanup_unit", "e8p12_block")),
            "modules": int(len(mechanism_rows)),
            "per_layer": mechanism_rows,
            "stage1_gamma1_mean_global": float(gamma_sum / max(gamma_count, 1)),
            "cleanup_changed_count_global": int(changed_sum),
            "cleanup_total_count_global": int(changed_total),
            "cleanup_changed_fraction_global": float(changed_sum / max(changed_total, 1)),
        }
    matrix_rows = stats["matrix_fingerprints"]
    matrix_rows.sort(key=lambda item: item["module"])
    encoded = json.dumps(matrix_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    stats["matrix_fingerprint_manifest"] = {
        "protocol": "paired_w_h_sha256_v1",
        "module_count": len(matrix_rows),
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "modules": matrix_rows,
    }
    return stats


def _has_partial_hessian_cache(hessian_dir: Path, stop_after_layer: Optional[int]) -> bool:
    if stop_after_layer is None:
        return False
    if not (hessian_dir / "calibration_manifest.json").exists():
        return False
    suffixes = ("qkv", "o", "up", "down")
    for idx in range(int(stop_after_layer) + 1):
        for suffix in suffixes:
            if not (hessian_dir / f"{idx}_{suffix}.pt").exists():
                return False
    return True


def _maybe_run_hessians(args, *, hessian_dir: Path, env: Dict[str, str], log_dir: Path) -> float:
    hessian_dir.mkdir(parents=True, exist_ok=True)
    lock_path = hessian_dir / ".producer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _maybe_run_hessians_locked(
            args,
            hessian_dir=hessian_dir,
            env=env,
            log_dir=log_dir,
        )


def _maybe_run_hessians_locked(
    args,
    *,
    hessian_dir: Path,
    env: Dict[str, str],
    log_dir: Path,
) -> float:
    hessian_dir.mkdir(parents=True, exist_ok=True)
    meta_path = hessian_dir / "meta.json"
    have_any = any(hessian_dir.glob("*.pt"))
    expected_meta = {
        "base_model": str(args.base_model),
        "seed": _calibration_seed(args),
        "devset_size": int(args.devset_size),
        "ctx_size": int(args.ctx_size),
        "sample_proc": int(args.sample_proc),
        "hessian_batch_size": int(args.hessian_batch_size),
        "calibration_dataset": str(args.calibration_dataset),
        "calibration_split": str(args.calibration_split),
    }
    if have_any and not bool(args.force_hessians) and meta_path.exists():
        try:
            on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            on_disk = None
        if isinstance(on_disk, dict) and on_disk == expected_meta:
            print(f"[hess] cache hit: {hessian_dir}", flush=True)
            return 0.0
        print(f"[hess] cache meta mismatch; recomputing: {hessian_dir}", flush=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "hessian_offline_llama_wrapper.py"),
        "--calibration-manifest-out",
        str(hessian_dir / "calibration_manifest.json"),
        "--save_path",
        str(hessian_dir),
        "--base_model",
        str(args.base_model),
        "--seed",
        str(_calibration_seed(args)),
        "--batch_size",
        str(int(args.hessian_batch_size)),
        "--devset_size",
        str(int(args.devset_size)),
        "--ctx_size",
        str(int(args.ctx_size)),
        "--sample_proc",
        str(int(args.sample_proc)),
        "--calibration_dataset",
        str(args.calibration_dataset),
        "--calibration_split",
        str(args.calibration_split),
    ]
    t0 = time.time()
    rc = _run_logged(cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / "hessian.log")
    if rc != 0:
        raise RuntimeError(f"hessian_offline_llama failed with code {rc}")
    _atomic_json(meta_path, expected_meta)
    return float(time.time() - t0)


def _resolve_quip_model_runtime(args) -> Dict[str, Any]:
    import json as _json
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(args.base_model))
    model_type = str(getattr(cfg, "model_type", "") or "").strip().lower()
    adapter = resolve_dense_decoder_adapter_for_model_id(
        str(args.base_model), override=getattr(args, "model_arch", None)
    )
    force_dense = bool(getattr(args, "force_dense_decoder", False))
    return {
        "model_type": model_type,
        "adapter_name": adapter.name,
        "use_legacy_llama_path": (model_type == "llama") and not force_dense,
    }


def _maybe_run_hessians_generic(args, *, hessian_dir: Path, env: Dict[str, str], log_dir: Path) -> float:
    hessian_dir.mkdir(parents=True, exist_ok=True)
    lock_path = hessian_dir / ".producer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _maybe_run_hessians_generic_locked(
            args,
            hessian_dir=hessian_dir,
            env=env,
            log_dir=log_dir,
        )


def _maybe_run_hessians_generic_locked(
    args,
    *,
    hessian_dir: Path,
    env: Dict[str, str],
    log_dir: Path,
) -> float:
    hessian_dir.mkdir(parents=True, exist_ok=True)
    meta_path = hessian_dir / "meta.json"
    have_any = any(hessian_dir.glob("*.pt"))
    expected_meta = {
        "base_model": str(args.base_model),
        "seed": _calibration_seed(args),
        "devset_size": int(args.devset_size),
        "ctx_size": int(args.ctx_size),
        "sample_proc": int(args.sample_proc),
        "hessian_batch_size": int(args.hessian_batch_size),
        "calibration_dataset": str(args.calibration_dataset),
        "calibration_split": str(args.calibration_split),
        "model_arch": str(getattr(args, "model_arch", "auto")),
        "pipeline": "dense_decoder_generic",
    }
    if have_any and not bool(args.force_hessians) and meta_path.exists():
        try:
            on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            on_disk = None
        if isinstance(on_disk, dict) and on_disk == expected_meta:
            print(f"[hess] cache hit: {hessian_dir}", flush=True)
            return 0.0
        print(f"[hess] cache meta mismatch; recomputing: {hessian_dir}", flush=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "hessian_offline_dense.py"),
        "--save_path",
        str(hessian_dir),
        "--base_model",
        str(args.base_model),
        "--seed",
        str(int(args.seed)),
        "--calibration_seed",
        str(_calibration_seed(args)),
        "--batch_size",
        str(int(args.hessian_batch_size)),
        "--devset_size",
        str(int(args.devset_size)),
        "--ctx_size",
        str(int(args.ctx_size)),
        "--sample_proc",
        str(int(args.sample_proc)),
        "--calibration_dataset",
        str(args.calibration_dataset),
        "--calibration_split",
        str(args.calibration_split),
        "--model-arch",
        str(getattr(args, "model_arch", "auto")),
    ]
    t0 = time.time()
    rc = _run_logged(cmd, cwd=REPO_ROOT, env=env, log_path=log_dir / "hessian.log")
    if rc != 0:
        raise RuntimeError(f"hessian_offline_dense failed with code {rc}")
    _atomic_json(meta_path, expected_meta)
    return float(time.time() - t0)


def _route_native_quip(combo: QuantCombo) -> tuple[str, int]:
    if combo.solver == "ldlq_tune10":
        return "quip", 10
    if combo.solver == "ldlq" and combo.polish_mode == "none":
        return "quip", 0
    return "mpq", 0


def _resolve_quip_combo(args) -> tuple[QuantCombo, str]:
    quant_solver = getattr(args, "quant_solver", None)
    if quant_solver is not None:
        combo = make_combo(
            backend=str(getattr(args, "quant_backend", "quip")),
            codebook=str(getattr(args, "quant_codebook", "e8p12")),
            preprocess=str(getattr(args, "quant_preprocess", "none")),
            rotation=str(getattr(args, "quant_rotation", "had")),
            solver=str(quant_solver),
            polish_mode=str(getattr(args, "quant_polish_mode", "none")),
            topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
        )
        return combo, "combo"

    method = str(getattr(args, "method", "") or "").strip().lower()
    if method == "quip":
        solver = "ldlq_tune10" if int(getattr(args, "quip_tune_iters", 10)) == 10 else "ldlq"
        combo = make_combo(
            backend="quip",
            codebook="e8p12",
            preprocess="none",
            rotation=str(getattr(args, "quant_rotation", "had")),
            solver=solver,
            polish_mode="none",
        )
        return combo, method
    if method == "quip+mpq":
        legacy_method = str(getattr(args, "mpq_method", "vamp") or "vamp").strip().lower()
        polish_mode = "topk_block_cd" if legacy_method == "vamp-topk-polish" else "block_cd"
        if int(getattr(args, "mpq_polish_sweeps", 3) or 0) <= 0:
            polish_mode = "none"
        combo = make_combo(
            backend="quip",
            codebook="e8p12",
            preprocess="none",
            rotation=str(getattr(args, "quant_rotation", "had")),
            solver="vamp",
            polish_mode=polish_mode,
        )
        return combo, method
    if method in {"cdquant", "quantease", "ojbkq"}:
        combo = make_combo(
            backend="quip",
            codebook="e8p12",
            preprocess="none",
            rotation=str(getattr(args, "quant_rotation", "had")),
            solver=method,
            polish_mode="none",
        )
        return combo, method

    combo = make_combo(
        backend=str(getattr(args, "quant_backend", "quip")),
        codebook=str(getattr(args, "quant_codebook", "e8p12")),
        preprocess=str(getattr(args, "quant_preprocess", "none")),
        rotation=str(getattr(args, "quant_rotation", "had")),
        solver=str(getattr(args, "quant_solver", None) or "ldlq_tune10"),
        polish_mode=str(getattr(args, "quant_polish_mode", "none")),
        topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
    )
    return combo, "default"


def run(args) -> Dict[str, Any]:
    _apply_patches()
    solver_seed = int(args.seed)
    calibration_seed = int(
        args.seed if getattr(args, "calibration_seed", None) is None else args.calibration_seed
    )
    calibration_draw = int(
        calibration_seed if getattr(args, "calibration_draw", None) is None else args.calibration_draw
    )
    rotation_seed = int(
        args.seed if getattr(args, "rotation_seed", None) is None else args.rotation_seed
    )
    eval_seed = int(args.seed if getattr(args, "eval_seed", None) is None else args.eval_seed)
    args.calibration_seed = calibration_seed
    combo, request_source = _resolve_quip_combo(args)
    solver_mode, native_tune_iters = _route_native_quip(combo)
    base_model_ref = str(args.base_model)
    args.base_model = _resolve_model_path(base_model_ref)
    runtime_arch = _resolve_quip_model_runtime(args)
    use_legacy_llama_path = bool(runtime_arch["use_legacy_llama_path"])

    exp_dir = Path(args.output_dir).resolve()
    log_dir = exp_dir / "log"
    ckpt_dir = exp_dir / "ckpt"
    hf_dir = exp_dir / "hf"
    hessian_dir = (Path(args.hessian_dir).resolve() if args.hessian_dir else (exp_dir / "hess"))

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    hessian_dir.mkdir(parents=True, exist_ok=True)
    if (
        getattr(args, "mpq_snapshot_budgets", None) is None
        and "mechanism__" in str(exp_dir)
        and solver_mode == "mpq"
        and combo.solver in {"ldlq", "vamp", "ojbkq"}
        and int(args.mpq_polish_sweeps) > 0
    ):
        args.mpq_snapshot_budgets = ",".join(
            str(budget) for budget in range(int(args.mpq_polish_sweeps) + 1)
        )
    start_layer = (
        None
        if getattr(args, "start_layer", None) is None
        else int(getattr(args, "start_layer"))
    )
    stop_after_layer = (
        None
        if getattr(args, "stop_after_layer", None) is None
        else int(getattr(args, "stop_after_layer"))
    )

    env = _ensure_pythonpath(dict(os.environ))
    env = _ensure_ld_library_path(env)

    awq_preprocess_s = _prepare_awq_base_model(
        args=args,
        combo=combo,
        exp_dir=exp_dir,
        env=env,
        log_dir=log_dir,
    )

    # Ensure Hessians exist once per (seed, base_model, devset).
    if _has_partial_hessian_cache(hessian_dir, stop_after_layer):
        print(f"[hess] partial cache hit: {hessian_dir} (layers <= {stop_after_layer})", flush=True)
        hessian_s = 0.0
    elif use_legacy_llama_path:
        hessian_s = _maybe_run_hessians(args, hessian_dir=hessian_dir, env=env, log_dir=log_dir)
    else:
        hessian_s = _maybe_run_hessians_generic(args, hessian_dir=hessian_dir, env=env, log_dir=log_dir)

    if use_legacy_llama_path:
        quant_entry = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "quantize_llama_mpq_wrapper.py"),
            "--calibration-seed",
            str(calibration_seed),
            "--expected-calibration-manifest",
            str(hessian_dir / "calibration_manifest.json"),
        ]
    else:
        quant_entry = [sys.executable, str(REPO_ROOT / "scripts" / "quantize_dense_decoder.py")]

    quant_cmd = [
        *quant_entry,
        "--save_path",
        str(ckpt_dir),
        "--hessian_path",
        str(hessian_dir),
        "--base_model",
        str(args.base_model),
        "--codebook",
        str(args.codebook),
        "--solver",
        solver_mode,
        "--quip_tune_iters",
        str(int(native_tune_iters)),
        "--ft_epochs",
        "0",
        "--seed",
        str(solver_seed),
        "--devset_size",
        str(int(args.devset_size)),
        "--ctx_size",
        str(int(args.ctx_size)),
        "--calibration_dataset",
        str(args.calibration_dataset),
        "--calibration_split",
        str(args.calibration_split),
        "--batch_size",
        str(int(args.batch_size)),
        "--scale_override",
        str(float(args.scale_override)),
        "--sample_proc",
        str(int(args.sample_proc)),
    ]
    if not use_legacy_llama_path:
        quant_cmd += [
            "--calibration_seed",
            str(calibration_seed),
            "--model-arch",
            str(getattr(args, "model_arch", "auto")),
        ]
    if solver_mode == "mpq":
        quant_cmd += [
            "--quant-backend",
            str(combo.backend),
            "--quant-codebook",
            str(combo.codebook),
            "--quant-preprocess",
            str(combo.preprocess),
            "--quant-rotation",
            str(combo.rotation),
            "--quant-solver",
            str(combo.solver),
            "--quant-polish-mode",
            str(combo.polish_mode),
            "--quant-topk-source",
            str(combo.topk_source),
            "--quant-topk-k",
            str(int(getattr(args, "quant_topk_k", getattr(args, "mpq_topk_k", 16)))),
            "--mpq-method",
            str(getattr(args, "mpq_method", "vamp")),
            "--mpq_init",
            str(args.mpq_init),
            "--mpq_iters",
            str(int(args.mpq_iters)),
            "--mpq_prior_tau",
            str(float(args.mpq_prior_tau)),
            "--mpq_prior_tau_scale_mode",
            str(args.mpq_prior_tau_scale_mode),
            "--mpq_damping",
            str(float(args.mpq_damping)),
            "--mpq_gamma_start",
            str(float(args.mpq_gamma_start)),
            "--mpq_gamma_retry_factor",
            str(float(args.mpq_gamma_retry_factor)),
            "--mpq_gamma_retry_steps",
            str(int(args.mpq_gamma_retry_steps)),
            "--mpq_obs_precision",
            str(float(args.mpq_obs_precision)),
            "--mpq_polish_sweeps",
            str(int(args.mpq_polish_sweeps)),
            "--mpq_polish_metric",
            str(args.mpq_polish_metric),
            "--mpq_polish_max_blocks",
            str(int(getattr(args, "mpq_polish_max_blocks", 0))),
            "--mpq_blocks_per_chunk",
            str(int(args.mpq_blocks_per_chunk)),
            "--mpq_block_rows",
            str(int(args.mpq_block_rows)),
            "--mpq_metric_rows_per_chunk",
            str(int(args.mpq_metric_rows_per_chunk)),
            "--mpq-topk-k",
            str(int(getattr(args, "mpq_topk_k", 16))),
        ]
        if getattr(args, "mpq_snapshot_budgets", None):
            quant_cmd += [
                "--mpq_snapshot_budgets",
                str(args.mpq_snapshot_budgets),
            ]
        if combo.solver == "ojbkq":
            quant_cmd += [
                "--ojbkq-random-trials",
                str(int(getattr(args, "ojbkq_random_trials", 5))),
                "--ojbkq-alpha-mode",
                str(getattr(args, "ojbkq_alpha_mode", "paper")),
                "--ojbkq-alpha-scale",
                str(float(getattr(args, "ojbkq_alpha_scale", 1.0))),
            ]
            if getattr(args, "ojbkq_alpha_override", None) is not None:
                quant_cmd += ["--ojbkq-alpha-override", str(float(args.ojbkq_alpha_override))]
            if getattr(args, "ojbkq_sample_topk", None) is not None:
                quant_cmd += ["--ojbkq-sample-topk", str(int(args.ojbkq_sample_topk))]
        quant_cmd.append("--mpq_em_obs_precision" if bool(args.mpq_em_obs_precision) else "--no-mpq_em_obs_precision")
        quant_cmd.append("--mpq_compile_prior" if bool(getattr(args, "mpq_compile_prior", True)) else "--no-mpq_compile_prior")
        quant_cmd.append("--mpq_best_iterate" if bool(getattr(args, "mpq_best_iterate", False)) else "--no-mpq_best_iterate")
    if bool(getattr(args, "summary_only", False)) and use_legacy_llama_path:
        quant_cmd.append("--summary-only")
    if stop_after_layer is not None:
        quant_cmd += ["--stop-after-layer", str(stop_after_layer)]
    if start_layer is not None:
        quant_cmd += ["--start-layer", str(start_layer)]

    t0 = time.time()
    rc = _run_logged(
        quant_cmd,
        cwd=QUIP_ROOT if use_legacy_llama_path else REPO_ROOT,
        env=env,
        log_path=log_dir / "quantize.log",
    )
    if rc != 0:
        raise RuntimeError(f"quantize failed with code {rc}")
    quantize_s = float(time.time() - t0)

    hfize_s = 0.0
    partial_run = bool(stop_after_layer is not None)
    skip_hfize = bool(getattr(args, "skip_hfize", False))
    if not partial_run and not skip_hfize:
        t1 = time.time()
        hfize_cmd = (
            [sys.executable, "-m", "quantize_llama.hfize_llama", "--quantized_path", str(ckpt_dir), "--hf_output_path", str(hf_dir)]
            if use_legacy_llama_path
            else [sys.executable, str(REPO_ROOT / "scripts" / "hfize_dense_decoder.py"), "--quantized_path", str(ckpt_dir), "--hf_output_path", str(hf_dir)]
        )
        rc = _run_logged(
            hfize_cmd,
            cwd=QUIP_ROOT if use_legacy_llama_path else REPO_ROOT,
            env=env,
            log_path=log_dir / "hfize.log",
        )
        if rc != 0:
            raise RuntimeError(f"hfize failed with code {rc}")
        hfize_s = float(time.time() - t1)

    skip_eval = bool(getattr(args, "skip_eval", False)) or partial_run
    eval_datasets = [] if skip_eval else list(getattr(args, "eval_datasets", ["wikitext2", "c4"]))

    ppls: Dict[str, float] = {}
    eval_token_fingerprints: Dict[str, Dict[str, Any]] = {}
    eval_times: Dict[str, float] = {}
    eval_protocol: Dict[str, Any] = {}
    for dataset in eval_datasets:
        dataset_norm = str(dataset).strip().lower()
        eval_protocol[dataset_norm] = eval_protocol_for_dataset(
            dataset_norm,
            seqlen=int(args.eval_seqlen),
            requested_max_eval_tokens=int(getattr(args, "eval_max_eval_tokens", 0)),
        )
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_hf_ppl.py"),
            "--hf_path",
            str(hf_dir),
            *(["--quip"] if use_legacy_llama_path else []),
            "--tokenizer-ref",
            str(base_model_ref),
            "--seed",
            str(eval_seed),
            "--seqlen",
            str(int(args.eval_seqlen)),
            "--datasets",
            dataset_norm,
            "--max_eval_tokens",
            str(int(getattr(args, "eval_max_eval_tokens", 0))),
            "--json_out",
            str(log_dir / f"eval_{dataset_norm}.json"),
        ]
        t_eval = time.time()
        rc = _run_logged(
            eval_cmd,
            cwd=REPO_ROOT,
            env=env,
            log_path=log_dir / f"eval_{dataset_norm}.log",
        )
        if rc != 0:
            raise RuntimeError(f"eval_ppl failed with code {rc} on dataset={dataset_norm}")
        eval_times[f"eval_{dataset_norm}_s"] = float(time.time() - t_eval)
        dataset_ppl = _load_ppl_from_json(log_dir / f"eval_{dataset_norm}.json")
        if not dataset_ppl:
            dataset_ppl = _parse_ppl_from_log(log_dir / f"eval_{dataset_norm}.log")
        ppls.update(dataset_ppl)
        eval_token_fingerprints.update(
            _load_eval_fingerprints(log_dir / f"eval_{dataset_norm}.json")
        )
    eval_s = float(sum(eval_times.values()))
    (log_dir / "eval_ppl.json").write_text(
        json.dumps(
            {
                "seed": eval_seed,
                "seqlen": int(args.eval_seqlen),
                "datasets": eval_datasets,
                "max_eval_tokens": int(getattr(args, "eval_max_eval_tokens", 0)),
                "eval_protocol": eval_protocol,
                "ppl": ppls,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lm_eval_tasks = list(getattr(args, "lm_eval_tasks", None) or [])
    lm_eval_result = {}
    lm_eval_time_s = 0.0
    if lm_eval_tasks and not skip_eval:
        t_lm = time.time()
        lm_eval_json = log_dir / "lm_eval_results.json"
        lm_eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_lm_eval.py"),
            "--model-path", str(hf_dir),
            *(["--quip"] if use_legacy_llama_path else []),
            "--tasks", ",".join(lm_eval_tasks),
            "--batch-size", str(getattr(args, "lm_eval_batch_size", "1")),
            "--num-fewshot", str(int(getattr(args, "lm_eval_num_fewshot", 0))),
            "--json-out", str(lm_eval_json),
        ]
        if getattr(args, "lm_eval_include_path", None):
            lm_eval_cmd += ["--include-path", str(args.lm_eval_include_path)]
        rc = _run_logged(
            lm_eval_cmd,
            cwd=QUIP_ROOT if use_legacy_llama_path else REPO_ROOT,
            env=env,
            log_path=log_dir / "lm_eval.log",
        )
        if rc != 0:
            raise RuntimeError(f"eval_lm_eval.py failed with code {rc}")
        lm_eval_time_s = float(time.time() - t_lm)
        try:
            lm_eval_result = json.loads(lm_eval_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lm_eval_result = {}

    mpq_layer_stats = _collect_mpq_layer_stats(ckpt_dir)
    calibration_manifest_path = hessian_dir / "calibration_manifest.json"
    if not calibration_manifest_path.exists():
        raise RuntimeError(f"missing calibration manifest after Hessian collection: {calibration_manifest_path}")
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    if int(calibration_manifest.get("seed", -1)) != calibration_seed:
        raise RuntimeError(
            "calibration manifest seed mismatch: "
            f"{calibration_manifest.get('seed')} != {calibration_seed}"
        )

    return {
        "backend": combo.backend,
        "backend_impl": "quip-sharp",
        "model_type": str(runtime_arch["model_type"]),
        "model_arch": str(runtime_arch["adapter_name"]),
        "codebook": combo.codebook,
        "preprocess": combo.preprocess,
        "rotation": combo.rotation,
        "solver": combo.solver,
        "polish_mode": combo.polish_mode,
        "topk_source": combo.topk_source,
        "requested_method": None if request_source in {"combo", "default"} else request_source,
        "resolved_method": solver_mode,
        "baseline_family": "quip" if combo.solver in {"ldlq", "ldlq_tune10"} else ("mpq" if combo.solver == "vamp" else combo.solver),
        "quantization_path": {
            "dense": not use_legacy_llama_path,
            "packed": True,
            "legacy_llama_runtime": bool(use_legacy_llama_path),
        },
        "awq_preprocess": {
            "enabled": bool(combo.preprocess == "awq"),
            "time_s": float(awq_preprocess_s),
            "model_dir": str(args.base_model) if combo.preprocess == "awq" else "",
            "cache_dir": str(getattr(args, "awq_cache_dir", "") or ""),
            "group_size": int(getattr(args, "awq_group_size", 128)),
            "sym": bool(getattr(args, "awq_sym", True)),
            "apply_clip": False,
        },
        "bits": 2,
        "seed": solver_seed,
        "seeds": {
            "solver": solver_seed,
            "calibration": calibration_seed,
            "rotation": rotation_seed,
            "eval": eval_seed,
        },
        "base_model": base_model_ref,
        "base_model_resolved": str(args.base_model),
        "run_scope": {
            "partial": partial_run,
            "stop_after_layer": stop_after_layer,
            "skip_hfize": skip_hfize,
        },
        "codebook_impl": str(args.codebook),
        "calibration": {
            "draw": calibration_draw,
            "seed": calibration_seed,
            "fingerprint": calibration_manifest,
            "manifest_path": str(calibration_manifest_path),
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "devset_size": int(args.devset_size),
            "ctx_size": int(args.ctx_size),
            "batch_size": int(args.batch_size),
            "sample_proc": int(args.sample_proc),
            "scale_override": float(args.scale_override),
            "quip_tune_iters": int(native_tune_iters),
        },
        "eval": {
            "seed": eval_seed,
            "datasets": eval_datasets,
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(getattr(args, "eval_max_eval_tokens", 0)),
            "eval_protocol": eval_protocol,
            "token_fingerprints": eval_token_fingerprints,
            "ppl": ppls,
        },
        "lm_eval": lm_eval_result,
        "timing": {
            "awq_preprocess_s": float(awq_preprocess_s),
            "hessian_s": float(hessian_s),
            "quantize_s": float(quantize_s),
            "hfize_s": float(hfize_s),
            "eval_s": float(eval_s),
            **eval_times,
            "lm_eval_s": float(lm_eval_time_s),
            "quantize_total_s": float(awq_preprocess_s + hessian_s + quantize_s + hfize_s),
            "eval_total_s": float(eval_s + lm_eval_time_s),
            "total_s": float(awq_preprocess_s + hessian_s + quantize_s + hfize_s + eval_s + lm_eval_time_s),
        },
        "paths": {
            "native_dir": str(ckpt_dir),
            "hf_dir": "" if skip_hfize else str(hf_dir),
            "exp_dir": str(exp_dir),
            "hessian_dir": str(hessian_dir),
            "ckpt_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
        },
        "artifact_format": {
            "native": "quip",
            "hf": "huggingface",
        },
        "mechanism": (mpq_layer_stats.get("mechanism", {}) if isinstance(mpq_layer_stats, dict) else {}),
        "matrix_fingerprints": (
            mpq_layer_stats.get("matrix_fingerprint_manifest", {})
            if isinstance(mpq_layer_stats, dict)
            else {}
        ),
        "mpq": mpq_layer_stats,
        "mpq_args": {
            "init": str(args.mpq_init),
            "method": str(getattr(args, "mpq_method", "vamp")),
            "iters": int(args.mpq_iters),
            "prior_tau": float(args.mpq_prior_tau),
            "prior_tau_scale_mode": str(args.mpq_prior_tau_scale_mode),
            "damping": float(args.mpq_damping),
            "em_obs_precision": bool(args.mpq_em_obs_precision),
            "compile_prior": bool(getattr(args, "mpq_compile_prior", True)),
            "gamma_start": float(args.mpq_gamma_start),
            "gamma_retry_factor": float(args.mpq_gamma_retry_factor),
            "gamma_retry_steps": int(args.mpq_gamma_retry_steps),
            "obs_precision": float(args.mpq_obs_precision),
            "polish_sweeps": int(args.mpq_polish_sweeps),
            "polish_metric": str(args.mpq_polish_metric),
            "polish_max_blocks": int(getattr(args, "mpq_polish_max_blocks", 0)),
            "blocks_per_chunk": int(args.mpq_blocks_per_chunk),
            "block_rows": int(args.mpq_block_rows),
            "metric_rows_per_chunk": int(args.mpq_metric_rows_per_chunk),
            "topk_k": int(getattr(args, "mpq_topk_k", 16)),
            "summary_only": bool(getattr(args, "summary_only", False)),
        }
        if solver_mode == "mpq"
        else {},
        "ojbkq": {
            "enabled": bool(combo.solver == "ojbkq"),
            "random_trials": int(getattr(args, "ojbkq_random_trials", 5))
            if combo.solver == "ojbkq"
            else 0,
            "total_candidates": 1 + int(getattr(args, "ojbkq_random_trials", 5))
            if combo.solver == "ojbkq"
            else 0,
            "paper_ojbkq": bool(
                combo.solver == "ojbkq"
                and str(getattr(args, "ojbkq_alpha_mode", "paper")) == "paper"
                and getattr(args, "ojbkq_alpha_override", None) is None
            ),
            "greedy_initialization": "ldlq",
            "alpha_mode": str(getattr(args, "ojbkq_alpha_mode", "paper"))
            if combo.solver == "ojbkq"
            else "",
            "alpha_override": getattr(args, "ojbkq_alpha_override", None)
            if combo.solver == "ojbkq"
            else None,
            "alpha_scale": float(getattr(args, "ojbkq_alpha_scale", 1.0))
            if combo.solver == "ojbkq"
            else 0.0,
            "sample_topk": getattr(args, "ojbkq_sample_topk", None)
            if combo.solver == "ojbkq"
            else None,
        },
        "meta": {
            "protocol_version": str(getattr(args, "protocol_version", "legacy")),
            "repo_sha": _git_head(REPO_ROOT),
            "third_party": {
                "fast-hadamard-transform": _git_head(
                    REPO_ROOT / "third_party" / "fast-hadamard-transform"
                ),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, choices=["quip", "quip+mpq", "cdquant", "quantease", "ojbkq"])
    parser.add_argument("--quant-backend", dest="quant_backend", type=str, default="quip")
    parser.add_argument("--quant-codebook", dest="quant_codebook", type=str, default="e8p12")
    parser.add_argument("--quant-preprocess", dest="quant_preprocess", type=str, default="none")
    parser.add_argument("--quant-rotation", dest="quant_rotation", type=str, default="had")
    parser.add_argument("--quant-solver", dest="quant_solver", type=str, default=None, choices=["ldlq", "vamp", "ldlq_tune10", "cdquant", "quantease", "ojbkq"])
    parser.add_argument("--quant-polish-mode", dest="quant_polish_mode", type=str, default="none", choices=["none", "block_cd", "topk_block_cd"])
    parser.add_argument("--quant-topk-source", dest="quant_topk_source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", dest="quant_topk_k", type=int, default=16)
    parser.add_argument("--model-arch", type=str, default="auto")
    parser.add_argument("--force-dense-decoder", dest="force_dense_decoder", action=argparse.BooleanOptionalAction, default=False, help="Force Llama models through dense_decoder path instead of legacy quip-sharp path.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-seed", type=int, default=None)
    parser.add_argument("--calibration-draw", type=int, default=None)
    parser.add_argument("--rotation-seed", type=int, default=None)
    parser.add_argument("--codebook", type=str, default="E8P12")

    # Calibration / Hessians
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--awq-batch-size", type=int, default=4)
    parser.add_argument("--awq-group-size", type=int, default=128)
    parser.add_argument("--awq-sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--awq-cache-dir", type=str, default=None)
    parser.add_argument("--gptq-env", type=str, default="mpq-gptq")
    parser.add_argument("--gptqmodel-extra-site-packages", type=str, default=None)
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--force-hessians", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--quip-tune-iters", dest="quip_tune_iters", type=int, default=10)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--eval-max-eval-tokens", type=int, default=0, help="0 => paper-style defaults: full Wikitext2, C4 = 256*seqlen")
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)

    # lm-evaluation-harness
    parser.add_argument("--lm-eval-tasks", nargs="*", default=None)
    parser.add_argument("--lm-eval-batch-size", default="1")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=0)
    parser.add_argument("--lm-eval-include-path", default=None,
                        help="extra task yaml dir for lm-eval TaskManager (e.g. custom gpqa_main_mc_local)")

    # MPQ (QUIP side) defaults aligned with Gate B recipe.
    parser.add_argument("--mpq-init", type=str, default="float", choices=["float", "random", "rtn", "ldlq"])
    parser.add_argument("--mpq-method", type=str, default="vamp", choices=["vamp", "vamp-topk-polish"])
    parser.add_argument("--mpq-iters", type=int, default=16)
    parser.add_argument("--mpq-damping", type=float, default=0.5)
    parser.add_argument("--mpq-prior-tau", type=float, default=0.25)
    parser.add_argument("--mpq-prior-tau-scale-mode", type=str, default="absmax", choices=["none", "absmax", "rms", "mean_abs"])
    parser.add_argument("--mpq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpq-obs-precision", type=float, default=0.0)
    parser.add_argument("--mpq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-compile-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--mpq-block-rows", type=int, default=128)
    parser.add_argument("--mpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpq-polish-metric", type=str, default="gram", choices=["gram", "diag", "isotropic"])
    parser.add_argument("--mpq-polish-max-blocks", type=int, default=0)
    parser.add_argument("--mpq-metric-rows-per-chunk", type=int, default=0)
    parser.add_argument("--mpq-snapshot-budgets", type=str, default=None)
    parser.add_argument("--mpq-topk-k", type=int, default=16)
    parser.add_argument("--mpq-best-iterate", dest="mpq_best_iterate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)
    parser.add_argument("--start-layer", type=int, default=None)
    parser.add_argument("--stop-after-layer", type=int, default=None)
    parser.add_argument("--skip-hfize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, required=True)
    parser.add_argument("--protocol-version", type=str, default="legacy")

    args = parser.parse_args()
    result = run(args)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
