from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.combo import QuantCombo, make_combo


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


def _collect_mpq_layer_stats(ckpt_dir: Path) -> Dict[str, Any]:
    """Collect lightweight surrogate-gate stats from per-layer checkpoints."""
    stats: Dict[str, Any] = {"modules": [], "violations": []}
    if not ckpt_dir.exists():
        return stats

    # QUIP# saves a small number of files per layer.
    ckpt_files = sorted(ckpt_dir.glob("*_qkv.pt")) + sorted(ckpt_dir.glob("*_o.pt")) + sorted(ckpt_dir.glob("*_up.pt")) + sorted(ckpt_dir.glob("*_down.pt"))
    for path in ckpt_files:
        try:
            payload = torch.load(path, map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            stats.setdefault("load_errors", []).append({"file": str(path), "error": str(exc)})
            continue

        mpq_stats = payload.get("mpq_stats")
        if not isinstance(mpq_stats, dict):
            continue

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

    stats["violation_modules"] = len(stats["violations"])
    stats["violation_chunks_total"] = int(sum(v.get("count", 0) for v in stats["violations"]))
    return stats


def _maybe_run_hessians(args, *, hessian_dir: Path, env: Dict[str, str], log_dir: Path) -> float:
    hessian_dir.mkdir(parents=True, exist_ok=True)
    meta_path = hessian_dir / "meta.json"
    have_any = any(hessian_dir.glob("*.pt"))
    expected_meta = {
        "base_model": str(args.base_model),
        "seed": int(args.seed),
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
        "-m",
        "quantize_llama.hessian_offline_llama",
        "--save_path",
        str(hessian_dir),
        "--base_model",
        str(args.base_model),
        "--seed",
        str(int(args.seed)),
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
    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True), encoding="utf-8")
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

    combo = make_combo(
        backend=str(getattr(args, "quant_backend", "quip")),
        codebook=str(getattr(args, "quant_codebook", "e8p12")),
        preprocess=str(getattr(args, "quant_preprocess", "none")),
        rotation=str(getattr(args, "quant_rotation", "had")),
        solver=str(getattr(args, "quant_solver", "ldlq_tune10")),
        polish_mode=str(getattr(args, "quant_polish_mode", "none")),
        topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
    )
    return combo, "default"


def run(args) -> Dict[str, Any]:
    _apply_patches()
    combo, request_source = _resolve_quip_combo(args)
    solver_mode, native_tune_iters = _route_native_quip(combo)

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

    env = _ensure_pythonpath(dict(os.environ))
    env = _ensure_ld_library_path(env)

    # Ensure Hessians exist once per (seed, base_model, devset).
    hessian_s = _maybe_run_hessians(args, hessian_dir=hessian_dir, env=env, log_dir=log_dir)

    quant_entry = [sys.executable, "-m", "quantize_llama.quantize_finetune_llama"]
    if solver_mode == "mpq":
        quant_entry = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "quantize_llama_mpq_wrapper.py"),
        ]

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
        str(int(args.seed)),
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
            str(int(getattr(args, "quant_topk_k", getattr(args, "mpq_topk_k", 8)))),
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
            str(int(getattr(args, "mpq_topk_k", 8))),
        ]
        quant_cmd.append("--mpq_em_obs_precision" if bool(args.mpq_em_obs_precision) else "--no-mpq_em_obs_precision")
        quant_cmd.append("--mpq_compile_prior" if bool(getattr(args, "mpq_compile_prior", True)) else "--no-mpq_compile_prior")

    t0 = time.time()
    rc = _run_logged(quant_cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / "quantize.log")
    if rc != 0:
        raise RuntimeError(f"quantize_finetune_llama failed with code {rc}")
    quantize_s = float(time.time() - t0)

    skip_eval = bool(getattr(args, "skip_eval", False))
    eval_datasets = list(getattr(args, "eval_datasets", ["wikitext2", "c4"]))
    hfize_s = 0.0
    if not skip_eval:
        t1 = time.time()
        rc = _run_logged(
            [sys.executable, "-m", "quantize_llama.hfize_llama", "--quantized_path", str(ckpt_dir), "--hf_output_path", str(hf_dir)],
            cwd=QUIP_ROOT,
            env=env,
            log_path=log_dir / "hfize.log",
        )
        if rc != 0:
            raise RuntimeError(f"hfize_llama failed with code {rc}")
        hfize_s = float(time.time() - t1)
    else:
        eval_datasets = []

    ppls: Dict[str, float] = {}
    eval_times: Dict[str, float] = {}
    for dataset in eval_datasets:
        dataset_norm = str(dataset).strip().lower()
        eval_cmd = [
            sys.executable,
            "-m",
            "eval.eval_ppl",
            "--hf_path",
            str(hf_dir),
            "--seed",
            str(int(args.seed)),
            "--seqlen",
            str(int(args.eval_seqlen)),
            "--datasets",
            dataset_norm,
            "--max_eval_tokens",
            str(int(getattr(args, "eval_max_eval_tokens", 0))),
            "--json_out",
            str(log_dir / f"eval_{dataset_norm}.json"),
            "--no_use_cuda_graph",
            "--no_use_flash_attn",
        ]
        t_eval = time.time()
        rc = _run_logged(eval_cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / f"eval_{dataset_norm}.log")
        if rc != 0:
            raise RuntimeError(f"eval_ppl failed with code {rc} on dataset={dataset_norm}")
        eval_times[f"eval_{dataset_norm}_s"] = float(time.time() - t_eval)
        dataset_ppl = _load_ppl_from_json(log_dir / f"eval_{dataset_norm}.json")
        if not dataset_ppl:
            dataset_ppl = _parse_ppl_from_log(log_dir / f"eval_{dataset_norm}.log")
        ppls.update(dataset_ppl)
    eval_s = float(sum(eval_times.values()))
    (log_dir / "eval_ppl.json").write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "seqlen": int(args.eval_seqlen),
                "datasets": eval_datasets,
                "max_eval_tokens": int(getattr(args, "eval_max_eval_tokens", 0)),
                "ppl": ppls,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    mpq_layer_stats = _collect_mpq_layer_stats(ckpt_dir) if solver_mode == "mpq" else {}

    return {
        "backend": combo.backend,
        "backend_impl": "quip-sharp",
        "codebook": combo.codebook,
        "preprocess": combo.preprocess,
        "rotation": combo.rotation,
        "solver": combo.solver,
        "polish_mode": combo.polish_mode,
        "topk_source": combo.topk_source,
        "requested_method": None if request_source in {"combo", "default"} else request_source,
        "resolved_method": solver_mode,
        "baseline_family": "quip" if combo.solver in {"ldlq", "ldlq_tune10"} else "mpq",
        "quantization_path": {"dense": False, "packed": True},
        "bits": 2,
        "seed": int(args.seed),
        "base_model": str(args.base_model),
        "codebook_impl": str(args.codebook),
        "calibration": {
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
            "datasets": eval_datasets,
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(getattr(args, "eval_max_eval_tokens", 0)),
            "ppl": ppls,
        },
        "timing": {
            "hessian_s": float(hessian_s),
            "quantize_s": float(quantize_s),
            "hfize_s": float(hfize_s),
            "eval_s": float(eval_s),
            **eval_times,
            "quantize_total_s": float(hessian_s + quantize_s + hfize_s),
            "eval_total_s": float(eval_s),
            "total_s": float(hessian_s + quantize_s + hfize_s + eval_s),
        },
        "paths": {
            "exp_dir": str(exp_dir),
            "hessian_dir": str(hessian_dir),
            "ckpt_dir": str(ckpt_dir),
            "hf_dir": str(hf_dir),
            "log_dir": str(log_dir),
        },
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
            "topk_k": int(getattr(args, "mpq_topk_k", 8)),
        }
        if solver_mode == "mpq"
        else {},
        "meta": {
            "repo_sha": _git_head(REPO_ROOT),
            "submodules": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, choices=["quip", "quip+mpq"])
    parser.add_argument("--quant-backend", dest="quant_backend", type=str, default="quip")
    parser.add_argument("--quant-codebook", dest="quant_codebook", type=str, default="e8p12")
    parser.add_argument("--quant-preprocess", dest="quant_preprocess", type=str, default="none")
    parser.add_argument("--quant-rotation", dest="quant_rotation", type=str, default="had")
    parser.add_argument("--quant-solver", dest="quant_solver", type=str, default=None, choices=["ldlq", "vamp", "ldlq_tune10"])
    parser.add_argument("--quant-polish-mode", dest="quant_polish_mode", type=str, default="none", choices=["none", "block_cd", "topk_block_cd"])
    parser.add_argument("--quant-topk-source", dest="quant_topk_source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", dest="quant_topk_k", type=int, default=8)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codebook", type=str, default="E8P12")

    # Calibration / Hessians
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--force-hessians", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--quip-tune-iters", dest="quip_tune_iters", type=int, default=10)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)

    # MPQ (QUIP side) defaults aligned with Gate B recipe.
    parser.add_argument("--mpq-init", type=str, default="random", choices=["random", "rtn", "ldlq"])
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
    parser.add_argument("--mpq-topk-k", type=int, default=8)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, required=True)

    args = parser.parse_args()
    result = run(args)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
