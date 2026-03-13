from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"


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


def _ensure_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    # Ensure patched quip-sharp can import `mpgq.*` while running under QUIP_ROOT.
    existing = env.get("PYTHONPATH", "")
    prefix = str(REPO_ROOT)
    if existing:
        env["PYTHONPATH"] = f"{prefix}:{existing}"
    else:
        env["PYTHONPATH"] = prefix
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
        log_fh = log_path.open("w", encoding="utf-8")

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


def _collect_mpgq_layer_stats(ckpt_dir: Path) -> Dict[str, Any]:
    """Collect lightweight surrogate-gate stats from per-layer checkpoints."""
    stats: Dict[str, Any] = {"modules": [], "violations": []}
    if not ckpt_dir.exists():
        return stats

    # QUIP# saves a small number of files per layer.
    ckpt_files = sorted(ckpt_dir.glob("*_qkv.pt")) + sorted(ckpt_dir.glob("*_o.pt")) + sorted(ckpt_dir.glob("*_up.pt")) + sorted(ckpt_dir.glob("*_down.pt"))
    for path in ckpt_files:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            stats.setdefault("load_errors", []).append({"file": str(path), "error": str(exc)})
            continue

        mpgq_stats = payload.get("mpgq_stats")
        if not isinstance(mpgq_stats, dict):
            continue

        module_name = str(mpgq_stats.get("module", path.name))
        violation_count = int(mpgq_stats.get("surrogate_chunk_violation_count", 0) or 0)
        max_delta = float(mpgq_stats.get("surrogate_chunk_violation_max_delta", 0.0) or 0.0)
        max_ratio = float(mpgq_stats.get("surrogate_chunk_violation_max_ratio", 0.0) or 0.0)

        stats["modules"].append(
            {
                "module": module_name,
                "exact_surrogate_ldlq_mean": float(mpgq_stats.get("exact_surrogate_ldlq_mean", float("nan"))),
                "exact_surrogate_after_mean": float(mpgq_stats.get("exact_surrogate_after_mean", float("nan"))),
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
    have_any = any(hessian_dir.glob("*.pt"))
    if have_any and not bool(args.force_hessians):
        print(f"[hess] cache hit: {hessian_dir}", flush=True)
        return 0.0

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
    ]
    t0 = time.time()
    rc = _run_logged(cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / "hessian.log")
    if rc != 0:
        raise RuntimeError(f"hessian_offline_llama failed with code {rc}")
    return float(time.time() - t0)


def run(args) -> Dict[str, Any]:
    _apply_patches()

    method = str(args.method).strip().lower()
    if method not in {"quip", "quip+mpgq"}:
        raise ValueError(f"unsupported method={method!r} for backend_quip")

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

    # Ensure Hessians exist once per (seed, base_model, devset).
    hessian_s = _maybe_run_hessians(args, hessian_dir=hessian_dir, env=env, log_dir=log_dir)

    solver = "quip" if method == "quip" else "mpgq"
    quant_cmd = [
        sys.executable,
        "-m",
        "quantize_llama.quantize_finetune_llama",
        "--save_path",
        str(ckpt_dir),
        "--hessian_path",
        str(hessian_dir),
        "--base_model",
        str(args.base_model),
        "--codebook",
        str(args.codebook),
        "--solver",
        solver,
        "--ft_epochs",
        "0",
        "--seed",
        str(int(args.seed)),
        "--devset_size",
        str(int(args.devset_size)),
        "--ctx_size",
        str(int(args.ctx_size)),
        "--batch_size",
        str(int(args.batch_size)),
        "--scale_override",
        str(float(args.scale_override)),
        "--sample_proc",
        str(int(args.sample_proc)),
    ]
    if solver == "mpgq":
        quant_cmd += [
            "--wmpq_init",
            str(args.wmpq_init),
            "--wmpq_iters",
            str(int(args.wmpq_iters)),
            "--wmpq_prior_tau",
            str(float(args.wmpq_prior_tau)),
            "--wmpq_prior_tau_scale_mode",
            str(args.wmpq_prior_tau_scale_mode),
            "--wmpq_damping",
            str(float(args.wmpq_damping)),
            "--wmpq_em_obs_precision",
            "--wmpq_gamma_start",
            str(float(args.wmpq_gamma_start)),
            "--wmpq_gamma_retry_factor",
            str(float(args.wmpq_gamma_retry_factor)),
            "--wmpq_gamma_retry_steps",
            str(int(args.wmpq_gamma_retry_steps)),
            "--wmpq_obs_precision",
            str(float(args.wmpq_obs_precision)),
            "--wmpq_vamp_x1_mode",
            str(args.wmpq_vamp_x1_mode),
            "--wmpq_vamp_map_mix",
            str(float(args.wmpq_vamp_map_mix)),
            "--wmpq_polish_sweeps",
            str(int(args.wmpq_polish_sweeps)),
            "--wmpq_polish_metric",
            str(args.wmpq_polish_metric),
            "--wmpq_blocks_per_chunk",
            str(int(args.wmpq_blocks_per_chunk)),
            "--wmpq_block_rows",
            str(int(args.wmpq_block_rows)),
            "--wmpq_metric_rows_per_chunk",
            str(int(args.wmpq_metric_rows_per_chunk)),
        ]
        if not bool(args.wmpq_adaptive_accept_reject):
            quant_cmd.append("--no-wmpq_adaptive_accept_reject")
        if not bool(args.wmpq_best_iterate_export):
            quant_cmd.append("--no-wmpq_best_iterate_export")

    t0 = time.time()
    rc = _run_logged(quant_cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / "quantize.log")
    if rc != 0:
        raise RuntimeError(f"quantize_finetune_llama failed with code {rc}")
    quantize_s = float(time.time() - t0)

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
        "--no_use_cuda_graph",
        "--no_use_flash_attn",
    ]
    t2 = time.time()
    rc = _run_logged(eval_cmd, cwd=QUIP_ROOT, env=env, log_path=log_dir / "eval.log")
    if rc != 0:
        raise RuntimeError(f"eval_ppl failed with code {rc}")
    eval_s = float(time.time() - t2)

    ppls = _parse_ppl_from_log(log_dir / "eval.log")
    mpgq_layer_stats = _collect_mpgq_layer_stats(ckpt_dir) if solver == "mpgq" else {}

    return {
        "backend": "quip-sharp",
        "method": method,
        "resolved_method": solver,
        "baseline_family": method,
        "quantization_path": {"dense": False, "packed": True},
        "bits": 2,
        "seed": int(args.seed),
        "base_model": str(args.base_model),
        "codebook": str(args.codebook),
        "calibration": {
            "dataset": "rp1t",
            "devset_size": int(args.devset_size),
            "ctx_size": int(args.ctx_size),
            "batch_size": int(args.batch_size),
            "sample_proc": int(args.sample_proc),
            "scale_override": float(args.scale_override),
        },
        "eval": {
            "datasets": ["wikitext2", "c4"],
            "seqlen": int(args.eval_seqlen),
            "ppl": ppls,
        },
        "timing": {
            "hessian_s": float(hessian_s),
            "quantize_s": float(quantize_s),
            "hfize_s": float(hfize_s),
            "eval_s": float(eval_s),
        },
        "paths": {
            "exp_dir": str(exp_dir),
            "hessian_dir": str(hessian_dir),
            "ckpt_dir": str(ckpt_dir),
            "hf_dir": str(hf_dir),
            "log_dir": str(log_dir),
        },
        "mpgq": mpgq_layer_stats,
        "wmpq_args": {
            "init": str(args.wmpq_init),
            "iters": int(args.wmpq_iters),
            "prior_tau": float(args.wmpq_prior_tau),
            "prior_tau_scale_mode": str(args.wmpq_prior_tau_scale_mode),
            "damping": float(args.wmpq_damping),
            "em_obs_precision": True,
            "gamma_start": float(args.wmpq_gamma_start),
            "gamma_retry_factor": float(args.wmpq_gamma_retry_factor),
            "gamma_retry_steps": int(args.wmpq_gamma_retry_steps),
            "obs_precision": float(args.wmpq_obs_precision),
            "vamp_x1_mode": str(args.wmpq_vamp_x1_mode),
            "vamp_map_mix": float(args.wmpq_vamp_map_mix),
            "adaptive_accept_reject": bool(args.wmpq_adaptive_accept_reject),
            "best_iterate_export": bool(args.wmpq_best_iterate_export),
            "polish_sweeps": int(args.wmpq_polish_sweeps),
            "polish_metric": str(args.wmpq_polish_metric),
            "blocks_per_chunk": int(args.wmpq_blocks_per_chunk),
            "block_rows": int(args.wmpq_block_rows),
            "metric_rows_per_chunk": int(args.wmpq_metric_rows_per_chunk),
        }
        if solver == "mpgq"
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
    parser.add_argument("--method", required=True, choices=["quip", "quip+mpgq"])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codebook", type=str, default="E8P12")

    # Calibration / Hessians
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--force-hessians", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hessian-batch-size", type=int, default=2)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)

    # MPGQ (QUIP side) defaults aligned with Gate B recipe.
    parser.add_argument("--wmpq-init", type=str, default="random", choices=["random", "rtn", "ldlq"])
    parser.add_argument("--wmpq-iters", type=int, default=4)
    parser.add_argument("--wmpq-damping", type=float, default=0.5)
    parser.add_argument("--wmpq-prior-tau", type=float, default=0.0)
    parser.add_argument("--wmpq-prior-tau-scale-mode", type=str, default="absmax", choices=["none", "absmax", "rms", "mean_abs"])
    parser.add_argument("--wmpq-gamma-start", type=float, default=1.0)
    parser.add_argument("--wmpq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--wmpq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--wmpq-obs-precision", type=float, default=0.0)
    parser.add_argument("--wmpq-vamp-x1-mode", type=str, default="map", choices=["mean", "map", "mix"])
    parser.add_argument("--wmpq-vamp-map-mix", type=float, default=0.5)
    parser.add_argument("--wmpq-adaptive-accept-reject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wmpq-best-iterate-export", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wmpq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--wmpq-block-rows", type=int, default=32)
    parser.add_argument("--wmpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--wmpq-polish-metric", type=str, default="gram", choices=["gram", "diag", "isotropic"])
    parser.add_argument("--wmpq-metric-rows-per-chunk", type=int, default=0)

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
