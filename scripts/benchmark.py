from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run_logged(cmd: List[str], *, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)} (see {log_path})")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    repo_sha: str
    submodules: Dict[str, str]
    started_at: str


def _default_methods(bits: int) -> List[str]:
    if int(bits) == 2:
        return ["quip", "quip+mpgq"]
    if int(bits) in {3, 4}:
        return ["rtn", "mpgq", "gptq", "ojbkq", "awq+mpgq", "awq+gptq"]
    raise ValueError(f"unsupported bits={bits}; expected 2/3/4")


def _parse_seeds(seeds: List[int]) -> List[int]:
    if not seeds:
        return [0]
    return [int(s) for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])

    # GPTQModel side
    parser.add_argument("--model-id", type=str, default=None, help="HF model id for GPTQModel backend (bits=3/4)")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)

    # MPGQ (GPTQ side) defaults (Gate A recipe)
    parser.add_argument("--mpgq-iters", type=int, default=10)
    parser.add_argument("--mpgq-damping", type=float, default=0.5)
    parser.add_argument("--mpgq-prior-tau", type=float, default=0.2)
    parser.add_argument("--mpgq-x1-mode", type=str, default="mmse", choices=["mmse", "map"])
    parser.add_argument("--mpgq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpgq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)

    # OJBKQ defaults
    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)

    # QUIP# side
    parser.add_argument("--base-model", type=str, default=None, help="HF model id for QUIP# backend (bits=2)")
    parser.add_argument("--codebook", type=str, default="E8P12")
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--force-hessians", action=argparse.BooleanOptionalAction, default=False)
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

    # Orchestration
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--gptq-env", type=str, default="mpgq-gptq")
    parser.add_argument("--quip-env", type=str, default="mpgq-quip")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    bits = int(args.bits)
    methods = args.methods if args.methods is not None and len(args.methods) else _default_methods(bits)
    seeds = _parse_seeds(list(args.seeds))

    if bits in {3, 4} and not args.model_id:
        raise ValueError("--model-id is required for bits=3/4")
    if bits == 2 and not args.base_model:
        raise ValueError("--base-model is required for bits=2")

    run_id = str(args.run_id) if args.run_id else f"{_timestamp_id()}_b{bits}"
    runs_dir = (REPO_ROOT / str(args.runs_dir)).resolve()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = RunMeta(
        run_id=run_id,
        repo_sha=_git_head(REPO_ROOT),
        submodules={
            "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
            "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
        },
        started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )

    all_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            method = str(method).strip()
            method_dir = seed_dir / method.replace("/", "_")
            method_dir.mkdir(parents=True, exist_ok=True)
            json_out = method_dir / "backend_results.json"
            log_path = method_dir / "run.log"

            if bits == 2:
                hess_dir = seed_dir / "hess"
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    str(args.quip_env),
                    "python",
                    str(REPO_ROOT / "scripts" / "backend_quip.py"),
                    "--method",
                    method,
                    "--base-model",
                    str(args.base_model),
                    "--seed",
                    str(int(seed)),
                    "--codebook",
                    str(args.codebook),
                    "--devset-size",
                    str(int(args.devset_size)),
                    "--ctx-size",
                    str(int(args.ctx_size)),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--sample-proc",
                    str(int(args.sample_proc)),
                    "--scale-override",
                    str(float(args.scale_override)),
                    "--hessian-dir",
                    str(hess_dir),
                    "--force-hessians" if bool(args.force_hessians) else "--no-force-hessians",
                    "--hessian-batch-size",
                    str(int(args.hessian_batch_size)),
                    "--eval-seqlen",
                    str(int(args.eval_seqlen)),
                    "--wmpq-init",
                    str(args.wmpq_init),
                    "--wmpq-iters",
                    str(int(args.wmpq_iters)),
                    "--wmpq-damping",
                    str(float(args.wmpq_damping)),
                    "--wmpq-prior-tau",
                    str(float(args.wmpq_prior_tau)),
                    "--wmpq-prior-tau-scale-mode",
                    str(args.wmpq_prior_tau_scale_mode),
                    "--wmpq-gamma-start",
                    str(float(args.wmpq_gamma_start)),
                    "--wmpq-gamma-retry-factor",
                    str(float(args.wmpq_gamma_retry_factor)),
                    "--wmpq-gamma-retry-steps",
                    str(int(args.wmpq_gamma_retry_steps)),
                    "--wmpq-obs-precision",
                    str(float(args.wmpq_obs_precision)),
                    "--wmpq-vamp-x1-mode",
                    str(args.wmpq_vamp_x1_mode),
                    "--wmpq-vamp-map-mix",
                    str(float(args.wmpq_vamp_map_mix)),
                    "--wmpq-adaptive-accept-reject" if bool(args.wmpq_adaptive_accept_reject) else "--no-wmpq-adaptive-accept-reject",
                    "--wmpq-best-iterate-export" if bool(args.wmpq_best_iterate_export) else "--no-wmpq-best-iterate-export",
                    "--wmpq-blocks-per-chunk",
                    str(int(args.wmpq_blocks_per_chunk)),
                    "--wmpq-block-rows",
                    str(int(args.wmpq_block_rows)),
                    "--wmpq-polish-sweeps",
                    str(int(args.wmpq_polish_sweeps)),
                    "--wmpq-polish-metric",
                    str(args.wmpq_polish_metric),
                    "--wmpq-metric-rows-per-chunk",
                    str(int(args.wmpq_metric_rows_per_chunk)),
                    "--output-dir",
                    str(method_dir),
                    "--json-out",
                    str(json_out),
                ]
            else:
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    str(args.gptq_env),
                    "python",
                    str(REPO_ROOT / "scripts" / "backend_gptq.py"),
                    "--method",
                    method,
                    "--model-id",
                    str(args.model_id),
                    "--bits",
                    str(int(bits)),
                    "--group-size",
                    str(int(args.group_size)),
                    "--sym" if bool(args.sym) else "--no-sym",
                    "--seed",
                    str(int(seed)),
                    "--calibration-dataset",
                    str(args.calibration_dataset),
                    "--calibration-split",
                    str(args.calibration_split),
                    "--calibration-limit",
                    str(int(args.calibration_limit)),
                    "--calibration-seqlen",
                    str(int(args.calibration_seqlen)),
                    "--eval-seqlen",
                    str(int(args.eval_seqlen)),
                    "--eval-max-eval-tokens",
                    str(int(args.eval_max_eval_tokens)),
                    "--awq-batch-size",
                    str(int(getattr(args, "awq_batch_size", 1))),
                    "--mpgq-iters",
                    str(int(args.mpgq_iters)),
                    "--mpgq-damping",
                    str(float(args.mpgq_damping)),
                    "--mpgq-prior-tau",
                    str(float(args.mpgq_prior_tau)),
                    "--mpgq-x1-mode",
                    str(args.mpgq_x1_mode),
                    "--mpgq-polish-sweeps",
                    str(int(args.mpgq_polish_sweeps)),
                    "--mpgq-em-obs-precision" if bool(args.mpgq_em_obs_precision) else "--no-mpgq-em-obs-precision",
                    "--mpgq-block-rows",
                    str(int(args.mpgq_block_rows)),
                    "--mpgq-rotation-seed",
                    str(int(args.mpgq_rotation_seed)),
                    "--ojbkq-random-trials",
                    str(int(args.ojbkq_random_trials)),
                    "--ojbkq-alpha-mode",
                    str(args.ojbkq_alpha_mode),
                    *(["--ojbkq-alpha-override", str(float(args.ojbkq_alpha_override))] if args.ojbkq_alpha_override is not None else []),
                    "--ojbkq-alpha-scale",
                    str(float(args.ojbkq_alpha_scale)),
                    *(["--ojbkq-sample-topk", str(int(args.ojbkq_sample_topk))] if args.ojbkq_sample_topk is not None else []),
                    "--output-dir",
                    str(seed_dir),
                    "--json-out",
                    str(json_out),
                ]

            if args.dry_run:
                print("$ " + " ".join(shlex.quote(x) for x in cmd))
                continue

            _run_logged(cmd, log_path=log_path, cwd=REPO_ROOT)
            all_results.append(_json_load(json_out))

    payload: Dict[str, Any] = {
        "meta": asdict(meta),
        "args": vars(args),
        "results": all_results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()

