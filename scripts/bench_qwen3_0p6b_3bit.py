from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    label: str
    method: str
    mpgq_iters: int = 10
    mpgq_polish_sweeps: int = 0


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run_logged(cmd: List[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)} (see {log_path})")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)

    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)

    parser.add_argument("--mpgq-damping", type=float, default=0.5)
    parser.add_argument("--mpgq-prior-tau", type=float, default=0.2)
    parser.add_argument("--mpgq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)

    parser.add_argument("--awq-batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-run even if backend_results.json exists.")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    run_id = str(args.run_id) if args.run_id else f"{_timestamp_id()}_qwen3_0p6b_3bit"
    run_dir = (REPO_ROOT / str(args.runs_dir) / run_id / f"seed{int(args.seed)}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        Job(label="mpgq_vamp10", method="mpgq", mpgq_iters=10, mpgq_polish_sweeps=0),
        Job(label="mpgq_vamp10_polish3", method="mpgq", mpgq_iters=10, mpgq_polish_sweeps=3),
        Job(label="awq_mpgq_vamp10", method="awq+mpgq", mpgq_iters=10, mpgq_polish_sweeps=0),
        Job(label="awq_mpgq_vamp10_polish3", method="awq+mpgq", mpgq_iters=10, mpgq_polish_sweeps=3),
        Job(label="gptq_strong_rht_descact", method="gptq-strong", mpgq_iters=0, mpgq_polish_sweeps=0),
    ]

    results: List[Dict[str, Any]] = []
    for job in jobs:
        method_dir = run_dir / job.label
        method_dir.mkdir(parents=True, exist_ok=True)
        json_out = method_dir / "backend_results.json"
        log_path = method_dir / "run.log"

        if json_out.exists() and not bool(args.force) and not bool(args.dry_run):
            try:
                payload = _json_load(json_out)
            except Exception:  # noqa: BLE001
                payload = {}
            if payload:
                payload["_job"] = asdict(job)
                payload["_reused"] = True
                results.append(payload)
                continue

        cmd: List[str] = [
            "python",
            str(REPO_ROOT / "scripts" / "backend_gptq.py"),
            "--method",
            str(job.method),
            "--model-id",
            str(args.model_id),
            "--bits",
            str(int(args.bits)),
            "--group-size",
            str(int(args.group_size)),
            "--sym" if bool(args.sym) else "--no-sym",
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
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
            str(int(args.awq_batch_size)),
            "--mpgq-iters",
            str(int(job.mpgq_iters)),
            "--mpgq-damping",
            str(float(args.mpgq_damping)),
            "--mpgq-prior-tau",
            str(float(args.mpgq_prior_tau)),
            "--mpgq-polish-sweeps",
            str(int(job.mpgq_polish_sweeps)),
            "--mpgq-em-obs-precision" if bool(args.mpgq_em_obs_precision) else "--no-mpgq-em-obs-precision",
            "--mpgq-block-rows",
            str(int(args.mpgq_block_rows)),
            "--mpgq-rotation-seed",
            str(int(args.mpgq_rotation_seed)),
            "--output-dir",
            str(run_dir),
            "--json-out",
            str(json_out),
        ]

        if args.dry_run:
            print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
            continue

        _run_logged(cmd, cwd=REPO_ROOT, log_path=log_path)
        payload = _json_load(json_out)
        payload["_job"] = asdict(job)
        results.append(payload)

    merged = {
        "meta": {
            "run_id": run_id,
            "seed": int(args.seed),
        },
        "args": vars(args),
        "results": results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {out_path}", flush=True)

    if results:
        def _ppl(rec: Dict[str, Any]) -> float:
            return float(rec.get("eval", {}).get("ppl", float("nan")))

        ordered = sorted(results, key=lambda r: str(r.get("_job", {}).get("label", "")))
        print("\nPPL summary (wikitext2):", flush=True)
        for rec in ordered:
            label = str(rec.get("_job", {}).get("label", rec.get("requested_method", "")))
            print(f"- {label}: {_ppl(rec):.4f}", flush=True)


if __name__ == "__main__":
    main()
