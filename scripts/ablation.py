from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple


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
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)} (see {log_path})")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class AblationSpec:
    polish_sweeps: int
    em_obs_precision: bool
    damping: float
    prior_tau: float

    def label(self) -> str:
        return (
            f"polish{int(self.polish_sweeps)}"
            f"_em{int(self.em_obs_precision)}"
            f"_damp{self.damping:g}"
            f"_tau{self.prior_tau:g}"
        )


def _grid() -> List[AblationSpec]:
    out: List[AblationSpec] = []
    for polish in (0, 3):
        for em in (False, True):
            for damping in (1.0, 0.5):
                for tau in (0.0, 0.2):
                    out.append(
                        AblationSpec(
                            polish_sweeps=int(polish),
                            em_obs_precision=bool(em),
                            damping=float(damping),
                            prior_tau=float(tau),
                        )
                    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])

    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)

    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)

    parser.add_argument("--mpgq-iters", type=int, default=10)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)

    parser.add_argument("--env", type=str, default="mpgq-gptq")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    run_id = str(args.run_id) if args.run_id else f"{_timestamp_id()}_ablation_b{int(args.bits)}"
    runs_dir = (REPO_ROOT / str(args.runs_dir)).resolve()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    grid = _grid()
    seeds = [int(s) for s in list(args.seeds)]

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for spec in grid:
            out_dir = seed_dir / spec.label()
            out_dir.mkdir(parents=True, exist_ok=True)
            json_out = out_dir / "backend_results.json"
            log_path = out_dir / "run.log"

            cmd = [
                "conda",
                "run",
                "-n",
                str(args.env),
                "python",
                str(REPO_ROOT / "scripts" / "backend_gptq.py"),
                "--method",
                "mpgq",
                "--model-id",
                str(args.model_id),
                "--bits",
                str(int(args.bits)),
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
                "--mpgq-iters",
                str(int(args.mpgq_iters)),
                "--mpgq-damping",
                str(float(spec.damping)),
                "--mpgq-prior-tau",
                str(float(spec.prior_tau)),
                "--mpgq-polish-sweeps",
                str(int(spec.polish_sweeps)),
                "--mpgq-em-obs-precision" if bool(spec.em_obs_precision) else "--no-mpgq-em-obs-precision",
                "--mpgq-block-rows",
                str(int(args.mpgq_block_rows)),
                "--mpgq-rotation-seed",
                str(int(args.mpgq_rotation_seed)),
                "--output-dir",
                str(seed_dir),
                "--json-out",
                str(json_out),
            ]

            if args.dry_run:
                print("$ " + " ".join(shlex.quote(x) for x in cmd))
                continue

            _run_logged(cmd, log_path=log_path, cwd=REPO_ROOT)
            payload = _json_load(json_out)
            payload["ablation"] = {
                "polish_sweeps": int(spec.polish_sweeps),
                "em_obs_precision": bool(spec.em_obs_precision),
                "damping": float(spec.damping),
                "prior_tau": float(spec.prior_tau),
            }
            results.append(payload)

    # Aggregate per spec across seeds.
    grouped: Dict[str, List[float]] = {}
    for row in results:
        label = AblationSpec(
            polish_sweeps=int(row["ablation"]["polish_sweeps"]),
            em_obs_precision=bool(row["ablation"]["em_obs_precision"]),
            damping=float(row["ablation"]["damping"]),
            prior_tau=float(row["ablation"]["prior_tau"]),
        ).label()
        ppl = float(row["eval"]["ppl"])
        if math.isfinite(ppl):
            grouped.setdefault(label, []).append(ppl)

    summary_rows: List[Tuple[str, float, float, int]] = []
    for label, ppls in grouped.items():
        summary_rows.append((label, float(mean(ppls)), float(pstdev(ppls)) if len(ppls) > 1 else 0.0, len(ppls)))
    summary_rows.sort(key=lambda x: x[1])

    out_json = {
        "meta": {
            "repo_sha": _git_head(REPO_ROOT),
            "submodules": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
        "args": vars(args),
        "results": results,
        "summary": [
            {"label": label, "mean_ppl": mu, "std_ppl": sd, "n": n} for (label, mu, sd, n) in summary_rows
        ],
    }
    (run_dir / "ablation.json").write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown table.
    lines = []
    lines.append(f"# Ablation: mpgq (bits={int(args.bits)})")
    lines.append("")
    lines.append("| Config | mean PPL | std | n |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label, mu, sd, n in summary_rows:
        lines.append(f"| `{label}` | {mu:.4f} | {sd:.4f} | {n} |")
    (run_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {run_dir / 'ablation.json'}")
    print(f"WROTE {run_dir / 'ablation.md'}")


if __name__ == "__main__":
    main()
