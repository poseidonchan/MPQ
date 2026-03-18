from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpgq.combo import QuantCombo, make_combo


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


def _combo_ppl(row: Dict[str, Any]) -> float:
    eval_payload = row.get("eval", {})
    if not isinstance(eval_payload, dict):
        return float("nan")
    ppl = eval_payload.get("ppl")
    if isinstance(ppl, dict):
        try:
            return float(ppl.get("wikitext2", next(iter(ppl.values()))))
        except (StopIteration, TypeError, ValueError):
            return float("nan")
    try:
        return float(ppl)
    except (TypeError, ValueError):
        return float("nan")


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


def _resolve_combo(bits: int, args) -> QuantCombo:
    if int(bits) == 2:
        return make_combo(
            backend="quip",
            codebook="e8p12",
            preprocess=str(args.quant_preprocess),
            rotation=str(args.quant_rotation),
            solver=str(args.quant_solver),
            polish_mode=str(args.quant_polish_mode),
            topk_source=str(args.quant_topk_source),
        )
    return make_combo(
        backend="gptqmodel",
        codebook="grid_g128",
        preprocess=str(args.quant_preprocess),
        rotation=str(args.quant_rotation),
        solver=str(args.quant_solver),
        polish_mode=str(args.quant_polish_mode),
        topk_source=str(args.quant_topk_source),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])

    parser.add_argument("--quant-preprocess", type=str, default="none")
    parser.add_argument("--quant-rotation", type=str, default=None)
    parser.add_argument("--quant-solver", type=str, default=None)
    parser.add_argument("--quant-polish-mode", type=str, default=None)
    parser.add_argument("--quant-topk-source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", type=int, default=8)

    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--codebook", type=str, default="E8P12")

    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)

    parser.add_argument("--mpgq-iters", type=int, default=10)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)
    parser.add_argument("--mpgq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpgq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpgq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpgq-obs-precision", type=float, default=0.0)
    parser.add_argument("--mpgq-init", type=str, default="random", choices=["random", "rtn", "ldlq"])
    parser.add_argument("--mpgq-method", type=str, default="vamp")
    parser.add_argument("--mpgq-prior-tau-scale-mode", type=str, default="absmax")
    parser.add_argument("--mpgq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--mpgq-polish-metric", type=str, default="gram")
    parser.add_argument("--mpgq-polish-max-blocks", type=int, default=0)
    parser.add_argument("--mpgq-metric-rows-per-chunk", type=int, default=0)

    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--gptq-env", type=str, default="mpgq-gptq")
    parser.add_argument("--quip-env", type=str, default="mpgq-quip")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    bits = int(args.bits)
    seeds = [int(s) for s in list(args.seeds)]

    if bits in {3, 4} and not args.model_id:
        raise ValueError("--model-id is required for bits=3/4")
    if bits == 2 and not args.base_model:
        raise ValueError("--base-model is required for bits=2")

    if args.quant_rotation is None:
        args.quant_rotation = "had" if bits == 2 else "hadamard"
    if args.quant_solver is None:
        args.quant_solver = "vamp"
    if args.quant_polish_mode is None:
        args.quant_polish_mode = "block_cd"

    combo = _resolve_combo(bits, args)

    run_id = str(args.run_id) if args.run_id else f"{_timestamp_id()}_ablation_b{bits}"
    runs_dir = (REPO_ROOT / str(args.runs_dir)).resolve()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_name = str(args.env) if args.env else (str(args.quip_env) if bits == 2 else str(args.gptq_env))
    grid = _grid()

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for spec in grid:
            out_dir = seed_dir / spec.label()
            out_dir.mkdir(parents=True, exist_ok=True)
            json_out = out_dir / "backend_results.json"
            log_path = out_dir / "run.log"

            if bits == 2:
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    str(REPO_ROOT / "scripts" / "backend_quip.py"),
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
                    str(int(args.quant_topk_k)),
                    "--base-model",
                    str(args.base_model),
                    "--seed",
                    str(int(seed)),
                    "--codebook",
                    str(args.codebook),
                    "--calibration-dataset",
                    str(args.calibration_dataset),
                    "--calibration-split",
                    str(args.calibration_split),
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
                    "--hessian-batch-size",
                    str(int(args.hessian_batch_size)),
                    "--eval-seqlen",
                    str(int(args.eval_seqlen)),
                    "--eval-max-eval-tokens",
                    str(int(args.eval_max_eval_tokens)),
                    "--mpgq-init",
                    str(args.mpgq_init),
                    "--mpgq-method",
                    str(args.mpgq_method),
                    "--mpgq-iters",
                    str(int(args.mpgq_iters)),
                    "--mpgq-damping",
                    str(float(spec.damping)),
                    "--mpgq-prior-tau",
                    str(float(spec.prior_tau)),
                    "--mpgq-prior-tau-scale-mode",
                    str(args.mpgq_prior_tau_scale_mode),
                    "--mpgq-gamma-start",
                    str(float(args.mpgq_gamma_start)),
                    "--mpgq-gamma-retry-factor",
                    str(float(args.mpgq_gamma_retry_factor)),
                    "--mpgq-gamma-retry-steps",
                    str(int(args.mpgq_gamma_retry_steps)),
                    "--mpgq-obs-precision",
                    str(float(args.mpgq_obs_precision)),
                    "--mpgq-em-obs-precision" if bool(spec.em_obs_precision) else "--no-mpgq-em-obs-precision",
                    "--mpgq-blocks-per-chunk",
                    str(int(args.mpgq_blocks_per_chunk)),
                    "--mpgq-block-rows",
                    str(int(args.mpgq_block_rows)),
                    "--mpgq-polish-sweeps",
                    str(int(spec.polish_sweeps)),
                    "--mpgq-polish-metric",
                    str(args.mpgq_polish_metric),
                    "--mpgq-polish-max-blocks",
                    str(int(args.mpgq_polish_max_blocks)),
                    "--mpgq-metric-rows-per-chunk",
                    str(int(args.mpgq_metric_rows_per_chunk)),
                    "--mpgq-topk-k",
                    str(int(args.quant_topk_k)),
                    "--output-dir",
                    str(out_dir),
                    "--json-out",
                    str(json_out),
                ]
            else:
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    str(REPO_ROOT / "scripts" / "backend_gptq.py"),
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
                    str(int(args.quant_topk_k)),
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
                    "--mpgq-gamma-start",
                    str(float(args.mpgq_gamma_start)),
                    "--mpgq-gamma-retry-factor",
                    str(float(args.mpgq_gamma_retry_factor)),
                    "--mpgq-gamma-retry-steps",
                    str(int(args.mpgq_gamma_retry_steps)),
                    "--mpgq-obs-precision",
                    str(float(args.mpgq_obs_precision)),
                    "--output-dir",
                    str(out_dir),
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

    grouped: Dict[str, List[float]] = {}
    for row in results:
        label = AblationSpec(
            polish_sweeps=int(row["ablation"]["polish_sweeps"]),
            em_obs_precision=bool(row["ablation"]["em_obs_precision"]),
            damping=float(row["ablation"]["damping"]),
            prior_tau=float(row["ablation"]["prior_tau"]),
        ).label()
        ppl = _combo_ppl(row)
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
            "combo": combo.label(),
        },
        "args": vars(args),
        "results": results,
        "summary": [
            {"label": label, "mean_ppl": mu, "std_ppl": sd, "n": n} for (label, mu, sd, n) in summary_rows
        ],
    }
    (run_dir / "ablation.json").write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append(f"# Ablation: {combo.label()} (bits={int(args.bits)})")
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
