from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"

_SUFFIXES = ("qkv", "o", "up", "down", "layernorm")


def _apply_patches() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "patches" / "apply_patches.py")],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("apply_patches.py failed")


def _ensure_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    existing = env.get("PYTHONPATH", "")
    prefix = str(REPO_ROOT)
    env["PYTHONPATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


def _ensure_ld_library_path(env: Dict[str, str]) -> Dict[str, str]:
    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return env
    lib_dir = str(Path(conda_prefix) / "lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env


def _run(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str]) -> float:
    t0 = time.time()
    proc = subprocess.run(list(cmd), cwd=str(cwd), env=env, check=False)
    dt = float(time.time() - t0)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return dt


def _touch_skip_files(save_dir: Path, *, skip_layers: Iterable[int]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx in skip_layers:
        for suffix in _SUFFIXES:
            (save_dir / f"{int(idx)}_{suffix}.pt").touch(exist_ok=True)


def _load_num_layers(model_id: str) -> int:
    cfg = AutoConfig.from_pretrained(model_id)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    raise RuntimeError(f"cannot infer number of layers from config for {model_id!r}")


def _parse_layers(layers: Sequence[int] | None, *, num_layers: int) -> List[int]:
    if not layers:
        layers = [0, 1]
    out = sorted({int(x) for x in layers})
    for x in out:
        if x < 0 or x >= num_layers:
            raise ValueError(f"layer index out of range: {x} (num_layers={num_layers})")
    return out


@dataclass(frozen=True)
class Timing:
    overhead_s: float
    quip_total_s: float
    mpq_total_s: float
    quip_quant_s: float
    mpq_quant_s: float
    ratio: float


def _estimate_quant_times(overhead_s: float, quip_total_s: float, mpq_total_s: float) -> Timing:
    if not math.isfinite(float(quip_total_s)):
        quip_quant_s = float("nan")
    else:
        quip_quant_s = max(0.0, float(quip_total_s - overhead_s))
    mpq_quant_s = max(0.0, float(mpq_total_s - overhead_s))
    ratio = float("nan") if not math.isfinite(quip_quant_s) or quip_quant_s <= 0 else float(mpq_quant_s / quip_quant_s)
    return Timing(
        overhead_s=float(overhead_s),
        quip_total_s=float(quip_total_s),
        mpq_total_s=float(mpq_total_s),
        quip_quant_s=float(quip_quant_s),
        mpq_quant_s=float(mpq_quant_s),
        ratio=float(ratio),
    )


def _load_reference_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid reference json: {path}")
    return payload


def _collect_mpq_stats(save_dir: Path, *, layers: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in layers:
        for suffix in ("qkv", "o", "up", "down"):
            path = save_dir / f"{int(idx)}_{suffix}.pt"
            payload = torch.load(path, map_location="cpu")
            stats = payload.get("mpq_stats")
            if not isinstance(stats, dict):
                raise RuntimeError(f"missing mpq_stats in {path}")
            ldlq_mean = float(stats.get("exact_surrogate_ldlq_mean", float("nan")))
            mpq_mean = float(stats.get("exact_surrogate_after_mean", float("nan")))
            rows.append(
                {
                    "layer": int(idx),
                    "name": str(stats.get("module", f"{idx}_{suffix}")),
                    "m": int(stats.get("m", 0) or 0),
                    "n": int(stats.get("n", 0) or 0),
                    "ldlq_mean": ldlq_mean,
                    "mpq_mean": mpq_mean,
                    "delta_mean": float(mpq_mean - ldlq_mean),
                    "violation_chunks": int(stats.get("surrogate_chunk_violation_count", 0) or 0),
                    "max_delta": float(stats.get("surrogate_chunk_violation_max_delta", 0.0) or 0.0),
                    "max_ratio": float(stats.get("surrogate_chunk_violation_max_ratio", 0.0) or 0.0),
                    "solver_time_s": float(stats.get("solver_time_s", 0.0) or 0.0),
                }
            )
    return rows


def _layer_aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        layer = int(r["layer"])
        m = int(r.get("m", 0) or 0)
        if layer not in by_layer:
            by_layer[layer] = {
                "layer": layer,
                "ldlq_total": 0.0,
                "mpq_total": 0.0,
                "m_total": 0,
                "violation_chunks": 0,
                "max_delta": 0.0,
                "max_ratio": 0.0,
            }
        agg = by_layer[layer]
        ldlq_total = float(r["ldlq_mean"]) * max(m, 1)
        mpq_total = float(r["mpq_mean"]) * max(m, 1)
        agg["ldlq_total"] += ldlq_total
        agg["mpq_total"] += mpq_total
        agg["m_total"] += int(m)
        agg["violation_chunks"] += int(r.get("violation_chunks", 0) or 0)
        agg["max_delta"] = max(float(agg["max_delta"]), float(r.get("max_delta", 0.0) or 0.0))
        agg["max_ratio"] = max(float(agg["max_ratio"]), float(r.get("max_ratio", 0.0) or 0.0))

    out: List[Dict[str, Any]] = []
    for layer in sorted(by_layer.keys()):
        agg = by_layer[layer]
        m_total = int(agg["m_total"]) or 1
        out.append(
            {
                "layer": int(layer),
                "ldlq_mean": float(agg["ldlq_total"]) / float(m_total),
                "mpq_mean": float(agg["mpq_total"]) / float(m_total),
                "delta_mean": float(agg["mpq_total"] - agg["ldlq_total"]) / float(m_total),
                "violation_chunks": int(agg["violation_chunks"]),
                "max_delta": float(agg["max_delta"]),
                "max_ratio": float(agg["max_ratio"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="QUIP# smoke: compare LDLQ vs MPQ proxy (surrogate) losses layer-wise.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--hessian-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codebook", type=str, default="E8P12")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="Layer indices to quantize (default: 0 1).")
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--quip-tune-iters", type=int, default=0, help="Pass through QUIP# --quip_tune_iters (0 disables QUIP tune sweeps).")
    parser.add_argument("--mpq-init", type=str, default="rtn", choices=["random", "rtn", "ldlq"])
    parser.add_argument("--mpq-iters", type=int, default=4)
    parser.add_argument("--mpq-damping", type=float, default=0.5)
    parser.add_argument("--mpq-prior-tau", type=float, default=0.0)
    parser.add_argument("--mpq-block-rows", type=int, default=128, help="MPQ row chunk size (passed to QUIP# --mpq_block_rows).")
    parser.add_argument("--mpq-blocks-per-chunk", type=int, default=16, help="MPQ block chunk size (passed to QUIP# --mpq_blocks_per_chunk).")
    parser.add_argument("--mpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpq-polish-metric", type=str, default="gram", choices=["gram", "isotropic", "diag"])
    parser.add_argument("--skip-quip-baseline", action=argparse.BooleanOptionalAction, default=False, help="Skip the standalone QUIP pass and only run MPQ (LDLQ baseline is still available via mpq_stats).")
    parser.add_argument("--reference-json", type=str, default=None, help="Reuse overhead/quip timing from a previous smoke json and only rerun MPQ.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    _apply_patches()

    env = _ensure_pythonpath(dict(os.environ))
    env = _ensure_ld_library_path(env)

    num_layers = _load_num_layers(str(args.base_model))
    layers = _parse_layers(args.layers, num_layers=num_layers)
    skip_layers = sorted(set(range(num_layers)).difference(layers))

    out_root = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "runs" / f"smoke_proxy_{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    overhead_dir = out_root / "overhead_all_skipped"
    quip_dir = out_root / "quip_layers"
    mpq_dir = out_root / "mpq_layers"
    _touch_skip_files(mpq_dir, skip_layers=skip_layers)

    base_cmd = [
        sys.executable,
        "-m",
        "quantize_llama.quantize_finetune_llama",
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
        "--sample_proc",
        str(int(args.sample_proc)),
        "--scale_override",
        str(float(args.scale_override)),
        "--quip_tune_iters",
        str(int(args.quip_tune_iters)),
        "--calibration_dataset",
        str(args.calibration_dataset),
        "--calibration_split",
        str(args.calibration_split),
        "--base_model",
        str(args.base_model),
        "--codebook",
        str(args.codebook),
        "--hessian_path",
        str(Path(args.hessian_dir).resolve()),
    ]

    if args.reference_json:
        reference = _load_reference_json(Path(args.reference_json).resolve())
        timing_ref = reference.get("timing", {})
        paths_ref = reference.get("paths", {})
        overhead_s = float(timing_ref["overhead_s"])
        quip_total_s = float(timing_ref["quip_total_s"])
        overhead_dir = Path(paths_ref["overhead_dir"])
        quip_dir = Path(paths_ref["quip_dir"])
    else:
        _touch_skip_files(overhead_dir, skip_layers=range(num_layers))
        _touch_skip_files(quip_dir, skip_layers=skip_layers)
        overhead_s = _run(base_cmd + ["--solver", "quip", "--save_path", str(overhead_dir)], cwd=QUIP_ROOT, env=env)
        if bool(args.skip_quip_baseline):
            quip_total_s = float("nan")
        else:
            quip_total_s = _run(base_cmd + ["--solver", "quip", "--save_path", str(quip_dir)], cwd=QUIP_ROOT, env=env)

    # Quantize only the requested layers for MPQ as well; otherwise QUIP# will
    # happily run the full-model pass.
    _touch_skip_files(mpq_dir, skip_layers=skip_layers)

    mpq_cmd = base_cmd + [
        "--solver",
        "mpq",
        "--save_path",
        str(mpq_dir),
        "--mpq_init",
        str(args.mpq_init),
        "--mpq_iters",
        str(int(args.mpq_iters)),
        "--mpq_damping",
        str(float(args.mpq_damping)),
        "--mpq_prior_tau",
        str(float(args.mpq_prior_tau)),
        "--mpq_block_rows",
        str(int(args.mpq_block_rows)),
        "--mpq_blocks_per_chunk",
        str(int(args.mpq_blocks_per_chunk)),
        "--mpq_polish_sweeps",
        str(int(args.mpq_polish_sweeps)),
        "--mpq_polish_metric",
        str(args.mpq_polish_metric),
    ]
    mpq_total_s = _run(mpq_cmd, cwd=QUIP_ROOT, env=env)

    timing = _estimate_quant_times(overhead_s, quip_total_s, mpq_total_s)
    rows = _collect_mpq_stats(mpq_dir, layers=layers)
    layer_rows = _layer_aggregate(rows)

    violations = [r for r in rows if int(r.get("violation_chunks", 0) or 0) > 0]
    summary = {
        "base_model": str(args.base_model),
        "seed": int(args.seed),
        "layers": list(layers),
        "timing": {
            "overhead_s": timing.overhead_s,
            "quip_total_s": timing.quip_total_s,
            "mpq_total_s": timing.mpq_total_s,
            "quip_quant_s": timing.quip_quant_s,
            "mpq_quant_s": timing.mpq_quant_s,
            "ratio_mpq_over_quip": timing.ratio,
        },
        "modules": rows,
        "layers_agg": layer_rows,
        "violations": violations,
        "violation_modules": len(violations),
        "paths": {
            "out_root": str(out_root),
            "overhead_dir": str(overhead_dir),
            "quip_dir": str(quip_dir),
            "mpq_dir": str(mpq_dir),
        },
    }

    json_out = Path(args.json_out).resolve() if args.json_out else (out_root / "smoke_proxy.json")
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {json_out}")
    if math.isfinite(timing.ratio):
        print(f"timing ratio mpq/quip ~= {timing.ratio:.2f} (quant-only, overhead-subtracted)")
    else:
        print("timing ratio mpq/quip ~= n/a (skipped quip baseline)")
    print(f"violation modules: {len(violations)} / {len(rows)}")
    for r in layer_rows:
        print(
            f"layer {r['layer']:02d}: ldlq_mean={r['ldlq_mean']:.6g} mpq_mean={r['mpq_mean']:.6g} delta={r['delta_mean']:.6g} viol_chunks={r['violation_chunks']} max_ratio={r['max_ratio']:.3g}"
        )


if __name__ == "__main__":
    main()
