from __future__ import annotations

import argparse
import csv
import json
import os
import copy
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.combo import QuantCombo, make_combo


def _conda_run_prefix(env_name: str) -> List[str]:
    current = os.environ.get("CONDA_DEFAULT_ENV")
    if current and str(current) == str(env_name):
        return []
    return ["conda", "run", "-n", str(env_name)]


def _tail_text(path: Path, *, max_bytes: int = 16384) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - int(max_bytes))
            fh.seek(start, os.SEEK_SET)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _looks_like_transient_cuda_error(log_path: Path) -> bool:
    tail = _tail_text(log_path).lower()
    return "cudagetdevicecount" in tail and ("error 304" in tail or " 304:" in tail)


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


def _sanitize_path_component(text: str) -> str:
    return str(text).replace("/", "_").replace(":", "_")


def _run_logged(cmd: List[str], *, log_path: Path, cwd: Path, retries: int = 0, retry_sleep_s: float = 2.0) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    retries = max(int(retries), 0)
    for attempt in range(retries + 1):
        mode = "w" if attempt == 0 else "a"
        with log_path.open(mode, encoding="utf-8") as fh:
            if attempt > 0:
                fh.write("\n")
                fh.write(f"# RETRY {attempt}/{retries}\n")
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
        if rc == 0:
            return
        if attempt < retries and _looks_like_transient_cuda_error(log_path):
            time.sleep(float(retry_sleep_s))
            continue
        raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)} (see {log_path})")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_result(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = _json_load(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    repo_sha: str
    submodules: Dict[str, str]
    started_at: str


def _default_methods(bits: int) -> List[str]:
    if int(bits) == 2:
        return ["quip", "quip+mpq"]
    if int(bits) in {3, 4}:
        return ["rtn", "mpq", "gptq", "ojbkq", "awq+mpq", "awq+gptq"]
    raise ValueError(f"unsupported bits={bits}; expected 2/3/4")


def _parse_seeds(seeds: List[int]) -> List[int]:
    if not seeds:
        return [0]
    return [int(s) for s in seeds]


def _combo_ppl(row: Dict[str, Any]) -> float:
    eval_payload = row.get("eval", {})
    if not isinstance(eval_payload, dict):
        return float("nan")


def _flatten_result(row: Dict[str, Any]) -> Dict[str, Any]:
    eval_payload = row.get("eval", {})
    timing = row.get("timing", {})
    ppl = eval_payload.get("ppl", {}) if isinstance(eval_payload, dict) else {}
    if not isinstance(ppl, dict):
        ppl = {}
    model_id = row.get("model_id") or row.get("base_model") or ""
    topk_k = 0
    if isinstance(row.get("mpq_args"), dict):
        topk_k = int(row["mpq_args"].get("topk_k", 0) or 0)
    elif isinstance(row.get("mpq"), dict):
        topk_k = int(row["mpq"].get("topk_k", 0) or 0)
    elif isinstance(row.get("awq_preprocess"), dict):
        topk_k = int(row.get("topk_k", 0) or 0)
    return {
        "model_id": str(model_id),
        "backend": str(row.get("backend", "")),
        "codebook": str(row.get("codebook", row.get("codebook_impl", ""))),
        "bits": int(row.get("bits", 0) or 0),
        "preprocess": str(row.get("preprocess", "")),
        "rotation": str(row.get("rotation", "")),
        "solver": str(row.get("solver", row.get("resolved_method", ""))),
        "polish_mode": str(row.get("polish_mode", "")),
        "topk_k": int(row.get("topk_k", topk_k) or topk_k),
        "seed": int(row.get("seed", 0) or 0),
        "wikitext2_ppl": float(ppl["wikitext2"]) if "wikitext2" in ppl else float("nan"),
        "c4_ppl": float(ppl["c4"]) if "c4" in ppl else float("nan"),
        "load_s": float(timing.get("load_s", 0.0) or 0.0),
        "preprocess_s": float(timing.get("awq_preprocess_s", 0.0) or 0.0),
        "hessian_or_stats_s": float(timing.get("hessian_s", 0.0) or 0.0),
        "quantize_s": float(timing.get("quantize_s", 0.0) or 0.0),
        "hfize_s": float(timing.get("hfize_s", timing.get("save_quantized_s", 0.0)) or 0.0),
        "eval_wikitext2_s": float(timing.get("eval_wikitext2_s", 0.0) or 0.0),
        "eval_c4_s": float(timing.get("eval_c4_s", 0.0) or 0.0),
        "quantize_total_s": float(timing.get("quantize_total_s", 0.0) or 0.0),
        "eval_total_s": float(timing.get("eval_total_s", 0.0) or 0.0),
        "total_s": float(timing.get("total_s", 0.0) or 0.0),
    }
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


def _resolve_combo_matrix(bits: int, args) -> List[QuantCombo]:
    if int(bits) == 2:
        preprocesses = list(args.quant_preprocesses) if args.quant_preprocesses else ["none"]
        rotations = list(args.quant_rotations) if args.quant_rotations else ["had", "kron"]
        solvers = list(args.quant_solvers) if args.quant_solvers else ["ldlq", "vamp", "ldlq_tune10"]
        polish_modes = list(args.quant_polish_modes) if args.quant_polish_modes else ["none", "block_cd", "topk_block_cd"]
        backend = "quip"
        codebook = "e8p12"
    else:
        preprocesses = list(args.quant_preprocesses) if args.quant_preprocesses else ["none", "awq"]
        rotations = list(args.quant_rotations) if args.quant_rotations else ["none", "hadamard", "random"]
        solvers = list(args.quant_solvers) if args.quant_solvers else ["gptq", "vamp"]
        polish_modes = list(args.quant_polish_modes) if args.quant_polish_modes else ["none", "block_cd", "topk_block_cd"]
        backend = "gptqmodel"
        codebook = "grid_g128"

    combos: List[QuantCombo] = []
    for preprocess in preprocesses:
        for rotation in rotations:
            for solver in solvers:
                for polish_mode in polish_modes:
                    try:
                        combo = make_combo(
                            backend=backend,
                            codebook=codebook,
                            preprocess=preprocess,
                            rotation=rotation,
                            solver=solver,
                            polish_mode=polish_mode,
                            topk_source=str(args.quant_topk_source),
                        )
                    except ValueError:
                        continue
                    if combo not in combos:
                        combos.append(combo)
    return combos


def _default_sanity_eval_label(bits: int) -> str:
    if int(bits) == 2:
        return "backend-quip__codebook-e8p12__pre-none__rot-had__solver-vamp__polish-topk_block_cd"
    return "backend-gptqmodel__codebook-grid_g128__pre-none__rot-hadamard__solver-vamp__polish-topk_block_cd"


def _build_job_args(args, *, bits: int, label: str, combo: Optional[QuantCombo]):
    job_args = argparse.Namespace(**copy.deepcopy(vars(args)))
    is_full_eval = True
    if str(getattr(args, "sanity_mode", "none")) == "light":
        requested = list(getattr(args, "sanity_full_eval_labels", []) or [])
        eval_labels = requested if requested else [_default_sanity_eval_label(bits)]
        is_full_eval = str(label) in set(str(item) for item in eval_labels)
        if not is_full_eval:
            if int(bits) == 2:
                job_args.devset_size = min(int(args.devset_size), int(args.sanity_smoke_devset_size))
            else:
                job_args.calibration_limit = min(int(args.calibration_limit), int(args.sanity_smoke_calibration_limit))
            job_args.skip_eval = True
            job_args.eval_datasets = []
        else:
            job_args.skip_eval = False
    else:
        job_args.skip_eval = False
    return job_args, is_full_eval


def _legacy_quip_cmd(method: str, *, seed: int, seed_dir: Path, method_dir: Path, json_out: Path, args) -> List[str]:
    hess_dir = seed_dir / f"hess_{_sanitize_path_component(args.calibration_dataset)}_{_sanitize_path_component(args.calibration_split)}"
    return [
        *_conda_run_prefix(str(args.quip_env)),
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
        "--hessian-dir",
        str(hess_dir),
        "--force-hessians" if bool(args.force_hessians) else "--no-force-hessians",
        "--hessian-batch-size",
        str(int(args.hessian_batch_size)),
        "--eval-seqlen",
        str(int(args.eval_seqlen)),
        "--eval-max-eval-tokens",
        str(int(args.eval_max_eval_tokens)),
        *(["--eval-datasets", *list(args.eval_datasets)] if getattr(args, "eval_datasets", None) else []),
        "--skip-eval" if bool(getattr(args, "skip_eval", False)) else "--no-skip-eval",
        "--mpq-init",
        str(args.mpq_init),
        "--mpq-method",
        str(args.mpq_method),
        "--mpq-iters",
        str(int(args.mpq_iters)),
        "--mpq-damping",
        str(float(args.mpq_damping)),
        "--mpq-prior-tau",
        str(float(args.mpq_prior_tau)),
        "--mpq-prior-tau-scale-mode",
        str(args.mpq_prior_tau_scale_mode),
        "--mpq-gamma-start",
        str(float(args.mpq_gamma_start)),
        "--mpq-gamma-retry-factor",
        str(float(args.mpq_gamma_retry_factor)),
        "--mpq-gamma-retry-steps",
        str(int(args.mpq_gamma_retry_steps)),
        "--mpq-obs-precision",
        str(float(args.mpq_obs_precision)),
        "--mpq-em-obs-precision" if bool(args.mpq_em_obs_precision) else "--no-mpq-em-obs-precision",
        "--mpq-blocks-per-chunk",
        str(int(args.mpq_blocks_per_chunk)),
        "--mpq-block-rows",
        str(int(args.mpq_block_rows)),
        "--mpq-polish-sweeps",
        str(int(args.mpq_polish_sweeps)),
        "--mpq-polish-metric",
        str(args.mpq_polish_metric),
        "--mpq-polish-max-blocks",
        str(int(args.mpq_polish_max_blocks)),
        "--mpq-metric-rows-per-chunk",
        str(int(args.mpq_metric_rows_per_chunk)),
        "--mpq-topk-k",
        str(int(args.quant_topk_k)),
        "--output-dir",
        str(method_dir),
        "--json-out",
        str(json_out),
    ]


def _combo_quip_cmd(combo: QuantCombo, *, seed: int, seed_dir: Path, method_dir: Path, json_out: Path, args) -> List[str]:
    hess_dir = seed_dir / f"hess_{_sanitize_path_component(args.calibration_dataset)}_{_sanitize_path_component(args.calibration_split)}"
    return [
        *_conda_run_prefix(str(args.quip_env)),
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
        "--hessian-dir",
        str(hess_dir),
        "--force-hessians" if bool(args.force_hessians) else "--no-force-hessians",
        "--hessian-batch-size",
        str(int(args.hessian_batch_size)),
        "--eval-seqlen",
        str(int(args.eval_seqlen)),
        "--eval-max-eval-tokens",
        str(int(args.eval_max_eval_tokens)),
        *(["--eval-datasets", *list(args.eval_datasets)] if getattr(args, "eval_datasets", None) else []),
        "--skip-eval" if bool(getattr(args, "skip_eval", False)) else "--no-skip-eval",
        "--mpq-init",
        str(args.mpq_init),
        "--mpq-method",
        str(args.mpq_method),
        "--mpq-iters",
        str(int(args.mpq_iters)),
        "--mpq-damping",
        str(float(args.mpq_damping)),
        "--mpq-prior-tau",
        str(float(args.mpq_prior_tau)),
        "--mpq-prior-tau-scale-mode",
        str(args.mpq_prior_tau_scale_mode),
        "--mpq-gamma-start",
        str(float(args.mpq_gamma_start)),
        "--mpq-gamma-retry-factor",
        str(float(args.mpq_gamma_retry_factor)),
        "--mpq-gamma-retry-steps",
        str(int(args.mpq_gamma_retry_steps)),
        "--mpq-obs-precision",
        str(float(args.mpq_obs_precision)),
        "--mpq-em-obs-precision" if bool(args.mpq_em_obs_precision) else "--no-mpq-em-obs-precision",
        "--mpq-blocks-per-chunk",
        str(int(args.mpq_blocks_per_chunk)),
        "--mpq-block-rows",
        str(int(args.mpq_block_rows)),
        "--mpq-polish-sweeps",
        str(int(args.mpq_polish_sweeps)),
        "--mpq-polish-metric",
        str(args.mpq_polish_metric),
        "--mpq-polish-max-blocks",
        str(int(args.mpq_polish_max_blocks)),
        "--mpq-metric-rows-per-chunk",
        str(int(args.mpq_metric_rows_per_chunk)),
        "--mpq-topk-k",
        str(int(args.quant_topk_k)),
        "--output-dir",
        str(method_dir),
        "--json-out",
        str(json_out),
    ]


def _legacy_gptq_cmd(method: str, *, seed: int, method_dir: Path, json_out: Path, bits: int, args) -> List[str]:
    return [
        *_conda_run_prefix(str(args.gptq_env)),
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
        *(["--eval-datasets", *list(args.eval_datasets)] if getattr(args, "eval_datasets", None) else []),
        "--skip-eval" if bool(getattr(args, "skip_eval", False)) else "--no-skip-eval",
        "--awq-batch-size",
        str(int(getattr(args, "awq_batch_size", 1))),
        "--mpq-iters",
        str(int(args.mpq_iters)),
        "--mpq-damping",
        str(float(args.mpq_damping)),
        "--mpq-prior-tau",
        str(float(args.mpq_prior_tau)),
        "--mpq-polish-sweeps",
        str(int(args.mpq_polish_sweeps)),
        "--mpq-em-obs-precision" if bool(args.mpq_em_obs_precision) else "--no-mpq-em-obs-precision",
        "--mpq-block-rows",
        str(int(args.mpq_block_rows)),
        "--mpq-rotation-seed",
        str(int(args.mpq_rotation_seed)),
        "--mpq-gamma-start",
        str(float(args.mpq_gamma_start)),
        "--mpq-gamma-retry-factor",
        str(float(args.mpq_gamma_retry_factor)),
        "--mpq-gamma-retry-steps",
        str(int(args.mpq_gamma_retry_steps)),
        "--mpq-obs-precision",
        str(float(args.mpq_obs_precision)),
        "--ojbkq-random-trials",
        str(int(args.ojbkq_random_trials)),
        "--ojbkq-alpha-mode",
        str(args.ojbkq_alpha_mode),
        *(["--ojbkq-alpha-override", str(float(args.ojbkq_alpha_override))] if args.ojbkq_alpha_override is not None else []),
        "--ojbkq-alpha-scale",
        str(float(args.ojbkq_alpha_scale)),
        *(["--ojbkq-sample-topk", str(int(args.ojbkq_sample_topk))] if args.ojbkq_sample_topk is not None else []),
        *(["--gptqmodel-extra-site-packages", str(args.gptqmodel_extra_site_packages)] if args.gptqmodel_extra_site_packages else []),
        "--output-dir",
        str(method_dir),
        "--json-out",
        str(json_out),
    ]


def _combo_gptq_cmd(combo: QuantCombo, *, seed: int, method_dir: Path, json_out: Path, bits: int, args) -> List[str]:
    return [
        *_conda_run_prefix(str(args.gptq_env)),
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
        *(["--eval-datasets", *list(args.eval_datasets)] if getattr(args, "eval_datasets", None) else []),
        "--skip-eval" if bool(getattr(args, "skip_eval", False)) else "--no-skip-eval",
        "--awq-batch-size",
        str(int(getattr(args, "awq_batch_size", 1))),
        "--mpq-iters",
        str(int(args.mpq_iters)),
        "--mpq-damping",
        str(float(args.mpq_damping)),
        "--mpq-prior-tau",
        str(float(args.mpq_prior_tau)),
        "--mpq-polish-sweeps",
        str(int(args.mpq_polish_sweeps)),
        "--mpq-em-obs-precision" if bool(args.mpq_em_obs_precision) else "--no-mpq-em-obs-precision",
        "--mpq-block-rows",
        str(int(args.mpq_block_rows)),
        "--mpq-rotation-seed",
        str(int(args.mpq_rotation_seed)),
        "--mpq-gamma-start",
        str(float(args.mpq_gamma_start)),
        "--mpq-gamma-retry-factor",
        str(float(args.mpq_gamma_retry_factor)),
        "--mpq-gamma-retry-steps",
        str(int(args.mpq_gamma_retry_steps)),
        "--mpq-obs-precision",
        str(float(args.mpq_obs_precision)),
        "--ojbkq-random-trials",
        str(int(args.ojbkq_random_trials)),
        "--ojbkq-alpha-mode",
        str(args.ojbkq_alpha_mode),
        *(["--ojbkq-alpha-override", str(float(args.ojbkq_alpha_override))] if args.ojbkq_alpha_override is not None else []),
        "--ojbkq-alpha-scale",
        str(float(args.ojbkq_alpha_scale)),
        *(["--ojbkq-sample-topk", str(int(args.ojbkq_sample_topk))] if args.ojbkq_sample_topk is not None else []),
        *(["--gptqmodel-extra-site-packages", str(args.gptqmodel_extra_site_packages)] if args.gptqmodel_extra_site_packages else []),
        "--output-dir",
        str(method_dir),
        "--json-out",
        str(json_out),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--quant-preprocesses", nargs="*", default=None)
    parser.add_argument("--quant-rotations", nargs="*", default=None)
    parser.add_argument("--quant-solvers", nargs="*", default=None)
    parser.add_argument("--quant-polish-modes", nargs="*", default=None)
    parser.add_argument("--quant-topk-source", type=str, default="aux_vamp")
    parser.add_argument("--quant-topk-k", type=int, default=8)

    parser.add_argument("--model-id", type=str, default=None, help="HF model id for GPTQModel backend (bits=3/4)")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)
    parser.add_argument("--eval-datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument(
        "--gptqmodel-extra-site-packages",
        type=str,
        default=None,
        help="Optional extra site-packages path for GPTQModel deps when running backend_gptq without installing GPTQModel.",
    )

    parser.add_argument("--mpq-iters", type=int, default=16)
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

    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)

    parser.add_argument("--base-model", type=str, default=None, help="HF model id for QUIP# backend (bits=2)")
    parser.add_argument("--codebook", type=str, default="E8P12")
    parser.add_argument("--devset-size", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-proc", type=int, default=1)
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--force-hessians", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mpq-init", type=str, default="random", choices=["random", "rtn", "ldlq"])
    parser.add_argument("--mpq-method", type=str, default="vamp", choices=["vamp", "vamp-topk-polish"])
    parser.add_argument("--mpq-prior-tau-scale-mode", type=str, default="absmax", choices=["none", "absmax", "rms", "mean_abs"])
    parser.add_argument("--mpq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--mpq-polish-metric", type=str, default="gram", choices=["gram", "diag", "isotropic"])
    parser.add_argument("--mpq-polish-max-blocks", type=int, default=0)
    parser.add_argument("--mpq-metric-rows-per-chunk", type=int, default=0)

    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--gptq-env", type=str, default="mpq-gptq")
    parser.add_argument("--quip-env", type=str, default="mpq-quip")
    parser.add_argument("--backend-retries", type=int, default=5)
    parser.add_argument("--backend-retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sanity-mode", type=str, default="none", choices=["none", "light"])
    parser.add_argument("--sanity-full-eval-labels", nargs="*", default=None)
    parser.add_argument("--sanity-smoke-calibration-limit", type=int, default=8)
    parser.add_argument("--sanity-smoke-devset-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    bits = int(args.bits)
    seeds = _parse_seeds(list(args.seeds))
    methods = args.methods if args.methods is not None and len(args.methods) else None

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

    if methods is None:
        work_items: List[tuple[str, Optional[QuantCombo]]] = [(combo.label(), combo) for combo in _resolve_combo_matrix(bits, args)]
    else:
        work_items = [(method, None) for method in methods]

    all_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for label, combo in work_items:
            method_dir = seed_dir / str(label).replace("/", "_")
            method_dir.mkdir(parents=True, exist_ok=True)
            json_out = method_dir / "backend_results.json"
            log_path = method_dir / "run.log"
            job_args, is_full_eval = _build_job_args(args, bits=bits, label=label, combo=combo)

            if bool(args.resume_existing):
                existing = _load_existing_result(json_out)
                if existing is not None:
                    print(f"SKIP existing result: {json_out}")
                    all_results.append(existing)
                    continue

            if combo is None and bits == 2:
                cmd = _legacy_quip_cmd(label, seed=seed, seed_dir=seed_dir, method_dir=method_dir, json_out=json_out, args=job_args)
            elif combo is None:
                cmd = _legacy_gptq_cmd(label, seed=seed, method_dir=method_dir, json_out=json_out, bits=bits, args=job_args)
            elif bits == 2:
                cmd = _combo_quip_cmd(combo, seed=seed, seed_dir=seed_dir, method_dir=method_dir, json_out=json_out, args=job_args)
            else:
                cmd = _combo_gptq_cmd(combo, seed=seed, method_dir=method_dir, json_out=json_out, bits=bits, args=job_args)

            if args.dry_run:
                print(f"# {'full-eval' if is_full_eval else 'smoke'}")
                print("$ " + " ".join(shlex.quote(x) for x in cmd))
                continue

            _run_logged(
                cmd,
                log_path=log_path,
                cwd=REPO_ROOT,
                retries=int(args.backend_retries),
                retry_sleep_s=float(args.backend_retry_sleep),
            )
            all_results.append(_json_load(json_out))

            # Clean up model artifacts to save disk space.
            for _subdir_name in ("ckpt", "hf", "quantized"):
                _subdir = method_dir / _subdir_name
                if _subdir.is_dir():
                    shutil.rmtree(_subdir, ignore_errors=True)

        # Clean up shared hessian directory for this seed.
        for _hess_dir in seed_dir.glob("hess_*"):
            if _hess_dir.is_dir():
                shutil.rmtree(_hess_dir, ignore_errors=True)

    payload: Dict[str, Any] = {
        "meta": asdict(meta),
        "args": vars(args),
        "results": all_results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    flat_rows = [_flatten_result(row) for row in all_results]
    jsonl_path = run_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in flat_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = run_dir / "results.csv"
    if flat_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    lines = [
        f"# Benchmark (bits={bits})",
        "",
        "| Model | Backend | Codebook | Preprocess | Rotation | Solver | Polish | Seed | Wikitext2 PPL | C4 PPL | Quantize Total (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        flat_rows,
        key=lambda item: (
            str(item.get("model_id", "")),
            str(item.get("backend", "")),
            str(item.get("codebook", "")),
            str(item.get("preprocess", "")),
            str(item.get("rotation", "")),
            str(item.get("solver", "")),
            str(item.get("polish_mode", "")),
            int(item.get("seed", 0)),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model_id", "")),
                    str(row.get("backend", "")),
                    str(row.get("codebook", "")),
                    str(row.get("preprocess", "")),
                    str(row.get("rotation", "")),
                    str(row.get("solver", "")),
                    str(row.get("polish_mode", "")),
                    str(int(row.get("seed", 0))),
                    f"{float(row.get('wikitext2_ppl', float('nan'))):.4f}",
                    f"{float(row.get('c4_ppl', float('nan'))):.4f}",
                    f"{float(row.get('quantize_total_s', 0.0)):.2f}",
                ]
            )
            + " |"
        )
    md_path = run_dir / "benchmark.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {out_path}")
    print(f"WROTE {jsonl_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {md_path}")


if __name__ == "__main__":
    main()
