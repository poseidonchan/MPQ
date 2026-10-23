from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .quantization import (
    QuantizationConfig,
    build_quantize_command,
    find_repo_root,
    format_command,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpq",
        description="Quantize dense Hugging Face causal LMs with MPQ.",
    )
    parser.add_argument("--version", action="version", version=f"MPQ {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize_parser = subparsers.add_parser(
        "quantize",
        help="quantize a Hugging Face model or local checkpoint",
        epilog="Place backend-specific options after '--', for example: -- --device cuda:1",
    )
    quantize_parser.add_argument("model", help="Hugging Face model id or local model path")
    quantize_parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    quantize_parser.add_argument("--output-dir", type=Path, required=True)
    quantize_parser.add_argument("--group-size", type=int, default=128)
    quantize_parser.add_argument(
        "--preprocess",
        choices=("auto", "none", "awq"),
        default="auto",
        help="auto selects none for E8P12 and AWQ for scalar quantization",
    )
    quantize_parser.add_argument(
        "--sym",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use symmetric scalar quantization (default: asymmetric)",
    )
    quantize_parser.add_argument(
        "--init",
        choices=("auto", "float", "rtn", "random", "ldlq"),
        default="auto",
    )
    quantize_parser.add_argument("--iters", type=int, default=16)
    quantize_parser.add_argument("--damping", type=float, default=0.5)
    quantize_parser.add_argument("--prior-tau", type=float, default=None)
    quantize_parser.add_argument("--polish-sweeps", type=int, default=3)
    quantize_parser.add_argument("--calibration-dataset", default="c4")
    quantize_parser.add_argument("--calibration-samples", type=int, default=128)
    quantize_parser.add_argument("--calibration-seqlen", type=int, default=2048)
    quantize_parser.add_argument("--seed", type=int, default=0)
    quantize_parser.add_argument(
        "--eval",
        action="store_true",
        help="evaluate WikiText-2 and C4 perplexity after quantization",
    )
    quantize_parser.add_argument("--skip-save", action="store_true")
    quantize_parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="MPQ source checkout; defaults to MPQ_REPO_ROOT or this installation",
    )
    quantize_parser.add_argument(
        "--backend-env",
        default=None,
        help="override the conda environment selected for the resolved backend",
    )
    quantize_parser.add_argument(
        "--no-conda",
        action="store_true",
        help="run the backend with the current Python interpreter",
    )
    quantize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved backend command without loading the model",
    )
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check source and backend checkouts, conda environments, and GPU tools",
    )
    doctor_parser.add_argument("--repo-root", type=Path, default=None)
    return parser


def _strip_separator(values: Sequence[str]) -> tuple[str, ...]:
    if values and values[0] == "--":
        return tuple(values[1:])
    return tuple(values)


def _run_quantize(args: argparse.Namespace) -> int:
    config = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        preprocess=args.preprocess,
        symmetric=args.sym,
        init=args.init,
        iters=args.iters,
        damping=args.damping,
        prior_tau=args.prior_tau,
        polish_sweeps=args.polish_sweeps,
        calibration_dataset=args.calibration_dataset,
        calibration_samples=args.calibration_samples,
        calibration_seqlen=args.calibration_seqlen,
        seed=args.seed,
        evaluate=args.eval,
        save=not args.skip_save,
        backend_env=args.backend_env,
    )
    command = build_quantize_command(
        args.model,
        args.output_dir,
        config,
        backend_args=_strip_separator(args.backend_args),
        repo_root=args.repo_root,
        use_conda=not args.no_conda,
    )
    if args.dry_run:
        print(format_command(command))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mpq] backend={config.resolve().backend} bits={config.bits}", flush=True)
    print(f"[mpq] output={args.output_dir}", flush=True)
    subprocess.run(command, check=True)
    result_path = args.output_dir / "backend_results.json"
    if not result_path.is_file():
        raise RuntimeError(f"backend did not write {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    paths = payload.get("paths", {}) if isinstance(payload, dict) else {}
    artifact = paths.get("hf_dir") or paths.get("native_dir") or args.output_dir
    print(f"[mpq] quantized model: {artifact}")
    print(f"[mpq] run metadata: {result_path}")
    return 0


def _conda_env_exists(name: str) -> bool:
    if shutil.which("conda") is None:
        return False
    proc = subprocess.run(
        ["conda", "env", "list", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False
    return any(Path(path).name == name for path in payload.get("envs", []))


def _run_doctor(args: argparse.Namespace) -> int:
    root = find_repo_root(args.repo_root)
    gptq_env = os.environ.get("MPQ_GPTQ_ENV") or "mpq-gptq"
    quip_env = os.environ.get("MPQ_QUIP_ENV") or "mpq-quip"
    gptq_checkout_ok = (root / "third_party" / "GPTQModel" / ".git").exists()
    quip_checkout_ok = (root / "third_party" / "quip-sharp" / ".git").exists()
    fht_checkout_ok = (
        root / "third_party" / "fast-hadamard-transform" / ".git"
    ).exists()
    gptq_env_ok = _conda_env_exists(gptq_env)
    quip_env_ok = _conda_env_exists(quip_env)
    checks = [
        ("source checkout", True, root),
        ("GPTQModel checkout", gptq_checkout_ok, root / "third_party" / "GPTQModel"),
        ("QUIP# checkout", quip_checkout_ok, root / "third_party" / "quip-sharp"),
        (
            "fast Hadamard checkout",
            fht_checkout_ok,
            root / "third_party" / "fast-hadamard-transform",
        ),
        (f"{gptq_env} environment", gptq_env_ok, f"conda env {gptq_env}"),
        (f"{quip_env} environment", quip_env_ok, f"conda env {quip_env}"),
        ("nvidia-smi", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "not found"),
    ]
    for label, ok, detail in checks:
        print(f"{'OK' if ok else 'MISSING':7} {label}: {detail}")
    gptq_ready = gptq_checkout_ok and gptq_env_ok
    quip_ready = quip_checkout_ok and quip_env_ok
    if not fht_checkout_ok or not (gptq_ready or quip_ready):
        print(
            "Install at least one complete backend with "
            "bash install.sh --gptq-only or --quip-only."
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    backend_args: tuple[str, ...] = ()
    if "--" in raw_argv:
        separator = raw_argv.index("--")
        backend_args = tuple(raw_argv[separator + 1 :])
        raw_argv = raw_argv[:separator]
    args = parser.parse_args(raw_argv)
    args.backend_args = backend_args
    try:
        if args.command == "quantize":
            return _run_quantize(args)
        if args.command == "doctor":
            return _run_doctor(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"mpq: error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
