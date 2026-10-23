from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_BITS = (2, 3, 4)


@dataclass(frozen=True)
class ResolvedQuantizationConfig:
    bits: int
    backend: str
    backend_env: str
    codebook: str
    preprocess: str
    rotation: str
    solver: str
    polish_mode: str
    group_size: int
    symmetric: bool | None
    init: str
    iters: int
    damping: float
    prior_tau: float
    polish_sweeps: int
    calibration_dataset: str
    calibration_samples: int
    calibration_seqlen: int
    seed: int
    evaluate: bool
    save: bool


@dataclass(frozen=True)
class QuantizationConfig:
    """User-facing MPQ configuration.

    Defaults select the paper-quality recipe for each bit width. Advanced
    backend flags can still be supplied by the CLI or ``quantize`` function.
    """

    bits: int = 4
    group_size: int = 128
    preprocess: str = "auto"
    symmetric: bool | None = None
    init: str = "auto"
    iters: int = 16
    damping: float = 0.5
    prior_tau: float | None = None
    polish_sweeps: int = 3
    calibration_dataset: str = "c4"
    calibration_samples: int = 128
    calibration_seqlen: int = 2048
    seed: int = 0
    evaluate: bool = False
    save: bool = True
    backend_env: str | None = None

    def resolve(self) -> ResolvedQuantizationConfig:
        bits = int(self.bits)
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}; got {bits}")
        if int(self.group_size) <= 0 and int(self.group_size) != -1:
            raise ValueError("group_size must be positive or -1")
        if int(self.calibration_samples) <= 0:
            raise ValueError("calibration_samples must be positive")
        if int(self.calibration_seqlen) <= 0:
            raise ValueError("calibration_seqlen must be positive")
        if int(self.iters) <= 0:
            raise ValueError("iters must be positive")
        if not 0.0 <= float(self.damping) <= 1.0:
            raise ValueError("damping must be in [0, 1]")
        if int(self.polish_sweeps) < 0:
            raise ValueError("polish_sweeps must be non-negative")
        preprocess = str(self.preprocess).strip().lower()
        if preprocess == "auto":
            preprocess = "none" if bits == 2 else "awq"
        if preprocess not in {"none", "awq"}:
            raise ValueError("preprocess must be one of: auto, none, awq")

        init = str(self.init).strip().lower()
        if init == "auto":
            init = "rtn"
        if init not in {"float", "rtn", "random", "ldlq"}:
            raise ValueError("init must be one of: auto, float, rtn, random, ldlq")

        if bits == 2:
            return ResolvedQuantizationConfig(
                bits=bits,
                backend="quip",
                backend_env=(
                    self.backend_env
                    or os.environ.get("MPQ_QUIP_ENV")
                    or "mpq-quip"
                ),
                codebook="e8p12",
                preprocess=preprocess,
                rotation="had",
                solver="vamp",
                polish_mode="block_cd" if self.polish_sweeps else "none",
                group_size=int(self.group_size),
                symmetric=None,
                init=init,
                iters=int(self.iters),
                damping=float(self.damping),
                prior_tau=0.2 if self.prior_tau is None else float(self.prior_tau),
                polish_sweeps=int(self.polish_sweeps),
                calibration_dataset=str(self.calibration_dataset),
                calibration_samples=int(self.calibration_samples),
                calibration_seqlen=int(self.calibration_seqlen),
                seed=int(self.seed),
                evaluate=bool(self.evaluate),
                save=bool(self.save),
            )

        return ResolvedQuantizationConfig(
            bits=bits,
            backend="gptqmodel",
            backend_env=(
                self.backend_env
                or os.environ.get("MPQ_GPTQ_ENV")
                or "mpq-gptq"
            ),
            codebook="grid_g128",
            preprocess=preprocess,
            rotation="internal_hadamard",
            solver="vamp",
            polish_mode="block_cd" if self.polish_sweeps else "none",
            group_size=int(self.group_size),
            symmetric=False if self.symmetric is None else bool(self.symmetric),
            init=init,
            iters=int(self.iters),
            damping=float(self.damping),
            prior_tau=0.2 if self.prior_tau is None else float(self.prior_tau),
            polish_sweeps=int(self.polish_sweeps),
            calibration_dataset=str(self.calibration_dataset),
            calibration_samples=int(self.calibration_samples),
            calibration_seqlen=int(self.calibration_seqlen),
            seed=int(self.seed),
            evaluate=bool(self.evaluate),
            save=bool(self.save),
        )


def find_repo_root(repo_root: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root))
    if os.environ.get("MPQ_REPO_ROOT"):
        candidates.append(Path(os.environ["MPQ_REPO_ROOT"]))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[1])

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "scripts" / "backend_gptq.py").is_file() and (
            root / "scripts" / "backend_quip.py"
        ).is_file():
            return root
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise RuntimeError(
        "Could not locate an MPQ source checkout. Set MPQ_REPO_ROOT or pass "
        f"repo_root. Searched: {searched}"
    )


def _python_prefix(
    *,
    backend_env: str,
    python_executable: str | None,
    use_conda: bool,
) -> list[str]:
    if python_executable:
        return [str(python_executable)]
    if not use_conda or os.environ.get("CONDA_DEFAULT_ENV") == backend_env:
        return [sys.executable]
    if shutil.which("conda") is None:
        raise RuntimeError(
            f"The {backend_env!r} environment is required, but conda was not found. "
            "Activate the environment or pass use_conda=False."
        )
    return ["conda", "run", "--no-capture-output", "-n", backend_env, "python"]


def build_quantize_command(
    model: str,
    output_dir: str | os.PathLike[str],
    config: QuantizationConfig | None = None,
    *,
    backend_args: Sequence[str] = (),
    repo_root: str | os.PathLike[str] | None = None,
    python_executable: str | None = None,
    use_conda: bool = True,
) -> list[str]:
    """Build the backend command without loading Torch or a model."""

    if not str(model).strip():
        raise ValueError("model must be a Hugging Face model id or local path")
    output = Path(output_dir).expanduser()
    if not str(output):
        raise ValueError("output_dir must not be empty")

    resolved = (config or QuantizationConfig()).resolve()
    root = find_repo_root(repo_root)
    prefix = _python_prefix(
        backend_env=resolved.backend_env,
        python_executable=python_executable,
        use_conda=use_conda,
    )
    result_json = output / "backend_results.json"

    if resolved.backend == "gptqmodel":
        command = [
            *prefix,
            str(root / "scripts" / "backend_gptq.py"),
            "--quant-backend",
            "gptqmodel",
            "--quant-codebook",
            resolved.codebook,
            "--quant-preprocess",
            resolved.preprocess,
            "--quant-rotation",
            resolved.rotation,
            "--quant-solver",
            resolved.solver,
            "--quant-polish-mode",
            resolved.polish_mode,
            "--model-id",
            str(model),
            "--bits",
            str(resolved.bits),
            "--group-size",
            str(resolved.group_size),
            "--sym" if resolved.symmetric else "--no-sym",
            "--seed",
            str(resolved.seed),
            "--calibration-dataset",
            resolved.calibration_dataset,
            "--calibration-limit",
            str(resolved.calibration_samples),
            "--calibration-seqlen",
            str(resolved.calibration_seqlen),
            "--mpq-init",
            resolved.init,
            "--mpq-iters",
            str(resolved.iters),
            "--mpq-damping",
            str(resolved.damping),
            "--mpq-prior-tau",
            str(resolved.prior_tau),
            "--mpq-polish-sweeps",
            str(resolved.polish_sweeps),
            "--mpq-vamp-backend",
            "auto",
            "--mpq-block-cd-backend",
            "auto",
            "--output-dir",
            str(output),
            "--json-out",
            str(result_json),
        ]
        if not resolved.evaluate:
            command.append("--skip-eval")
        if not resolved.save:
            command.append("--skip-save")
    else:
        command = [
            *prefix,
            str(root / "scripts" / "backend_quip.py"),
            "--quant-backend",
            "quip",
            "--quant-codebook",
            resolved.codebook,
            "--quant-preprocess",
            resolved.preprocess,
            "--quant-rotation",
            resolved.rotation,
            "--quant-solver",
            resolved.solver,
            "--quant-polish-mode",
            resolved.polish_mode,
            "--base-model",
            str(model),
            "--seed",
            str(resolved.seed),
            "--calibration-dataset",
            resolved.calibration_dataset,
            "--devset-size",
            str(resolved.calibration_samples),
            "--ctx-size",
            str(resolved.calibration_seqlen),
            "--batch-size",
            "4",
            "--quip-tune-iters",
            "0",
            "--mpq-init",
            resolved.init,
            "--mpq-iters",
            str(resolved.iters),
            "--mpq-damping",
            str(resolved.damping),
            "--mpq-prior-tau",
            str(resolved.prior_tau),
            "--mpq-prior-tau-scale-mode",
            "absmax",
            "--mpq-polish-sweeps",
            str(resolved.polish_sweeps),
            "--mpq-block-rows",
            "128",
            "--mpq-blocks-per-chunk",
            "16",
            "--mpq-em-obs-precision",
            "--mpq-compile-prior",
            "--output-dir",
            str(output),
            "--json-out",
            str(result_json),
        ]
        if not resolved.evaluate:
            command.append("--skip-eval")
        if not resolved.save:
            command.append("--skip-hfize")

    command.extend(str(value) for value in backend_args)
    return command


def format_command(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def quantize(
    model: str,
    output_dir: str | os.PathLike[str],
    config: QuantizationConfig | None = None,
    *,
    backend_args: Sequence[str] = (),
    repo_root: str | os.PathLike[str] | None = None,
    python_executable: str | None = None,
    use_conda: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Quantize a model and return the normalized backend result payload."""

    command = build_quantize_command(
        model,
        output_dir,
        config,
        backend_args=backend_args,
        repo_root=repo_root,
        python_executable=python_executable,
        use_conda=use_conda,
    )
    run_env = None if env is None else {**os.environ, **dict(env)}
    subprocess.run(command, check=True, env=run_env)
    result_path = Path(output_dir).expanduser() / "backend_results.json"
    if not result_path.is_file():
        raise RuntimeError(f"Quantization completed without writing {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {result_path}")
    return payload


__all__ = [
    "QuantizationConfig",
    "ResolvedQuantizationConfig",
    "SUPPORTED_BITS",
    "build_quantize_command",
    "find_repo_root",
    "format_command",
    "quantize",
]
