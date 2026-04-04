from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))

import mpq.rotation as mpq_rotation


def _maybe_reexec_with_conda_ld_library_path() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = str(Path(conda_prefix) / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [p for p in existing.split(":") if p]
    if lib_dir in paths:
        return
    if os.environ.get("MPQ_REEXEC_LD_LIBRARY_PATH") == "1":
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    env["MPQ_REEXEC_LD_LIBRARY_PATH"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_ld_library_path()

quip_had = importlib.import_module("lib.utils.matmul_had")


WIDTHS = (3072, 4096, 8192, 9728, 12288, 14336, 16384, 19456, 24576, 28672)


def _quip_plan(width: int) -> tuple[str, int]:
    _hadK, K = quip_had.get_hadK(int(width))
    if K == 0:
        return "none", 1
    if K < 0:
        return "block", int(-K)
    return "exact", int(K)


def _mpq_plan(width: int) -> tuple[str, int]:
    _hadK, K = mpq_rotation.get_hadK(int(width))
    return "exact", int(K)


def _assert_exact_support() -> None:
    print("Width support:")
    for width in WIDTHS:
        mpq_kind, mpq_value = _mpq_plan(width)
        quip_kind, quip_value = _quip_plan(width)
        print(f"  width={width:<5d} mpq={mpq_kind}:{mpq_value} quip={quip_kind}:{quip_value}")
        if mpq_kind != "exact" or quip_kind != "exact" or mpq_value != quip_value:
            raise SystemExit(f"support mismatch at width={width}: mpq={mpq_kind}:{mpq_value}, quip={quip_kind}:{quip_value}")

    for width in (9728, 19456):
        _hadK, mpq_k = mpq_rotation.get_hadK(width)
        _hadK_quip, quip_k = quip_had.get_hadK(width)
        if int(mpq_k) != 76 or int(quip_k) != 76:
            raise SystemExit(f"expected K=76 for width={width}, got mpq={mpq_k}, quip={quip_k}")


def _assert_had76_orthogonal() -> None:
    had76 = mpq_rotation.get_had76().to(dtype=torch.float32)
    gram = had76 @ had76.T
    expected = 76.0 * torch.eye(76, dtype=torch.float32)
    if not torch.equal(gram, expected):
        diff = float((gram - expected).abs().max().item())
        raise SystemExit(f"had76 orthogonality check failed; max_abs_diff={diff}")
    print("had76 orthogonality: OK")


def _smoke_transform(width: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, int(width), dtype=torch.float32)
    for name, had_u, had_ut in (
        ("mpq", mpq_rotation.matmul_hadU, mpq_rotation.matmul_hadUt),
        ("quip", quip_had.matmul_hadU, quip_had.matmul_hadUt),
    ):
        y = had_u(x)
        z = had_ut(y)
        if y.shape != x.shape or z.shape != x.shape:
            raise SystemExit(f"{name} shape mismatch at width={width}: y={tuple(y.shape)} z={tuple(z.shape)}")
        if not torch.isfinite(y).all() or not torch.isfinite(z).all():
            raise SystemExit(f"{name} produced non-finite values at width={width}")
        if not torch.allclose(z, x, atol=1e-4, rtol=1e-4):
            err = float((z - x).abs().max().item())
            raise SystemExit(f"{name} inverse check failed at width={width}; max_abs_err={err}")
        print(f"{name} smoke width={width}: OK")


def main() -> int:
    _assert_exact_support()
    _assert_had76_orthogonal()
    for width in (9728, 19456):
        _smoke_transform(width)
    print("Hadamard support checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
