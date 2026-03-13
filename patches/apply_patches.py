from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SUBMODULES = {
    "GPTQModel": {
        "path": REPO_ROOT / "third_party" / "GPTQModel",
        "expected_sha": "ed96f2e04ec868b04f969733a242ce3b5bb7cb1e",
        "patch": REPO_ROOT / "patches" / "gptqmodel.patch",
    },
    "fast-hadamard-transform": {
        "path": REPO_ROOT / "third_party" / "fast-hadamard-transform",
        "expected_sha": "e7706faf8d1c3b9f241e36860640ad1dac644ede",
        "patch": REPO_ROOT / "patches" / "fast_hadamard_transform.patch",
    },
    "quip-sharp": {
        "path": REPO_ROOT / "third_party" / "quip-sharp",
        "expected_sha": "1d8f873e9a2a8b86b12bb1064c312c5689b77d98",
        "patch": REPO_ROOT / "patches" / "quip_sharp.patch",
    },
}


def _run(cmd: Sequence[str], *, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _require_clean_worktree(path: pathlib.Path, *, name: str) -> None:
    proc = _run(["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: failed to run git status:\n{proc.stdout}")
    if proc.stdout.strip():
        raise RuntimeError(f"{name}: submodule has local modifications; please clean it first.\n{proc.stdout}")


def _rev_parse_head(path: pathlib.Path, *, name: str) -> str:
    proc = _run(["git", "rev-parse", "HEAD"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: failed to get HEAD sha:\n{proc.stdout}")
    return proc.stdout.strip()


def _git_apply(path: pathlib.Path, patch: pathlib.Path, *, reverse: bool = False, check: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "apply"]
    if reverse:
        cmd.append("--reverse")
    if check:
        cmd.append("--check")
    cmd.append(str(patch))
    return _run(cmd, cwd=path)


def apply_patch(*, name: str, path: pathlib.Path, expected_sha: str, patch: pathlib.Path, allow_sha_mismatch: bool) -> None:
    if not patch.exists():
        raise FileNotFoundError(f"{name}: patch file not found: {patch}")
    if not path.exists():
        raise FileNotFoundError(f"{name}: submodule path not found: {path}")

    head = _rev_parse_head(path, name=name)
    if head != expected_sha and not allow_sha_mismatch:
        raise RuntimeError(
            f"{name}: submodule SHA mismatch.\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {head}\n"
            f"Refuse to apply patch to an unknown revision.\n"
            f"Re-checkout submodules or run with --allow-sha-mismatch."
        )

    # Idempotent apply:
    # - If already applied, `git apply --reverse --check` succeeds even if the worktree is dirty.
    reverse_check = _git_apply(path, patch, reverse=True, check=True)
    if reverse_check.returncode == 0:
        print(f"{name}: patch already applied")
        return

    # Patch not detected; require a clean worktree before applying to avoid clobbering local edits.
    _require_clean_worktree(path, name=name)

    check_proc = _git_apply(path, patch, check=True)
    if check_proc.returncode != 0:
        raise RuntimeError(f"{name}: patch does not apply cleanly:\n{check_proc.stdout}")

    apply_proc = _git_apply(path, patch, check=False)
    if apply_proc.returncode != 0:
        raise RuntimeError(f"{name}: failed to apply patch:\n{apply_proc.stdout}")

    print(f"{name}: patch applied")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-sha-mismatch", action="store_true", help="Apply patches even if submodule SHA differs.")
    args = parser.parse_args()

    for name, meta in SUBMODULES.items():
        apply_patch(
            name=name,
            path=meta["path"],
            expected_sha=meta["expected_sha"],
            patch=meta["patch"],
            allow_sha_mismatch=bool(args.allow_sha_mismatch),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
