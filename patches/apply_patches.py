from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SUBMODULES = {
    "GPTQModel": {
        "path": pathlib.Path("third_party") / "GPTQModel",
        "patch": REPO_ROOT / "patches" / "gptqmodel.patch",
        "url": "https://github.com/ModelCloud/GPTQModel.git",
        "sha": "ed96f2e04ec868b04f969733a242ce3b5bb7cb1e",
    },
    "fast-hadamard-transform": {
        "path": pathlib.Path("third_party") / "fast-hadamard-transform",
        "patch": REPO_ROOT / "patches" / "fast_hadamard_transform.patch",
        "url": "https://github.com/Dao-AILab/fast-hadamard-transform.git",
        "sha": "e7706faf8d1c3b9f241e36860640ad1dac644ede",
    },
    "quip-sharp": {
        "path": pathlib.Path("third_party") / "quip-sharp",
        "patch": REPO_ROOT / "patches" / "quip_sharp.patch",
        "url": "https://github.com/Cornell-RelaxML/quip-sharp.git",
        "sha": "1d8f873e9a2a8b86b12bb1064c312c5689b77d98",
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
        raise RuntimeError(f"{name}: checkout has local modifications; please clean it first.\n{proc.stdout}")


def _rev_parse_head(path: pathlib.Path, *, name: str) -> str:
    proc = _run(["git", "rev-parse", "HEAD"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: failed to get HEAD sha:\n{proc.stdout}")
    return proc.stdout.strip()


def _bootstrap_checkout(*, name: str, path: pathlib.Path, url: str, expected_sha: str) -> None:
    if path.exists():
        probe = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path)
        if probe.returncode == 0:
            return
        if any(path.iterdir()):
            raise RuntimeError(
                f"{name}: {path} exists but is not a git checkout; remove it and retry"
            )
        path.rmdir()

    path.parent.mkdir(parents=True, exist_ok=True)
    clone = _run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)],
        cwd=REPO_ROOT,
    )
    if clone.returncode != 0:
        raise RuntimeError(f"{name}: failed to clone {url}:\n{clone.stdout}")
    checkout = _run(["git", "checkout", "--detach", expected_sha], cwd=path)
    if checkout.returncode != 0:
        raise RuntimeError(
            f"{name}: failed to checkout pinned revision {expected_sha}:\n{checkout.stdout}"
        )
    print(f"{name}: cloned pinned revision {expected_sha}")


def _git_apply(path: pathlib.Path, patch: pathlib.Path, *, reverse: bool = False, check: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "apply"]
    if reverse:
        cmd.append("--reverse")
    if check:
        cmd.append("--check")
    cmd.append(str(patch))
    return _run(cmd, cwd=path)


def _verify_patch_against_pinned_revision(*, name: str, repo_path: pathlib.Path, expected_sha: str, patch: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"patchcheck_{name.replace('/', '_')}_") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        add_proc = _run(["git", "worktree", "add", "--detach", "--force", str(tmpdir_path), expected_sha], cwd=repo_path)
        if add_proc.returncode != 0:
            raise RuntimeError(f"{name}: failed to create temporary worktree for patch verification:\n{add_proc.stdout}")
        try:
            check_proc = _git_apply(tmpdir_path, patch, check=True)
            if check_proc.returncode != 0:
                raise RuntimeError(
                    f"{name}: patch does not apply cleanly to pinned revision {expected_sha}:\n{check_proc.stdout}"
                )
        finally:
            remove_proc = _run(["git", "worktree", "remove", "--force", str(tmpdir_path)], cwd=repo_path)
            if remove_proc.returncode != 0:
                raise RuntimeError(f"{name}: failed to remove temporary worktree:\n{remove_proc.stdout}")


def _check_patch_state(*, name: str, path: pathlib.Path, expected_sha: str, patch: pathlib.Path, allow_sha_mismatch: bool) -> None:
    if not patch.exists():
        raise FileNotFoundError(f"{name}: patch file not found: {patch}")
    if not path.exists():
        raise FileNotFoundError(f"{name}: checkout path not found: {path}")

    head = _rev_parse_head(path, name=name)
    if head != expected_sha and not allow_sha_mismatch:
        raise RuntimeError(
            f"{name}: checkout SHA mismatch.\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {head}\n"
            f"Refuse to apply patch to an unknown revision.\n"
            f"Restore the pinned checkout or run with --allow-sha-mismatch."
        )

    reverse_check = _git_apply(path, patch, reverse=True, check=True)
    already_applied = reverse_check.returncode == 0
    _verify_patch_against_pinned_revision(name=name, repo_path=path, expected_sha=expected_sha, patch=patch)
    if not already_applied:
        _require_clean_worktree(path, name=name)
    state = "patch already applied" if already_applied else "patch ready to apply"
    print(f"{name}: {state}")


def apply_patch(*, name: str, path: pathlib.Path, expected_sha: str, patch: pathlib.Path, allow_sha_mismatch: bool) -> None:
    _check_patch_state(
        name=name,
        path=path,
        expected_sha=expected_sha,
        patch=patch,
        allow_sha_mismatch=allow_sha_mismatch,
    )
    reverse_check = _git_apply(path, patch, reverse=True, check=True)
    if reverse_check.returncode == 0:
        return

    apply_proc = _git_apply(path, patch, check=False)
    if apply_proc.returncode != 0:
        raise RuntimeError(f"{name}: failed to apply patch:\n{apply_proc.stdout}")

    print(f"{name}: patch applied")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-sha-mismatch", action="store_true", help="Apply patches even if a checkout SHA differs.")
    parser.add_argument("--check", action="store_true", help="Validate pinned checkouts and patch health without applying.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Clone missing third-party checkouts before validating and applying patches.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--gptq-only",
        action="store_true",
        help="Prepare GPTQModel and fast-hadamard-transform only.",
    )
    selection.add_argument(
        "--quip-only",
        action="store_true",
        help="Prepare QUIP# and fast-hadamard-transform only.",
    )
    args = parser.parse_args()

    selected = tuple(SUBMODULES)
    if args.gptq_only:
        selected = ("GPTQModel", "fast-hadamard-transform")
    elif args.quip_only:
        selected = ("fast-hadamard-transform", "quip-sharp")

    for name in selected:
        meta = SUBMODULES[name]
        path = REPO_ROOT / meta["path"]
        expected_sha = str(meta["sha"])
        if args.bootstrap:
            _bootstrap_checkout(
                name=name,
                path=path,
                url=str(meta["url"]),
                expected_sha=expected_sha,
            )
        if args.check:
            _check_patch_state(
                name=name,
                path=path,
                expected_sha=expected_sha,
                patch=meta["patch"],
                allow_sha_mismatch=bool(args.allow_sha_mismatch),
            )
        else:
            apply_patch(
                name=name,
                path=path,
                expected_sha=expected_sha,
                patch=meta["patch"],
                allow_sha_mismatch=bool(args.allow_sha_mismatch),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
