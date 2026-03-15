from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List

import torch


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def _calibration_cache_path(
    *,
    output_dir: str,
    tokenizer,
    dataset_name: str,
    split: str,
    limit: int,
    seqlen: int,
    seed: int,
) -> str:
    cache_dir = os.path.join(output_dir, "_calibration_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(
        f"{tokenizer_id}|{dataset_name}|{split}|{limit}|{seqlen}|{seed}|token_blocks_v1".encode("utf-8")
    ).hexdigest()[:16]
    tokenizer_label = _sanitize_name(Path(tokenizer_id).name or "tokenizer")
    return os.path.join(cache_dir, f"{dataset_name}_{split}_{tokenizer_label}_{limit}_{seqlen}_{seed}_{digest}.pt")


def _iter_text_rows(dataset_name: str, split: str, seed: int) -> Iterator[str]:
    if dataset_name == "c4":
        # Prefer an offline-friendly path by reading directly from cached HF Hub snapshots.
        # `datasets` streaming mode requires network access for remote shard URLs.
        hub_cache = Path(os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache" / "huggingface" / "hub"))
        repo = hub_cache / "datasets--allenai--c4"
        snapshots = repo / "snapshots"
        snapshot: Path | None = None
        if snapshots.exists():
            ref = repo / "refs" / "main"
            if ref.exists():
                snap_id = ref.read_text(encoding="utf-8", errors="replace").strip()
                cand = snapshots / snap_id
                if cand.exists():
                    snapshot = cand
            if snapshot is None:
                snap_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
                if snap_dirs:
                    snap_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    snapshot = snap_dirs[0]

        if snapshot is not None:
            lang_dir = snapshot / "en"
            split_key = str(split).strip().lower()
            if split_key in {"valid"}:
                split_key = "validation"
            patterns = [
                f"c4-{split_key}.*.json.gz",
                f"c4-{split_key}.json.gz",
            ]
            files: List[Path] = []
            for pat in patterns:
                files = sorted(lang_dir.glob(pat))
                if files:
                    break
            if not files:
                raise RuntimeError(
                    f"C4 cache snapshot found but no shard files matched split={split!r} under: {lang_dir}"
                )

            rng = random.Random(int(seed))
            rng.shuffle(files)
            for shard in files:
                with gzip.open(shard, "rt", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = row.get("text")
                        if isinstance(text, str) and text.strip():
                            yield text
            return

        # Fallback: online streaming path (may fail under offline mode).
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        buffer_size = max(8192, min(65536, 64 * 128))
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        for row in ds:
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield text
        return

    if dataset_name in ("wikitext2", "wikitext-2-raw-v1"):
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split).shuffle(seed=seed)
        for row in ds:
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield text
        return

    raise ValueError(f"Unsupported dataset_name={dataset_name!r}.")


def load_or_build_calibration_blocks(
    *,
    tokenizer,
    dataset_name: str,
    split: str,
    limit: int,
    seqlen: int,
    seed: int,
    output_dir: str,
    log_fn: Callable[[str], None] = print,
) -> List[Dict[str, torch.Tensor]]:
    """Build GPTQModel-style calibration blocks (token blocks)."""
    cache_path = _calibration_cache_path(
        output_dir=output_dir,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        limit=limit,
        seqlen=seqlen,
        seed=seed,
    )
    if os.path.exists(cache_path):
        log_fn(f"calibration cache hit: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    rng = random.Random(int(seed))
    blocks: List[Dict[str, torch.Tensor]] = []
    rows_seen = 0
    for text in _iter_text_rows(str(dataset_name), str(split), int(seed)):
        rows_seen += 1
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        if len(token_ids) < seqlen:
            continue

        start = 0 if len(token_ids) == seqlen else rng.randint(0, len(token_ids) - seqlen)
        sample = token_ids[start : start + seqlen]
        input_ids = torch.tensor(sample, dtype=torch.long).unsqueeze(0)
        blocks.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }
        )
        if len(blocks) >= limit:
            break

    if len(blocks) < limit:
        raise RuntimeError(
            f"Unable to build {limit} calibration blocks of length {seqlen} from `{dataset_name}` `{split}`. "
            f"Only collected {len(blocks)} after scanning {rows_seen} rows."
        )

    torch.save(blocks, cache_path)
    log_fn(f"calibration cache write: {cache_path}")
    return blocks


__all__ = ["load_or_build_calibration_blocks"]
