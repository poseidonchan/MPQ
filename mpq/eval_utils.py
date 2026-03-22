from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset


def eval_protocol_for_dataset(dataset: str, *, seqlen: int, requested_max_eval_tokens: int) -> dict:
    dataset_norm = str(dataset).strip().lower()
    requested = int(requested_max_eval_tokens)
    if requested > 0:
        rounded = (requested // int(seqlen)) * int(seqlen)
        if rounded <= 0:
            raise ValueError("max_eval_tokens must be >= seqlen when set")
        return {
            "dataset": dataset_norm,
            "mode": "token_cap",
            "requested_max_eval_tokens": requested,
            "effective_max_eval_tokens": rounded,
        }

    if dataset_norm == "wikitext2":
        return {
            "dataset": dataset_norm,
            "mode": "full_test_split",
            "requested_max_eval_tokens": 0,
            "effective_max_eval_tokens": 0,
        }
    if dataset_norm == "c4":
        return {
            "dataset": dataset_norm,
            "mode": "gptq_c4_validation_256xseqlen",
            "requested_max_eval_tokens": 0,
            "effective_max_eval_tokens": 256 * int(seqlen),
        }
    raise ValueError(f"unsupported eval dataset: {dataset_norm!r}")


@torch.inference_mode()
def eval_cache_path(output_dir: str, tokenizer, dataset: str, max_eval_tokens: int, seqlen: int, seed: int) -> str:
    cache_dir = os.path.join(output_dir, "_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(
        f"{tokenizer_id}|{dataset}|{max_eval_tokens}|{seqlen}|{seed}|eval_v3".encode("utf-8")
    ).hexdigest()[:16]
    tokenizer_label = Path(tokenizer_id).name.replace("/", "_") or "tokenizer"
    dataset_label = str(dataset).replace("/", "_")
    token_label = "full" if int(max_eval_tokens) <= 0 else str(int(max_eval_tokens))
    return os.path.join(cache_dir, f"{dataset_label}_{tokenizer_label}_{token_label}_{digest}.pt")


@torch.inference_mode()
def load_or_build_wikitext2_tokens(tokenizer, *, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    cache_path = eval_cache_path(output_dir, tokenizer, "wikitext2", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        if int(max_eval_tokens) > 0:
            protocol = eval_protocol_for_dataset("wikitext2", seqlen=int(seqlen), requested_max_eval_tokens=int(max_eval_tokens))
            return tokens[:, : int(protocol["effective_max_eval_tokens"])].contiguous()
        return tokens.contiguous()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids.cpu().contiguous()
    if int(max_eval_tokens) > 0:
        protocol = eval_protocol_for_dataset("wikitext2", seqlen=int(seqlen), requested_max_eval_tokens=int(max_eval_tokens))
        tokens = tokens[:, : int(protocol["effective_max_eval_tokens"])].contiguous()
    torch.save(tokens, cache_path)
    return tokens


@torch.inference_mode()
def load_or_build_c4_tokens(tokenizer, *, repo_root: str, model_ref: str, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    cache_path = eval_cache_path(output_dir, tokenizer, "c4", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        if int(max_eval_tokens) > 0:
            protocol = eval_protocol_for_dataset("c4", seqlen=int(seqlen), requested_max_eval_tokens=int(max_eval_tokens))
            return tokens[:, : int(protocol["effective_max_eval_tokens"])].contiguous()
        return tokens.contiguous()

    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    # Match the common GPTQ/QUIP repo convention: build a fixed validation token
    # stream from 256 random contiguous windows, independent of calibration seed.
    rng = random.Random(0)
    valenc = []
    target_chunks = 256
    while len(valenc) < target_chunks:
        sample_idx = rng.randint(0, len(valdata) - 1)
        text = str(valdata[int(sample_idx)]["text"])
        tmp = tokenizer(text, return_tensors="pt", truncation=False).input_ids
        if int(tmp.shape[1]) < int(seqlen):
            continue
        max_start = int(tmp.shape[1]) - int(seqlen)
        start = 0 if max_start <= 0 else rng.randint(0, max_start)
        valenc.append(tmp[:, start : start + int(seqlen)])

    tokens = torch.hstack(valenc).cpu().contiguous()
    if int(max_eval_tokens) > 0:
        protocol = eval_protocol_for_dataset("c4", seqlen=int(seqlen), requested_max_eval_tokens=int(max_eval_tokens))
        tokens = tokens[:, : int(protocol["effective_max_eval_tokens"])].contiguous()
    torch.save(tokens, cache_path)
    return tokens


@torch.no_grad()
def eval_ppl_from_tokens(
    model,
    tokens: torch.Tensor,
    *,
    seqlen: int = 2048,
) -> float:
    runtime_device = next(model.parameters()).device
    tokens = tokens.to(runtime_device)

    nlls = []
    nsamples = tokens.shape[1] // int(seqlen)
    for i in range(nsamples):
        batch = tokens[:, i * seqlen : (i + 1) * seqlen]
        outputs = model(batch, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * (int(seqlen) - 1))

    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * (int(seqlen) - 1)
    return float(torch.exp(total_nll / total_tokens).item())
