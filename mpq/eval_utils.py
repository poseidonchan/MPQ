from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from datasets import load_dataset


@torch.inference_mode()
def eval_cache_path(output_dir: str, tokenizer, dataset: str, max_eval_tokens: int, seqlen: int, seed: int) -> str:
    cache_dir = os.path.join(output_dir, "_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(
        f"{tokenizer_id}|{dataset}|{max_eval_tokens}|{seqlen}|{seed}|eval_v2".encode("utf-8")
    ).hexdigest()[:16]
    tokenizer_label = Path(tokenizer_id).name.replace("/", "_") or "tokenizer"
    dataset_label = str(dataset).replace("/", "_")
    return os.path.join(cache_dir, f"{dataset_label}_{tokenizer_label}_{max_eval_tokens}_{digest}.pt")


@torch.inference_mode()
def load_or_build_wikitext2_tokens(tokenizer, *, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    cache_path = eval_cache_path(output_dir, tokenizer, "wikitext2", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[:, :max_eval_tokens].cpu().contiguous()
    torch.save(tokens, cache_path)
    return tokens


@torch.inference_mode()
def load_or_build_c4_tokens(tokenizer, *, repo_root: str, model_ref: str, max_eval_tokens: int, seqlen: int, seed: int, output_dir: str) -> torch.Tensor:
    cache_path = eval_cache_path(output_dir, tokenizer, "c4", max_eval_tokens, seqlen, seed)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gptq_data_utils",
        Path(repo_root) / "third_party" / "quip-sharp" / "lib" / "utils" / "gptq_data_utils.py",
    )
    gptq_data_utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gptq_data_utils)

    tokens = gptq_data_utils.get_test_tokens("c4", seed=int(seed), seqlen=int(seqlen), model=str(model_ref))
    tokens = tokens[:, :max_eval_tokens].cpu().contiguous()
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

