"""Standalone GPTQModel-aligned AWQ preprocessing for QUIP#.

This module ports the AWQ scale-search path used by GPTQModel into the local
QUIP# pipeline so the preprocess can run in the QUIP environment without
depending on GPTQModel being installed there.
"""

from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from mpq.dense_decoder_adapter import DenseDecoderAdapter

log = logging.getLogger(__name__)


@dataclass
class _AWQGroupDebug:
    name: str
    prev_op: str
    layers: list[str]
    best_ratio: float
    best_error: float
    max_abs_scale: float
    min_abs_scale: float


@dataclass
class _CapturedBatch:
    next_hidden: torch.Tensor
    qkv_x: Optional[torch.Tensor] = None
    qkv_out: Optional[torch.Tensor] = None
    o_x: Optional[torch.Tensor] = None
    o_out: Optional[torch.Tensor] = None
    gate_x: Optional[torch.Tensor] = None
    mlp_out: Optional[torch.Tensor] = None
    down_x: Optional[torch.Tensor] = None
    down_out: Optional[torch.Tensor] = None


@torch.no_grad()
def _compute_w_mean(linears: Sequence[nn.Linear], group_size: int) -> torch.Tensor:
    """Match GPTQModel AWQProcessor._search_best_scale step 1."""

    first_weight = linears[0].weight
    weight_dtype = first_weight.dtype
    weight_device = first_weight.device
    num_channels = first_weight.shape[1]
    w_sum = torch.zeros(num_channels, dtype=torch.float32, device=weight_device)
    row_count = 0

    for layer in linears:
        weight = layer.weight
        if weight.shape[1] != num_channels:
            raise ValueError(
                f"Expected consistent in_features across layers ({num_channels}), got {weight.shape[1]}."
            )
        org_shape = weight.shape
        weight_abs = weight.abs()
        if group_size > 0:
            if org_shape[-1] % group_size != 0:
                raise ValueError(
                    f"weight shape {tuple(org_shape)} is not divisible by group_size={group_size}"
                )
            weight_group = weight_abs.view(-1, group_size)
        else:
            weight_group = weight_abs.reshape(org_shape[0], -1)
        group_scale = weight_group.amax(dim=1, keepdim=True) + 1e-6
        normalized = (weight_group / group_scale).view(org_shape)
        w_sum += normalized.sum(dim=0, dtype=torch.float32)
        row_count += org_shape[0]

    if row_count == 0:
        return torch.zeros(num_channels, dtype=weight_dtype, device=weight_device)
    return (w_sum / row_count).to(weight_dtype)


def _sanitize_kwargs(inputs_kwargs: Dict[str, Any], module: nn.Module) -> Dict[str, Any]:
    module_signature = inspect.signature(module.forward).parameters
    sanitized_kwargs = {}
    for key, value in inputs_kwargs.items():
        if key in module_signature:
            sanitized_kwargs[key] = value
    return sanitized_kwargs


def _move_kwargs_to_device(inputs_kwargs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs_kwargs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, tuple):
            moved[key] = tuple(
                item.to(device) if isinstance(item, torch.Tensor) else item for item in value
            )
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def _pseudo_quantize_tensor_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
) -> None:
    """Port of GPTQModel AWQProcessor._pseudo_quantize_tensor_into."""

    org_shape = src.shape
    if group_size > 0:
        if org_shape[-1] % group_size != 0:
            raise ValueError(
                f"org_shape[-1]={org_shape[-1]} must be divisible by group_size={group_size}"
            )
        src_view = src.view(-1, group_size)
        dst_view = dst.view(-1, group_size)
    else:
        src_view = src.reshape(org_shape[0], -1)
        dst_view = dst.reshape_as(src_view)

    if not sym:
        max_val = src_view.amax(dim=1, keepdim=True)
        min_val = src_view.amin(dim=1, keepdim=True)
        max_int = 2**bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        dst_view.copy_(src_view)
        dst_view.div_(scales)
        torch.round(dst_view, out=dst_view)
        dst_view.add_(zeros)
        dst_view.clamp_(min_int, max_int)
        dst_view.sub_(zeros)
        dst_view.mul_(scales)
    else:
        max_val = src_view.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (bits - 1) - 1
        min_int = -(2 ** (bits - 1))
        scales = max_val / max_int

        dst_view.copy_(src_view)
        dst_view.div_(scales)
        torch.round(dst_view, out=dst_view)
        dst_view.clamp_(min_int, max_int)
        dst_view.mul_(scales)


@torch.no_grad()
def _scale_ln_fcs(ln: nn.Module, fcs: Sequence[nn.Linear], scales: torch.Tensor) -> None:
    scales = scales.to(ln.weight.device, dtype=ln.weight.dtype)
    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device, dtype=fc.weight.dtype))


@torch.no_grad()
def _scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor) -> None:
    scales = scales.to(fc1.weight.device, dtype=fc1.weight.dtype)
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    fc2.weight.mul_(scales.view(1, -1).to(fc2.weight.device, dtype=fc2.weight.dtype))


def _to_cpu_detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_cpu_detached(value)
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _extract_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def _capture_layer_batch(
    layer: nn.Module,
    handles: Any,
    hidden_states: torch.Tensor,
    layer_kwargs: Dict[str, Any],
) -> _CapturedBatch:
    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def _capture_input(name: str):
        def hook_fn(module, inp, out):
            del module, out
            x = inp[0] if isinstance(inp, tuple) else inp
            captured[name] = _to_cpu_detached(x)

        return hook_fn

    def _capture_output(name: str):
        def hook_fn(module, inp, out):
            del module, inp
            captured[name] = _to_cpu_detached(_extract_tensor(out))

        return hook_fn

    hooks.append(handles.q_proj.register_forward_hook(_capture_input("qkv_x")))
    hooks.append(layer.self_attn.register_forward_hook(_capture_output("qkv_out")))
    hooks.append(handles.o_proj.register_forward_hook(_capture_input("o_x")))
    hooks.append(handles.o_proj.register_forward_hook(_capture_output("o_out")))
    hooks.append(handles.gate_proj.register_forward_hook(_capture_input("gate_x")))
    hooks.append(layer.mlp.register_forward_hook(_capture_output("mlp_out")))
    hooks.append(handles.down_proj.register_forward_hook(_capture_input("down_x")))
    hooks.append(handles.down_proj.register_forward_hook(_capture_output("down_out")))

    try:
        out = layer(hidden_states, **layer_kwargs)
        next_hidden = _to_cpu_detached(_extract_tensor(out))
    finally:
        for hook in hooks:
            hook.remove()

    return _CapturedBatch(next_hidden=next_hidden, **captured)


def _batch_calibration_blocks(
    calibration_blocks: Sequence[Dict[str, torch.Tensor]],
    batch_size: int,
) -> list[Dict[str, torch.Tensor]]:
    batched = []
    for i in range(0, len(calibration_blocks), batch_size):
        chunk = calibration_blocks[i : i + batch_size]
        input_ids = torch.cat([row["input_ids"] for row in chunk], dim=0)
        attention_mask = None
        if all("attention_mask" in row for row in chunk):
            attention_mask = torch.cat([row["attention_mask"] for row in chunk], dim=0)
        batched.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batched


def _capture_first_layer_inputs(
    model: nn.Module,
    first_layer: nn.Module,
    calibration_batches: Sequence[Dict[str, torch.Tensor]],
) -> tuple[list[torch.Tensor], list[Dict[str, Any]]]:
    class _StopCapture(RuntimeError):
        pass

    hidden_batches: list[torch.Tensor] = []
    kwargs_batches: list[Dict[str, Any]] = []

    for batch in calibration_batches:
        captured: dict[str, Any] = {}

        def _pre_hook(module, args, kwargs):
            del module
            captured["hidden_states"] = _to_cpu_detached(args[0])
            captured["kwargs"] = {
                key: _clone_to_cpu(value)
                for key, value in kwargs.items()
                if key != "past_key_values"
            }
            raise _StopCapture()

        handle = first_layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        try:
            try:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    use_cache=False,
                )
            except _StopCapture:
                pass
        finally:
            handle.remove()

        hidden_batches.append(captured["hidden_states"])
        kwargs_batches.append(captured["kwargs"])

    return hidden_batches, kwargs_batches


@torch.no_grad()
def _build_layer_kwargs(
    model: nn.Module,
    adapter: DenseDecoderAdapter,
    hidden_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
) -> Dict[str, Any]:
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0).expand(batch_size, -1)

    layer_kwargs: Dict[str, Any] = {"position_ids": position_ids}
    position_embeddings = adapter.compute_position_embeddings(model, hidden_states, position_ids)
    if position_embeddings is not None:
        layer_kwargs["position_embeddings"] = position_embeddings

    attention_mask = attention_mask_2d
    if attention_mask is not None:
        attention_mask = attention_mask.to(hidden_states.device)
    if attention_mask is None:
        attention_mask = adapter.build_attention_mask(
            model,
            batch_size=batch_size,
            ctx_size=seq_len,
            hidden_states=hidden_states,
        )
    else:
        attention_mask = adapter.build_attention_mask(
            model,
            batch_size=batch_size,
            ctx_size=seq_len,
            hidden_states=hidden_states,
        )
    if attention_mask is not None:
        layer_kwargs["attention_mask"] = attention_mask
    return layer_kwargs


def _candidate_scales(
    x_mean: torch.Tensor,
    w_mean: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
    scales = scales / (scales.max() * scales.min()).sqrt()
    scales[torch.isinf(scales)] = 1
    scales[torch.isnan(scales)] = 1
    return scales


@torch.no_grad()
def _compute_loss_sum(
    fp_output: torch.Tensor,
    q_output: torch.Tensor,
) -> tuple[float, int]:
    diff = (fp_output - q_output).float()
    return float(diff.pow(2).sum().item()), int(diff.numel())


def _merge_kwargs_batches(module_kwargs_batches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not module_kwargs_batches:
        return {}

    merged: Dict[str, Any] = {}
    keys = set().union(*(batch.keys() for batch in module_kwargs_batches))
    for key in keys:
        values = [batch[key] for batch in module_kwargs_batches if key in batch]
        if len(values) != len(module_kwargs_batches):
            merged[key] = values[0]
            continue

        first_value = values[0]
        if isinstance(first_value, torch.Tensor) and all(isinstance(v, torch.Tensor) for v in values):
            if first_value.dim() > 1:
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = first_value
            continue

        if isinstance(first_value, tuple) and all(isinstance(v, tuple) for v in values):
            if not first_value:
                merged[key] = ()
                continue
            tuple_items = []
            for item_idx in range(len(first_value)):
                item_values = [value[item_idx] for value in values]
                if all(isinstance(v, torch.Tensor) for v in item_values):
                    if item_values[0].dim() > 1:
                        tuple_items.append(torch.cat(item_values, dim=0))
                    else:
                        tuple_items.append(item_values[0])
                else:
                    tuple_items.append(item_values[0])
            merged[key] = tuple(tuple_items)
            continue

        merged[key] = first_value

    return merged


def _forward_with_quant_batching(
    model: nn.Module,
    module: nn.Module,
    x: torch.Tensor,
    module_kwargs: Dict[str, Any],
    quant_batch_size: int,
) -> torch.Tensor:
    target_device = None
    try:
        target_device = next(module.parameters()).device
    except (StopIteration, Exception):
        target_device = None

    for key, value in list(module_kwargs.items()):
        if isinstance(value, torch.Tensor):
            if target_device is not None and value.device != target_device:
                module_kwargs[key] = value.to(target_device)
        elif isinstance(value, (list, tuple)):
            converted = []
            changed = False
            for item in value:
                if isinstance(item, torch.Tensor) and target_device is not None and item.device != target_device:
                    converted.append(item.to(target_device))
                    changed = True
                else:
                    converted.append(item)
            if changed:
                module_kwargs[key] = type(value)(converted)

    seq_len = x.shape[1] if x.dim() >= 2 else x.shape[0]
    batch_dim = x.shape[0] if x.dim() >= 2 else 1

    supports_position_ids = False
    supports_position_embeddings = False
    try:
        signature = inspect.signature(module.forward).parameters
        supports_position_ids = "position_ids" in signature
        supports_position_embeddings = "position_embeddings" in signature
    except (ValueError, TypeError):
        pass

    rotary = getattr(getattr(model, "model", model), "rotary_emb", None)
    if seq_len is not None and rotary is not None and supports_position_embeddings:
        pos_ids = module_kwargs.get("position_ids") if supports_position_ids else None
        if not supports_position_ids:
            pos_ids = None
        if pos_ids is None or pos_ids.shape[-1] != seq_len:
            pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
            if x.dim() >= 2:
                pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
            if supports_position_ids:
                module_kwargs["position_ids"] = pos_values
            pos_for_rotary = pos_values
        else:
            pos_for_rotary = pos_ids.to(target_device or pos_ids.device)
            if supports_position_ids:
                module_kwargs["position_ids"] = pos_for_rotary

        x_for_rotary = x if target_device is None else x.to(target_device)
        module_kwargs["position_embeddings"] = rotary(x_for_rotary, pos_for_rotary)
    elif supports_position_ids and seq_len is not None and "position_ids" not in module_kwargs:
        pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
        if x.dim() >= 2:
            pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
        module_kwargs["position_ids"] = pos_values

    if quant_batch_size <= 1:
        return _extract_tensor(module(x, **module_kwargs))

    outputs = []
    for x_partial in torch.split(x, quant_batch_size, dim=0):
        partial_kwargs: Dict[str, Any] = {}
        for key, value in module_kwargs.items():
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == x.shape[0]:
                partial_kwargs[key] = value[: x_partial.shape[0]]
            elif isinstance(value, tuple):
                partial_kwargs[key] = tuple(
                    item[: x_partial.shape[0]]
                    if isinstance(item, torch.Tensor) and item.shape and item.shape[0] == x.shape[0]
                    else item
                    for item in value
                )
            else:
                partial_kwargs[key] = value
        out = module(x_partial, **partial_kwargs)
        outputs.append(_extract_tensor(out))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def _search_best_scale(
    *,
    model: nn.Module,
    linears: Sequence[nn.Linear],
    x_batches: Sequence[torch.Tensor],
    fp_outputs: Sequence[torch.Tensor],
    module2inspect: nn.Module,
    module_kwargs_batches: Sequence[Dict[str, Any]],
    bits: int,
    group_size: int,
    sym: bool,
    quant_batch_size: int,
    prev_scale_hint: Optional[float] = None,
) -> tuple[torch.Tensor, float, float]:
    if len(x_batches) != len(fp_outputs) or len(x_batches) != len(module_kwargs_batches):
        raise ValueError("x_batches, fp_outputs, and module_kwargs_batches must have the same length")

    x = torch.cat(list(x_batches), dim=0)
    merged_kwargs = _merge_kwargs_batches(module_kwargs_batches)

    x_flat = x.abs().view(-1, x.shape[-1]).to(torch.float32)
    x_mean = (x_flat.sum(dim=0) / max(int(x_flat.shape[0]), 1)).to(x.dtype)

    orig_weights_cpu = {
        fc: fc.weight.detach().to(torch.float32).cpu().contiguous() for fc in linears
    }

    best_ratio = -1.0
    best_error = float("inf")
    best_scales: Optional[torch.Tensor] = None
    device = next(module2inspect.parameters()).device
    x_mean = x_mean.view(-1).to(device)
    w_mean = _compute_w_mean(list(linears), group_size=group_size).view(-1).to(device)
    if prev_scale_hint is not None:
        w_mean = w_mean * float(prev_scale_hint)
    x_dev = x.to(device)
    module_kwargs = _move_kwargs_to_device(_sanitize_kwargs(merged_kwargs, module2inspect), device)
    fp_output_dev = _forward_with_quant_batching(
        model,
        module2inspect,
        x_dev,
        copy.deepcopy(module_kwargs),
        quant_batch_size=quant_batch_size,
    )
    fp_output_dev = fp_output_dev.clip(
        torch.finfo(fp_output_dev.dtype).min,
        torch.finfo(fp_output_dev.dtype).max,
    )

    for ratio_idx in range(20):
        ratio = ratio_idx / 20.0
        scales = _candidate_scales(x_mean, w_mean, ratio).to(device)
        scales_view = scales.view(1, -1)

        for fc in linears:
            fc.weight.mul_(scales_view.to(fc.weight.device, dtype=fc.weight.dtype))
            _pseudo_quantize_tensor_into(
                fc.weight,
                fc.weight,
                bits=bits,
                group_size=group_size,
                sym=sym,
            )
            fc.weight.div_(scales_view.to(fc.weight.device, dtype=fc.weight.dtype))

        out = _forward_with_quant_batching(
            model,
            module2inspect,
            x_dev,
            module_kwargs,
            quant_batch_size=quant_batch_size,
        )
        out = out.clip(torch.finfo(out.dtype).min, torch.finfo(out.dtype).max)
        loss_sum, numel_sum = _compute_loss_sum(fp_output_dev.to(out.device), out)

        for fc in linears:
            fc.weight.copy_(orig_weights_cpu[fc].to(device=fc.weight.device, dtype=fc.weight.dtype))

        loss = loss_sum / max(numel_sum, 1)
        if loss < best_error:
            best_error = loss
            best_ratio = ratio
            best_scales = scales.detach().cpu()

    for fc in linears:
        fc.weight.copy_(orig_weights_cpu[fc].to(device=fc.weight.device, dtype=fc.weight.dtype))

    if best_scales is None:
        raise RuntimeError("AWQ scale search failed to find a valid ratio")
    return best_scales, best_error, best_ratio


def _mean_abs_weight(modules: Iterable[nn.Linear]) -> float:
    values = []
    for module in modules:
        values.append(float(module.weight.detach().abs().mean().item()))
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def apply_awq_scaling(
    model: nn.Module,
    adapter: DenseDecoderAdapter,
    calibration_blocks: Sequence[Dict[str, torch.Tensor]],
    *,
    bits: int,
    group_size: int = 128,
    sym: bool = False,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Apply GPTQModel-aligned AWQ scaling to *model* in-place."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    root = adapter.get_model_root(model)
    layers = adapter.get_layers(model)

    calibration_batches = _batch_calibration_blocks(calibration_blocks, batch_size)
    current_hidden_batches, layer_kwargs_templates = _capture_first_layer_inputs(
        model,
        layers[0],
        calibration_batches,
    )

    debug: list[dict[str, Any]] = []

    for layer_idx, layer in enumerate(layers):
        layer = layer.to(device)
        handles = adapter.get_layer_handles(layer)
        log.info("AWQ scaling layer %s/%s", layer_idx, len(layers))

        captured_batches: list[_CapturedBatch] = []
        layer_kwargs_batches: list[Dict[str, Any]] = []
        for hidden_cpu, layer_kwargs_template in zip(current_hidden_batches, layer_kwargs_templates):
            hidden_dev = hidden_cpu.to(device)
            layer_kwargs = _move_kwargs_to_device(copy.deepcopy(layer_kwargs_template), hidden_dev.device)
            layer_kwargs_batches.append(
                {
                    key: (
                        tuple(_to_cpu_detached(item) if isinstance(item, torch.Tensor) else item for item in value)
                        if isinstance(value, tuple)
                        else (_to_cpu_detached(value) if isinstance(value, torch.Tensor) else value)
                    )
                    for key, value in layer_kwargs.items()
                }
            )
            captured_batches.append(_capture_layer_batch(layer, handles, hidden_dev, layer_kwargs))

        qkv_scale_hint = _mean_abs_weight([handles.q_proj, handles.k_proj, handles.v_proj])
        o_proj_scale_hint = _mean_abs_weight([handles.o_proj])
        gate_up_scale_hint = _mean_abs_weight([handles.gate_proj, handles.up_proj])
        layer_debug: list[_AWQGroupDebug] = []
        pending_scales: list[tuple[str, torch.Tensor]] = []

        qkv_x_batches = [batch.qkv_x for batch in captured_batches if batch.qkv_x is not None]
        qkv_out_batches = [batch.qkv_out for batch in captured_batches if batch.qkv_out is not None]
        if qkv_x_batches and qkv_out_batches:
            qkv_scales, qkv_error, qkv_ratio = _search_best_scale(
                model=model,
                linears=[handles.q_proj, handles.k_proj, handles.v_proj],
                x_batches=qkv_x_batches,
                fp_outputs=qkv_out_batches,
                module2inspect=layer.self_attn,
                module_kwargs_batches=layer_kwargs_batches,
                bits=bits,
                group_size=group_size,
                sym=sym,
                quant_batch_size=batch_size,
            )
            pending_scales.append(("qkv", qkv_scales))
            layer_debug.append(
                _AWQGroupDebug(
                    name="qkv",
                    prev_op="input_layernorm",
                    layers=["q_proj", "k_proj", "v_proj"],
                    best_ratio=float(qkv_ratio),
                    best_error=float(qkv_error),
                    max_abs_scale=float(qkv_scales.abs().max().item()),
                    min_abs_scale=float(qkv_scales.abs().min().item()),
                )
            )
        if handles.v_proj.weight.shape == handles.o_proj.weight.shape:
            prev_scale_hint = qkv_scale_hint
            o_x_batches = [batch.o_x for batch in captured_batches if batch.o_x is not None]
            o_out_batches = [batch.o_out for batch in captured_batches if batch.o_out is not None]
            if o_x_batches and o_out_batches:
                o_scales, o_error, o_ratio = _search_best_scale(
                    model=model,
                    linears=[handles.o_proj],
                    x_batches=o_x_batches,
                    fp_outputs=o_out_batches,
                    module2inspect=handles.o_proj,
                    module_kwargs_batches=[{} for _ in o_x_batches],
                    bits=bits,
                    group_size=group_size,
                    sym=sym,
                    quant_batch_size=batch_size,
                    prev_scale_hint=prev_scale_hint,
                )
                pending_scales.append(("o_proj", o_scales))
                layer_debug.append(
                    _AWQGroupDebug(
                        name="o_proj",
                        prev_op="v_proj",
                        layers=["o_proj"],
                        best_ratio=float(o_ratio),
                        best_error=float(o_error),
                        max_abs_scale=float(o_scales.abs().max().item()),
                        min_abs_scale=float(o_scales.abs().min().item()),
                    )
                )

        prev_scale_hint = o_proj_scale_hint if handles.v_proj.weight.shape == handles.o_proj.weight.shape else None
        gate_x_batches = [batch.gate_x for batch in captured_batches if batch.gate_x is not None]
        mlp_out_batches = [batch.mlp_out for batch in captured_batches if batch.mlp_out is not None]
        if gate_x_batches and mlp_out_batches:
            gate_scales, gate_error, gate_ratio = _search_best_scale(
                model=model,
                linears=[handles.gate_proj, handles.up_proj],
                x_batches=gate_x_batches,
                fp_outputs=mlp_out_batches,
                module2inspect=layer.mlp,
                module_kwargs_batches=[{} for _ in gate_x_batches],
                bits=bits,
                group_size=group_size,
                sym=sym,
                quant_batch_size=batch_size,
                prev_scale_hint=prev_scale_hint,
            )
            pending_scales.append(("gate_up", gate_scales))
            layer_debug.append(
                _AWQGroupDebug(
                    name="gate_up",
                    prev_op="post_attention_layernorm",
                    layers=["gate_proj", "up_proj"],
                    best_ratio=float(gate_ratio),
                    best_error=float(gate_error),
                    max_abs_scale=float(gate_scales.abs().max().item()),
                        min_abs_scale=float(gate_scales.abs().min().item()),
                    )
                )

        prev_scale_hint = gate_up_scale_hint
        down_x_batches = [batch.down_x for batch in captured_batches if batch.down_x is not None]
        down_out_batches = [batch.down_out for batch in captured_batches if batch.down_out is not None]
        if down_x_batches and down_out_batches:
            down_scales, down_error, down_ratio = _search_best_scale(
                model=model,
                linears=[handles.down_proj],
                x_batches=down_x_batches,
                fp_outputs=down_out_batches,
                module2inspect=handles.down_proj,
                module_kwargs_batches=[{} for _ in down_x_batches],
                bits=bits,
                group_size=group_size,
                sym=sym,
                quant_batch_size=batch_size,
                prev_scale_hint=prev_scale_hint,
            )
            pending_scales.append(("down_proj", down_scales))
            layer_debug.append(
                _AWQGroupDebug(
                    name="down_proj",
                    prev_op="up_proj",
                    layers=["down_proj"],
                    best_ratio=float(down_ratio),
                    best_error=float(down_error),
                    max_abs_scale=float(down_scales.abs().max().item()),
                    min_abs_scale=float(down_scales.abs().min().item()),
                )
            )

        for group_name, scales in pending_scales:
            if group_name == "qkv":
                _scale_ln_fcs(handles.input_layernorm, [handles.q_proj, handles.k_proj, handles.v_proj], scales)
            elif group_name == "o_proj":
                _scale_fc_fc(handles.v_proj, handles.o_proj, scales)
            elif group_name == "gate_up":
                _scale_ln_fcs(handles.post_attention_layernorm, [handles.gate_proj, handles.up_proj], scales)
            elif group_name == "down_proj":
                _scale_fc_fc(handles.up_proj, handles.down_proj, scales)
            else:
                raise ValueError(f"unsupported AWQ group {group_name!r}")

        next_hidden_batches = []
        for hidden_cpu, layer_kwargs_template in zip(current_hidden_batches, layer_kwargs_templates):
            hidden_dev = hidden_cpu.to(device)
            layer_kwargs = _move_kwargs_to_device(copy.deepcopy(layer_kwargs_template), hidden_dev.device)
            out = layer(hidden_dev, **layer_kwargs)
            next_hidden_batches.append(_to_cpu_detached(_extract_tensor(out)))
        current_hidden_batches = next_hidden_batches

        debug.append(
            {
                "layer_index": int(layer_idx),
                "groups": [
                    {
                        "name": item.name,
                        "prev_op": item.prev_op,
                        "layers": item.layers,
                        "best_ratio": item.best_ratio,
                        "best_error": item.best_error,
                        "max_abs_scale": item.max_abs_scale,
                        "min_abs_scale": item.min_abs_scale,
                    }
                    for item in layer_debug
                ],
            }
        )

        layer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "layers": debug,
        "bits": int(bits),
        "group_size": int(group_size),
        "sym": bool(sym),
        "batch_size": int(batch_size),
    }


__all__ = ["apply_awq_scaling"]
