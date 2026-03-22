from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import os
import weakref
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .dense_decoder_adapter import DenseDecoderAdapter, resolve_dense_decoder_adapter_for_model


class SharedFusedSliceLinear(nn.Module):
    """Expose one slice of a fused module as a normal linear-like module."""

    def __init__(
        self,
        owner: nn.Module,
        core_attr: str,
        slice_idx: int,
        fuse_sizes: Sequence[int],
        *,
        slice_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._owner_ref = weakref.ref(owner)
        self.core_attr = str(core_attr)
        self.slice_idx = int(slice_idx)
        self.fuse_sizes = tuple(int(x) for x in fuse_sizes)
        self.start = int(sum(self.fuse_sizes[: self.slice_idx]))
        self.end = int(self.start + self.fuse_sizes[self.slice_idx])
        if slice_bias is not None:
            self.register_buffer("slice_bias", slice_bias.detach().clone().to(torch.float32))
        else:
            self.slice_bias = None

    def _core(self) -> nn.Module:
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("shared fused owner was garbage collected")
        return getattr(owner, self.core_attr)

    @property
    def in_features(self) -> int:
        return int(getattr(self._core(), "in_features"))

    @property
    def out_features(self) -> int:
        return int(self.end - self.start)

    @property
    def weight(self) -> torch.Tensor:
        core = self._core()
        weight = getattr(core, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight[self.start : self.end]
        sv = getattr(core, "SV", None)
        if isinstance(sv, torch.Tensor):
            return sv[self.start : self.end]
        raise AttributeError("shared fused core has no weight-like tensor")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.slice_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self._core()(input)
        if not isinstance(out, (tuple, list)):
            raise RuntimeError(f"{self.core_attr} did not return split outputs")
        y = out[self.slice_idx]
        if self.slice_bias is not None:
            y = y + self.slice_bias.to(device=y.device, dtype=y.dtype)
        return y


class DenseDecoderMixedLayer(nn.Module):
    """A generic decoder-layer wrapper with tuple output for QUIP finetune."""

    def __init__(
        self,
        *,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        rotary_emb: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=bool(use_cache),
            output_attentions=bool(output_attentions),
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(attn_out, (tuple, list)):
            hidden_states = attn_out[0]
        else:
            hidden_states = attn_out
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class FusedUpGateMLP(nn.Module):
    def __init__(
        self,
        *,
        source_mlp: nn.Module,
        upgate_proj: nn.Module,
        up_bias: Optional[torch.Tensor] = None,
        gate_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.upgate_proj = upgate_proj
        self.down_proj = copy.deepcopy(source_mlp.down_proj).float().cpu()
        self.act_fn = source_mlp.act_fn
        if up_bias is not None:
            self.register_buffer("up_bias", up_bias.detach().clone().to(torch.float32))
        else:
            self.up_bias = None
        if gate_bias is not None:
            self.register_buffer("gate_bias", gate_bias.detach().clone().to(torch.float32))
        else:
            self.gate_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states, gate_states = self.upgate_proj(hidden_states)
        if self.up_bias is not None:
            up_states = up_states + self.up_bias.to(device=up_states.device, dtype=up_states.dtype)
        if self.gate_bias is not None:
            gate_states = gate_states + self.gate_bias.to(device=gate_states.device, dtype=gate_states.dtype)
        return self.down_proj(self.act_fn(gate_states) * up_states)


class QwenLikeMixedAttention(nn.Module):
    def __init__(
        self,
        *,
        source_attn: nn.Module,
        qkv_proj: nn.Module,
        q_bias: Optional[torch.Tensor] = None,
        k_bias: Optional[torch.Tensor] = None,
        v_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = source_attn.config
        self.layer_idx = source_attn.layer_idx
        self.head_dim = source_attn.head_dim
        self.num_key_value_groups = source_attn.num_key_value_groups
        self.scaling = source_attn.scaling
        self.attention_dropout = source_attn.attention_dropout
        self.is_causal = source_attn.is_causal
        self.sliding_window = getattr(source_attn, "sliding_window", None)
        self.qkv_proj = qkv_proj
        self.o_proj = copy.deepcopy(source_attn.o_proj).float().cpu()
        self.q_norm = copy.deepcopy(getattr(source_attn, "q_norm", None)).float().cpu() if getattr(source_attn, "q_norm", None) is not None else None
        self.k_norm = copy.deepcopy(getattr(source_attn, "k_norm", None)).float().cpu() if getattr(source_attn, "k_norm", None) is not None else None
        module = inspect.getmodule(type(source_attn))
        if module is None:
            raise RuntimeError("could not resolve Qwen attention module")
        self._apply_rotary_pos_emb: Callable[..., Any] = getattr(module, "apply_rotary_pos_emb")
        self._eager_attention_forward: Callable[..., Any] = getattr(module, "eager_attention_forward")
        self._all_attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS")
        if q_bias is not None:
            self.register_buffer("q_bias", q_bias.detach().clone().to(torch.float32))
        else:
            self.q_bias = None
        if k_bias is not None:
            self.register_buffer("k_bias", k_bias.detach().clone().to(torch.float32))
        else:
            self.k_bias = None
        if v_bias is not None:
            self.register_buffer("v_bias", v_bias.detach().clone().to(torch.float32))
        else:
            self.v_bias = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.qkv_proj(hidden_states)
        if self.q_bias is not None:
            query_states = query_states + self.q_bias.to(device=query_states.device, dtype=query_states.dtype)
        if self.k_bias is not None:
            key_states = key_states + self.k_bias.to(device=key_states.device, dtype=key_states.dtype)
        if self.v_bias is not None:
            value_states = value_states + self.v_bias.to(device=value_states.device, dtype=value_states.dtype)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable[..., Any] = self._eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = self._all_attention_functions[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class QwenLikeMixedLayer(nn.Module):
    def __init__(
        self,
        *,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        rotary_emb: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = _prepare_4d_causal_attention_mask(
                None,
                (int(hidden_states.shape[0]), int(hidden_states.shape[1])),
                hidden_states,
                0,
                sliding_window=getattr(self.self_attn, "sliding_window", None),
            )
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=bool(use_cache),
            output_attentions=bool(output_attentions),
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


def _check_exist(idx: int, save_path: str) -> bool:
    suffixes = ("qkv", "o", "up", "down", "layernorm")
    return all(Path(save_path, f"{int(idx)}_{suffix}.pt").exists() for suffix in suffixes)


def _split_optional_bias(bias: Optional[torch.Tensor], sizes: Sequence[int]) -> List[Optional[torch.Tensor]]:
    if bias is None:
        return [None for _ in sizes]
    out: List[Optional[torch.Tensor]] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        out.append(bias[start:end].detach().clone())
        start = end
    return out


def _quantize_weights_to_attr(
    *,
    weights: List[torch.Tensor],
    save_name: str,
    hessian_path: str,
    cb: Any,
    args: Any,
    device: Any,
) -> Tuple[Dict[str, Any], float, float]:
    from lib import utils as quip_utils
    from lib.algo import quip

    dtype_ = torch.float64 if bool(args.use_fp64) else torch.float32
    shapes = [w.shape for w in weights]
    scales = [w.to(dtype_).square().mean().sqrt() for w in weights]

    H_data = torch.load(hessian_path, map_location=torch.device("cpu"))
    H = quip_utils.flat_to_sym(H_data["flatH"], H_data["n"])
    mu = H_data["mu"]
    H.add_(mu[None, :] * mu[:, None])
    n = int(H_data["n"])
    W = torch.vstack([weights[i].to(dtype_) / scales[i] for i in range(len(weights))]).to(dtype_)
    H = quip_utils.regularize_H(H, n, float(args.sigma_reg))
    base_seed = int(getattr(args, "seed", 0) or 0)
    digest = hashlib.sha256(str(save_name).encode("utf-8")).digest()
    local_seed = int((base_seed + int.from_bytes(digest[:8], byteorder="little", signed=False)) % (2**31 - 1))
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)
        hatW, attr = quip.quantize(H, W, int(args.lora_rank), cb, args, device, module_name=str(save_name))
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    if len(scales) == 1:
        attr["SV"] *= scales[0]
        scales = [1.0]
    attr.update(
        {
            "fused": len(shapes) > 1,
            "shapes": shapes,
            "scales": scales,
        }
    )

    diff = hatW.to(dtype_) - W
    denom_frob = float(W.square().sum().clamp_min_(1e-12).item())
    denom_proxy = float(((W @ H) * W).sum().abs().clamp_min_(1e-12).item())
    err_frob = float(diff.square().sum().item() / denom_frob)
    err_proxy = float((((diff @ H) * diff).sum().item()) / denom_proxy)
    return attr, err_frob, err_proxy


def _save_quantized_attr(attr: Dict[str, Any], save_path: str) -> None:
    torch.save(attr, save_path)


def _quantize_linear_generic_fail_closed(
    *,
    weights: List[torch.Tensor],
    save_path: str,
    hessian_path: str,
    cb: Any,
    args: Any,
    device: Any,
    enable_safe_tune: bool,
) -> Dict[str, Any]:
    tune_iters = int(getattr(args, "quip_tune_iters", 0) or 0)
    if not enable_safe_tune or tune_iters <= 0:
        attr, err_frob, err_proxy = _quantize_weights_to_attr(
            weights=weights,
            save_name=os.path.basename(save_path),
            hessian_path=hessian_path,
            cb=cb,
            args=args,
            device=device,
        )
        attr["selected_quip_tune_iters"] = int(tune_iters)
        attr["selected_frob_error"] = float(err_frob)
        attr["selected_proxy_error"] = float(err_proxy)
        _save_quantized_attr(attr, save_path)
        return attr

    base_args = copy.deepcopy(args)
    base_args.quip_tune_iters = 0
    base_attr, base_frob, base_proxy = _quantize_weights_to_attr(
        weights=weights,
        save_name=os.path.basename(save_path),
        hessian_path=hessian_path,
        cb=cb,
        args=base_args,
        device=device,
    )
    tuned_attr, tuned_frob, tuned_proxy = _quantize_weights_to_attr(
        weights=weights,
        save_name=os.path.basename(save_path),
        hessian_path=hessian_path,
        cb=cb,
        args=args,
        device=device,
    )

    if tuned_frob <= base_frob:
        chosen_attr = tuned_attr
        chosen_iters = int(tune_iters)
        alt_iters = 0
        alt_frob = float(base_frob)
        alt_proxy = float(base_proxy)
        chosen_frob = float(tuned_frob)
        chosen_proxy = float(tuned_proxy)
    else:
        chosen_attr = base_attr
        chosen_iters = 0
        alt_iters = int(tune_iters)
        alt_frob = float(tuned_frob)
        alt_proxy = float(tuned_proxy)
        chosen_frob = float(base_frob)
        chosen_proxy = float(base_proxy)

    chosen_attr["selected_quip_tune_iters"] = int(chosen_iters)
    chosen_attr["selected_frob_error"] = float(chosen_frob)
    chosen_attr["selected_proxy_error"] = float(chosen_proxy)
    chosen_attr["alternate_quip_tune_iters"] = int(alt_iters)
    chosen_attr["alternate_frob_error"] = float(alt_frob)
    chosen_attr["alternate_proxy_error"] = float(alt_proxy)
    _save_quantized_attr(chosen_attr, save_path)
    return chosen_attr


def build_mixed_layer(
    *,
    layer: nn.Module,
    model: Any,
    adapter: DenseDecoderAdapter,
) -> nn.Module:
    handles = adapter.get_layer_handles(layer)
    layer_copy = copy.deepcopy(layer).float().cpu()
    copy_handles = adapter.get_layer_handles(layer_copy)

    attn_biases = [
        None if handles.q_proj.bias is None else handles.q_proj.bias.detach().cpu().float(),
        None if handles.k_proj.bias is None else handles.k_proj.bias.detach().cpu().float(),
        None if handles.v_proj.bias is None else handles.v_proj.bias.detach().cpu().float(),
    ]
    qkv_weights = [
        copy_handles.q_proj.weight.detach().cpu().float(),
        copy_handles.k_proj.weight.detach().cpu().float(),
        copy_handles.v_proj.weight.detach().cpu().float(),
    ]
    from lib.linear.fused_linear import FusedLinear

    qkv_sizes = [int(w.shape[0]) for w in qkv_weights]
    qkv_proj = FusedLinear(-1, qkv_sizes, int(qkv_weights[0].shape[1]), int(sum(qkv_sizes)), bias=False)
    with torch.no_grad():
        cur = 0
        for w in qkv_weights:
            qkv_proj.weight[cur : cur + w.shape[0]].copy_(w)
            cur += int(w.shape[0])
    copy_handles = adapter.get_layer_handles(layer_copy)
    layer_copy.self_attn.qkv_proj = qkv_proj
    layer_copy.self_attn.q_bias = attn_biases[0]
    layer_copy.self_attn.k_bias = attn_biases[1]
    layer_copy.self_attn.v_bias = attn_biases[2]
    layer_copy.self_attn.q_proj = SharedFusedSliceLinear(layer_copy.self_attn, "qkv_proj", 0, qkv_sizes, slice_bias=attn_biases[0])
    layer_copy.self_attn.k_proj = SharedFusedSliceLinear(layer_copy.self_attn, "qkv_proj", 1, qkv_sizes, slice_bias=attn_biases[1])
    layer_copy.self_attn.v_proj = SharedFusedSliceLinear(layer_copy.self_attn, "qkv_proj", 2, qkv_sizes, slice_bias=attn_biases[2])

    mlp_biases = [
        None if handles.up_proj.bias is None else handles.up_proj.bias.detach().cpu().float(),
        None if handles.gate_proj.bias is None else handles.gate_proj.bias.detach().cpu().float(),
    ]
    up_weights = [
        copy_handles.up_proj.weight.detach().cpu().float(),
        copy_handles.gate_proj.weight.detach().cpu().float(),
    ]
    up_sizes = [int(w.shape[0]) for w in up_weights]
    upgate_proj = FusedLinear(-1, up_sizes, int(up_weights[0].shape[1]), int(sum(up_sizes)), bias=False)
    with torch.no_grad():
        cur = 0
        for w in up_weights:
            upgate_proj.weight[cur : cur + w.shape[0]].copy_(w)
            cur += int(w.shape[0])
    layer_copy.mlp.upgate_proj = upgate_proj
    layer_copy.mlp.up_bias = mlp_biases[0]
    layer_copy.mlp.gate_bias = mlp_biases[1]
    layer_copy.mlp.up_proj = SharedFusedSliceLinear(layer_copy.mlp, "upgate_proj", 0, up_sizes, slice_bias=mlp_biases[0])
    layer_copy.mlp.gate_proj = SharedFusedSliceLinear(layer_copy.mlp, "upgate_proj", 1, up_sizes, slice_bias=mlp_biases[1])

    rotary = copy.deepcopy(getattr(adapter.get_model_root(model), "rotary_emb", None))
    if rotary is not None:
        rotary = rotary.cpu()
    if adapter.name == "qwen_like_dense_decoder":
        return QwenLikeMixedLayer(
            self_attn=QwenLikeMixedAttention(
                source_attn=layer_copy.self_attn,
                qkv_proj=qkv_proj,
                q_bias=attn_biases[0],
                k_bias=attn_biases[1],
                v_bias=attn_biases[2],
            ),
            mlp=FusedUpGateMLP(
                source_mlp=layer_copy.mlp,
                upgate_proj=upgate_proj,
                up_bias=mlp_biases[0],
                gate_bias=mlp_biases[1],
            ),
            input_layernorm=copy.deepcopy(layer_copy.input_layernorm).float().cpu(),
            post_attention_layernorm=copy.deepcopy(layer_copy.post_attention_layernorm).float().cpu(),
            rotary_emb=rotary,
        )
    return DenseDecoderMixedLayer(
        self_attn=layer_copy.self_attn,
        mlp=layer_copy.mlp,
        input_layernorm=layer_copy.input_layernorm,
        post_attention_layernorm=layer_copy.post_attention_layernorm,
        rotary_emb=rotary,
    )


def quantize_finetune_dense_decoder_layer(
    mixed_layer: nn.Module,
    quant_order: Sequence[tuple[str, str]],
    idx: int,
    cb: Any,
    args: Any,
    device: Any,
    pre_orig_emb: torch.Tensor,
    orig_emb: torch.Tensor,
    *,
    adapter: DenseDecoderAdapter,
) -> None:
    from lib import codebook, utils
    from lib.algo import finetune
    from lib.linear.fused_linear import FusedLinear
    from lib.linear.fused_quantized_linear import FusedQuantizedLinear
    from lib.linear.quantized_linear import QuantizedLinear

    torch.manual_seed(idx)
    torch.set_num_threads(int(args.num_cpu_threads))

    codebook_id = codebook.get_id(args.codebook)
    mixed_layer = mixed_layer.float()
    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype), cb.version)
    shared_kwargs = {
        "rank": int(args.lora_rank),
        "rescale_WH": bool(args.rescale_WH),
        "resid_scale_override": float(args.resid_scale_override),
        "bias": False,
        "train_mode": bool(args.ft_train_mode),
        "grad_ckpt": bool(args.ft_grad_ckpt),
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f"{args.save_path}/{idx}_{name}.pt"
        hessian_path = f"{args.hessian_path}/{idx}_{name}.pt"
        split_biases: Optional[List[Optional[torch.Tensor]]] = None
        single_bias: Optional[torch.Tensor] = None

        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(orig_linear.weight, orig_linear.fuse_sizes, 0)
                if name == "qkv":
                    split_biases = [
                        getattr(mixed_layer.self_attn, "q_bias", None),
                        getattr(mixed_layer.self_attn, "k_bias", None),
                        getattr(mixed_layer.self_attn, "v_bias", None),
                    ]
                elif name == "up":
                    split_biases = [
                        getattr(mixed_layer.mlp, "up_bias", None),
                        getattr(mixed_layer.mlp, "gate_bias", None),
                    ]
            else:
                weights = [orig_linear.weight]
                if getattr(orig_linear, "bias", None) is not None:
                    single_bias = orig_linear.bias.detach().cpu().float()

            saved_linear = _quantize_linear_generic_fail_closed(
                weights=[w.detach().cpu() for w in weights],
                save_path=save_path,
                hessian_path=hessian_path,
                cb=cb,
                args=args,
                device=device,
                enable_safe_tune=(adapter.name == "qwen_like_dense_decoder"),
            )
            dirty = False
            if split_biases is not None and any(b is not None for b in split_biases):
                saved_linear["split_biases"] = [
                    None if b is None else b.detach().cpu().float() for b in split_biases
                ]
                dirty = True
            if single_bias is not None:
                saved_linear["bias"] = single_bias
                dirty = True
            if dirty:
                torch.save(saved_linear, save_path)

            if bool(saved_linear["fused"]):
                quant_linear = FusedQuantizedLinear(
                    -1,
                    [_[0] for _ in saved_linear["shapes"]],
                    saved_linear["shapes"][0][1],
                    sum([_[0] for _ in saved_linear["shapes"]]),
                    *shared_args,
                    **shared_kwargs,
                )
                for i in range(len(saved_linear["scales"])):
                    quant_linear.fuse_scales[i].copy_(torch.as_tensor(saved_linear["scales"][i]))
            else:
                quant_linear = QuantizedLinear(
                    saved_linear["shapes"][0][1],
                    saved_linear["shapes"][0][0],
                    *shared_args,
                    bias=("bias" in saved_linear),
                    rescale_WH=bool(args.rescale_WH),
                    resid_scale_override=float(args.resid_scale_override),
                    rank=int(args.lora_rank),
                    train_mode=bool(args.ft_train_mode),
                    grad_ckpt=bool(args.ft_grad_ckpt),
                )
                if "bias" in saved_linear and quant_linear.has_bias:
                    quant_linear.bias.copy_(saved_linear["bias"].to(torch.float32))

            utils.unpack_quip(quant_linear, saved_linear, codebook_id, cb.codesz)

        quant_linear.SU = nn.Parameter(quant_linear.SU.float(), requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(), requires_grad=True)
        split_attr = linear_attr.split(".")
        setattr(attrgetter(".".join(split_attr[:-1]))(mixed_layer), split_attr[-1], quant_linear)
        if quant_i < len(quant_order) - 1:
            finetune.finetune_decoder_layer(mixed_layer, f"{idx}_{name}", device, train_dl, valid_dl, args)

    with torch.no_grad():
        utils.clean()
        for linear_attr, name in quant_order:
            finetune.utils.save_susv(attrgetter(linear_attr)(mixed_layer), f"{args.save_path}/{idx}_{name}.pt")

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def save_layernorm_state(
    mixed_layer: nn.Module,
    *,
    save_path: str,
    idx: int,
    adapter: DenseDecoderAdapter,
) -> None:
    payload = {
        "input_layernorm": mixed_layer.input_layernorm.weight.detach().cpu(),
        "post_attention_layernorm": mixed_layer.post_attention_layernorm.weight.detach().cpu(),
    }
    if getattr(mixed_layer.input_layernorm, "bias", None) is not None:
        payload["input_layernorm_bias"] = mixed_layer.input_layernorm.bias.detach().cpu()
    if getattr(mixed_layer.post_attention_layernorm, "bias", None) is not None:
        payload["post_attention_layernorm_bias"] = mixed_layer.post_attention_layernorm.bias.detach().cpu()
    payload.update(adapter.extra_state_dict(mixed_layer))
    torch.save(payload, f"{save_path}/{idx}_layernorm.pt")


def _decompress_saved_quip(saved_layer: dict[str, Any], cb: Any, args: Any, *, device: torch.device) -> torch.Tensor:
    from lib.algo import quip

    SU = saved_layer["SU"].to(device)
    SV = saved_layer["SV"].to(device)
    Qidxs = saved_layer["Qidxs"].to(device)
    n = len(SU)
    m = len(SV)
    hatWr = cb.to(device).by_idxs(Qidxs, packed=(cb.packsz != 1)).view(m, n)
    if int(getattr(args, "lora_rank", 0) or 0) > 0 and saved_layer.get("A") is not None:
        hatWr = hatWr + saved_layer["A"].to(device) @ saved_layer["B"].to(device)
    hatW = quip.incoherence_process(hatWr, SU, SV, saved_layer.get("scaleWH"), args)
    return hatW.to(torch.float32).cpu()


def _load_split_weights(path: str, cb: Any, args: Any) -> tuple[list[torch.Tensor], list[Optional[torch.Tensor]]]:
    saved_layer = torch.load(path, map_location="cpu", weights_only=False)
    decode_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hatW = _decompress_saved_quip(saved_layer, cb, args, device=decode_device)
    shapes = list(saved_layer["shapes"])
    sizes = [int(shape[0]) for shape in shapes]
    weights = list(torch.split(hatW, sizes, dim=0))
    scales = list(saved_layer.get("scales", [1.0 for _ in weights]))
    weights = [w.to(torch.float32) * torch.as_tensor(scales[i], dtype=torch.float32) for i, w in enumerate(weights)]
    if "split_biases" in saved_layer:
        biases = [None if b is None else b.to(torch.float32) for b in saved_layer["split_biases"]]
    elif "bias" in saved_layer:
        biases = [saved_layer["bias"].to(torch.float32)]
    else:
        biases = [None for _ in weights]
    return weights, biases


def apply_quantized_ckpt_to_model(
    *,
    model: Any,
    ckpt_dir: str,
    args: Any,
    adapter: DenseDecoderAdapter,
    cb: Any,
) -> None:
    with torch.no_grad():
        layers = adapter.get_layers(model)
        for idx, layer in enumerate(layers):
            handles = adapter.get_layer_handles(layer)
            ln_path = Path(ckpt_dir) / f"{idx}_layernorm.pt"
            if ln_path.exists():
                ln_data = torch.load(ln_path, map_location="cpu", weights_only=False)
                handles.input_layernorm.weight.copy_(ln_data["input_layernorm"])
                handles.post_attention_layernorm.weight.copy_(ln_data["post_attention_layernorm"])
                if getattr(handles.input_layernorm, "bias", None) is not None and "input_layernorm_bias" in ln_data:
                    handles.input_layernorm.bias.copy_(ln_data["input_layernorm_bias"])
                if getattr(handles.post_attention_layernorm, "bias", None) is not None and "post_attention_layernorm_bias" in ln_data:
                    handles.post_attention_layernorm.bias.copy_(ln_data["post_attention_layernorm_bias"])
                adapter.load_extra_state_dict(layer, ln_data)

            qkv_w, qkv_b = _load_split_weights(str(Path(ckpt_dir) / f"{idx}_qkv.pt"), cb, args)
            handles.q_proj.weight.copy_(qkv_w[0])
            handles.k_proj.weight.copy_(qkv_w[1])
            handles.v_proj.weight.copy_(qkv_w[2])
            if handles.q_proj.bias is not None and qkv_b[0] is not None:
                handles.q_proj.bias.copy_(qkv_b[0])
            if handles.k_proj.bias is not None and qkv_b[1] is not None:
                handles.k_proj.bias.copy_(qkv_b[1])
            if handles.v_proj.bias is not None and qkv_b[2] is not None:
                handles.v_proj.bias.copy_(qkv_b[2])

            o_w, o_b = _load_split_weights(str(Path(ckpt_dir) / f"{idx}_o.pt"), cb, args)
            handles.o_proj.weight.copy_(o_w[0])
            if handles.o_proj.bias is not None and o_b[0] is not None:
                handles.o_proj.bias.copy_(o_b[0])

            up_w, up_b = _load_split_weights(str(Path(ckpt_dir) / f"{idx}_up.pt"), cb, args)
            handles.up_proj.weight.copy_(up_w[0])
            handles.gate_proj.weight.copy_(up_w[1])
            if handles.up_proj.bias is not None and up_b[0] is not None:
                handles.up_proj.bias.copy_(up_b[0])
            if handles.gate_proj.bias is not None and up_b[1] is not None:
                handles.gate_proj.bias.copy_(up_b[1])

            down_w, down_b = _load_split_weights(str(Path(ckpt_dir) / f"{idx}_down.pt"), cb, args)
            handles.down_proj.weight.copy_(down_w[0])
            if handles.down_proj.bias is not None and down_b[0] is not None:
                handles.down_proj.bias.copy_(down_b[0])


def collect_hessians_generic(args: Any) -> None:
    from lib import utils as quip_utils

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    adapter = resolve_dense_decoder_adapter_for_model(model, override=getattr(args, "model_arch", None))
    root = adapter.get_model_root(model)
    devset = quip_utils.sample_token_blocks(
        tokenizer,
        dataset_name=args.calibration_dataset,
        split=args.calibration_split,
        size=args.devset_size,
        ctx_size=args.ctx_size,
        nproc=args.sample_proc,
        seed=args.seed,
    )
    dev_emb = root.embed_tokens(devset).cpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(args.batch_size, -1)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for layer_idx, layer in enumerate(adapter.get_layers(model)):
        expected = [save_path / f"{layer_idx}_{suffix}.pt" for suffix in ("qkv", "o", "up", "down")]
        if all(p.exists() for p in expected):
            continue

        layer = layer.to(device)
        handles = adapter.get_layer_handles(layer)
        attention_mask = adapter.build_attention_mask(
            model,
            batch_size=args.batch_size,
            ctx_size=args.ctx_size,
            hidden_states=dev_emb[: args.batch_size],
            layer=layer,
        )
        done_qkv = quip_utils.register_H_hook(handles.q_proj, device)
        done_o = quip_utils.register_H_hook(handles.o_proj, device)
        done_up = quip_utils.register_H_hook(handles.up_proj, device)
        done_down = quip_utils.register_H_hook(handles.down_proj, device)

        out_emb = torch.zeros_like(dev_emb)
        with torch.no_grad():
            for start in range(0, len(dev_emb), args.batch_size):
                end = min(start + args.batch_size, len(dev_emb))
                batch = dev_emb[start:end].to(device)
                batch_pos = position_ids[: end - start].to(device)
                batch_mask = attention_mask[: end - start].to(device) if attention_mask is not None else None
                out_emb[start:end] = adapter.forward_layer(
                    model,
                    layer,
                    batch,
                    position_ids=batch_pos,
                    attention_mask=batch_mask,
                    use_cache=False,
                    output_attentions=False,
                ).cpu()

        for name, done in {
            "qkv": done_qkv,
            "o": done_o,
            "up": done_up,
            "down": done_down,
        }.items():
            H, mu, ct = done()
            mu = mu / ct
            H = H / ct
            H.addmm_(-mu.unsqueeze(-1), mu.unsqueeze(0))
            torch.save(
                {
                    "flatH": quip_utils.sym_to_flat(H.to(torch.float32)),
                    "mu": mu.to(torch.float32),
                    "n": H.shape[0],
                    "ct": int(ct),
                },
                save_path / f"{layer_idx}_{name}.pt",
            )

        layer.cpu()
        dev_emb = out_emb
        torch.cuda.empty_cache()


def quantize_model_generic(args: Any) -> None:
    from lib import codebook as quip_codebook
    from lib import utils as quip_utils

    dtype_ = torch.float64 if bool(args.use_fp64) else torch.float32
    cb = quip_codebook.get_codebook(args.codebook)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    adapter = resolve_dense_decoder_adapter_for_model(model, override=getattr(args, "model_arch", None))
    root = adapter.get_model_root(model)

    all_config = {"quant_args": args, "model_config": model.config, "adapter_name": adapter.name}
    quip_params = {
        "lora_rank": args.lora_rank,
        "rescale_WH": args.rescale_WH,
        "codebook": args.codebook,
        "codebook_version": cb.version,
        "codesz": cb.codesz,
        "idx_dtype": str(cb.idx_dtype),
        "packsz": cb.packsz,
        "resid_scale_override": args.resid_scale_override,
        "adapter_name": adapter.name,
    }
    all_config["model_config"].update({"quip_params": quip_params})
    torch.save(all_config, os.path.join(args.save_path, "config.pt"))

    devset = quip_utils.sample_token_blocks(
        tokenizer,
        dataset_name=args.calibration_dataset,
        split=args.calibration_split,
        size=args.devset_size,
        ctx_size=args.ctx_size,
        nproc=args.sample_proc,
        seed=args.seed,
    )
    current_emb = root.embed_tokens(devset).cpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(args.batch_size, -1)
    layers = adapter.get_layers(model)
    for idx, layer in enumerate(layers):
        layer = layer.to(device)
        attention_mask = adapter.build_attention_mask(
            model,
            batch_size=args.batch_size,
            ctx_size=args.ctx_size,
            hidden_states=current_emb[: args.batch_size],
            layer=layer,
        )
        next_emb = torch.zeros_like(current_emb)
        for start in range(0, len(current_emb), args.batch_size):
            end = min(start + args.batch_size, len(current_emb))
            batch = current_emb[start:end].to(device)
            batch_pos = position_ids[: end - start].to(device)
            batch_mask = attention_mask[: end - start].to(device) if attention_mask is not None else None
            next_emb[start:end] = adapter.forward_layer(
                model,
                layer,
                batch,
                position_ids=batch_pos,
                attention_mask=batch_mask,
                use_cache=False,
                output_attentions=False,
            ).cpu()

        if not _check_exist(idx, args.save_path):
            mixed_layer = build_mixed_layer(layer=layer.cpu(), model=model, adapter=adapter)
            quantize_finetune_dense_decoder_layer(
                mixed_layer,
                [
                    ("self_attn.qkv_proj", "qkv"),
                    ("self_attn.o_proj", "o"),
                    ("mlp.upgate_proj", "up"),
                    ("mlp.down_proj", "down"),
                ],
                idx,
                cb,
                args,
                device,
                current_emb.to(dtype_),
                next_emb.to(dtype_),
                adapter=adapter,
            )
            save_layernorm_state(mixed_layer, save_path=args.save_path, idx=idx, adapter=adapter)

        layer.cpu()
        current_emb = next_emb
        torch.cuda.empty_cache()


def export_dense_hf_from_ckpt(args: Any) -> None:
    from lib import codebook as quip_codebook

    saved_config = torch.load(os.path.join(args.quantized_path, "config.pt"), map_location="cpu", weights_only=False)
    model_config = saved_config["model_config"]
    model = AutoModelForCausalLM.from_pretrained(
        model_config._name_or_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        config=model_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, use_fast=True)
    adapter = resolve_dense_decoder_adapter_for_model(model, override=getattr(saved_config.get("quant_args"), "model_arch", None))
    cb = quip_codebook.get_codebook(model_config.quip_params["codebook"])
    apply_quantized_ckpt_to_model(
        model=model,
        ckpt_dir=args.quantized_path,
        args=saved_config["quant_args"],
        adapter=adapter,
        cb=cb,
    )
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
