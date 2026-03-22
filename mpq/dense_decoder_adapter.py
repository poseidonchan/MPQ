from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers import AutoConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


@dataclass(frozen=True)
class DenseDecoderLayerHandles:
    q_proj: torch.nn.Module
    k_proj: torch.nn.Module
    v_proj: torch.nn.Module
    o_proj: torch.nn.Module
    gate_proj: torch.nn.Module
    up_proj: torch.nn.Module
    down_proj: torch.nn.Module
    input_layernorm: torch.nn.Module
    post_attention_layernorm: torch.nn.Module


@dataclass(frozen=True)
class DenseDecoderAdapter:
    name: str
    model_types: frozenset[str]

    def supports(self, model_type: str) -> bool:
        return str(model_type).strip().lower() in self.model_types

    def get_model_root(self, model: Any) -> Any:
        root = getattr(model, "model", None)
        if root is not None and hasattr(root, "layers") and hasattr(root, "embed_tokens"):
            return root
        raise RuntimeError(f"{self.name}: could not locate decoder root at `model.model`")

    def get_layers(self, model: Any) -> Sequence[Any]:
        root = self.get_model_root(model)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise RuntimeError(f"{self.name}: could not locate decoder layers")
        return layers

    def get_layer_handles(self, layer: Any) -> DenseDecoderLayerHandles:
        try:
            attn = layer.self_attn
            mlp = layer.mlp
            return DenseDecoderLayerHandles(
                q_proj=attn.q_proj,
                k_proj=attn.k_proj,
                v_proj=attn.v_proj,
                o_proj=attn.o_proj,
                gate_proj=mlp.gate_proj,
                up_proj=mlp.up_proj,
                down_proj=mlp.down_proj,
                input_layernorm=layer.input_layernorm,
                post_attention_layernorm=layer.post_attention_layernorm,
            )
        except AttributeError as exc:
            raise RuntimeError(
                f"{self.name}: layer does not expose the expected dense decoder modules"
            ) from exc

    def compute_position_embeddings(
        self,
        model: Any,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        root = self.get_model_root(model)
        rotary = getattr(root, "rotary_emb", None)
        if rotary is None:
            return None
        return rotary(hidden_states, position_ids)

    def build_attention_mask(
        self,
        model: Any,
        *,
        batch_size: int,
        ctx_size: int,
        hidden_states: torch.Tensor,
        layer: Optional[Any] = None,
    ) -> torch.Tensor:
        sliding_window = self.get_layer_sliding_window(model, layer=layer)
        return _prepare_4d_causal_attention_mask(
            None,
            (int(batch_size), int(ctx_size)),
            hidden_states,
            0,
            sliding_window=sliding_window,
        )

    def get_layer_sliding_window(self, model: Any, *, layer: Optional[Any] = None) -> Optional[int]:
        config = getattr(model, "config", None)
        return getattr(config, "sliding_window", None)

    def forward_layer(
        self,
        model: Any,
        layer: Any,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        kwargs = {
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": bool(use_cache),
            "output_attentions": bool(output_attentions),
        }
        sig = inspect.signature(layer.forward)
        if "position_embeddings" in sig.parameters:
            pos_emb = self.compute_position_embeddings(model, hidden_states, position_ids)
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb
        out = layer(hidden_states, **kwargs)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def extract_logical_weights(self, model: Any, layer_idx: int, suffix: str) -> list[torch.Tensor]:
        layer = self.get_layers(model)[int(layer_idx)]
        h = self.get_layer_handles(layer)
        suffix = str(suffix).strip().lower()
        if suffix == "qkv":
            return [h.q_proj.weight, h.k_proj.weight, h.v_proj.weight]
        if suffix == "o":
            return [h.o_proj.weight]
        if suffix == "down":
            return [h.down_proj.weight]
        if suffix == "up":
            # Match QUIP# convention: stack up + gate in the fused "up" checkpoint.
            return [h.up_proj.weight, h.gate_proj.weight]
        raise ValueError(f"unsupported suffix={suffix!r}; expected qkv|o|up|down")

    def extra_state_dict(self, layer: Any) -> dict[str, torch.Tensor]:
        return {}

    def load_extra_state_dict(self, layer: Any, state: dict[str, torch.Tensor]) -> None:
        del layer, state


class QwenLikeDenseDecoderAdapter(DenseDecoderAdapter):
    def get_layer_sliding_window(self, model: Any, *, layer: Optional[Any] = None) -> Optional[int]:
        del model
        if layer is None:
            return None
        return getattr(getattr(layer, "self_attn", None), "sliding_window", None)

    def extra_state_dict(self, layer: Any) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        attn = getattr(layer, "self_attn", None)
        for name in ("q_norm", "k_norm"):
            norm = getattr(attn, name, None)
            weight = getattr(norm, "weight", None)
            if isinstance(weight, torch.Tensor):
                out[name] = weight.detach().cpu()
            bias = getattr(norm, "bias", None)
            if isinstance(bias, torch.Tensor):
                out[f"{name}_bias"] = bias.detach().cpu()
        return out

    def load_extra_state_dict(self, layer: Any, state: dict[str, torch.Tensor]) -> None:
        attn = getattr(layer, "self_attn", None)
        for name in ("q_norm", "k_norm"):
            norm = getattr(attn, name, None)
            if norm is None:
                continue
            weight = getattr(norm, "weight", None)
            if isinstance(weight, torch.Tensor) and name in state:
                weight.copy_(state[name])
            bias = getattr(norm, "bias", None)
            bias_key = f"{name}_bias"
            if isinstance(bias, torch.Tensor) and bias_key in state:
                bias.copy_(state[bias_key])


_DENSE_DECODER_ADAPTERS: tuple[DenseDecoderAdapter, ...] = (
    QwenLikeDenseDecoderAdapter(
        name="qwen_like_dense_decoder",
        model_types=frozenset(
            {
                "qwen2",
                "qwen3",
            }
        ),
    ),
    DenseDecoderAdapter(
        name="llama_like_dense_decoder",
        model_types=frozenset(
            {
                "llama",
                "mistral",
            }
        ),
    ),
)


def resolve_dense_decoder_adapter_for_model(model: Any, *, override: Optional[str] = None) -> DenseDecoderAdapter:
    if override and str(override).strip().lower() not in {"", "auto"}:
        requested = str(override).strip().lower()
        for adapter in _DENSE_DECODER_ADAPTERS:
            if adapter.name == requested:
                return adapter
        raise RuntimeError(f"unsupported decoder adapter override: {override!r}")

    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").strip().lower()
    for adapter in _DENSE_DECODER_ADAPTERS:
        if adapter.supports(model_type):
            return adapter
    raise RuntimeError(f"unsupported dense decoder model_type for QUIP/MPQ path: {model_type!r}")


def resolve_dense_decoder_adapter_for_model_id(model_id: str, *, override: Optional[str] = None) -> DenseDecoderAdapter:
    if override and str(override).strip().lower() not in {"", "auto"}:
        requested = str(override).strip().lower()
        for adapter in _DENSE_DECODER_ADAPTERS:
            if adapter.name == requested:
                return adapter
        raise RuntimeError(f"unsupported decoder adapter override: {override!r}")

    cfg = AutoConfig.from_pretrained(str(model_id))
    model_type = str(getattr(cfg, "model_type", "") or "").strip().lower()
    for adapter in _DENSE_DECODER_ADAPTERS:
        if adapter.supports(model_type):
            return adapter
    raise RuntimeError(f"unsupported dense decoder model_type for QUIP/MPQ path: {model_type!r}")
