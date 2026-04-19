"""Quantized wrappers for Qwen3-VL smooth calibration.

Only the language decoder stack is wrapped; the vision encoder stays in FP.
Qwen3-VL attention adds q_norm/k_norm (head-dim RMSNorm) after q/k projections,
and the language_model accepts deepstack visual embeddings.
"""

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QQQ.smooth.migration.migration_qwen2_vl import migration
from QQQ.smooth.quantization import QuantizedLayer, QuantizedModule, Quantizer

logger = logging.getLogger("QQQ")


class QuantizedQwen3VLTextMLP(Qwen3VLTextMLP, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3VLTextMLP, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size
        self.intermediate_size = org_module.intermediate_size
        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.gate_proj = QuantizedLayer(org_module.gate_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.up_proj = QuantizedLayer(org_module.up_proj, None, w_qconfig, a_qconfig, self.qinput)
        if getattr(self.a_qconfig, "disable_down_proj", False):
            self.down_proj = org_module.down_proj
        else:
            self.a_qconfig.disable_down_proj = False
            self.down_proj = QuantizedLayer(org_module.down_proj, None, w_qconfig, a_qconfig, True)
        self.act_fn = org_module.act_fn

    def forward(self, hidden_states, **kwargs):
        observation_mask = kwargs["observation_mask"]
        if self.cac_migrate:
            weight_list = torch.cat([self.gate_proj.module.weight, self.up_proj.module.weight])
            extra_dict = {"observation_mask": observation_mask, "act_fn": self.act_fn}
            best_scale = migration(
                hidden_states, weight_list, None, self.a_qconfig, self.w_qconfig, "up_and_gate", extra_dict,
            )
            hidden_states /= best_scale
            self.gate_proj.module.weight.data *= best_scale
            self.up_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

        if not self.a_qconfig.disable_down_proj and self.cac_migrate:
            weight_list = torch.cat([self.down_proj.module.weight])
            extra_dict = {"observation_mask": observation_mask}
            best_scale = migration(
                hidden_states, weight_list, None, self.a_qconfig, self.w_qconfig, "down_proj", extra_dict,
            )
            hidden_states /= best_scale
            self.down_proj.module.weight.data *= best_scale
        hidden_states = self.down_proj(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedQwen3VLTextAttention(Qwen3VLTextAttention, QuantizedModule):
    """Qwen3-VL attention: no biases on q/k/v/o, has q_norm/k_norm."""

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3VLTextAttention, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.layer_idx = org_module.layer_idx

        self.attention_dropout = org_module.attention_dropout
        self.head_dim = org_module.head_dim
        self.num_key_value_groups = org_module.num_key_value_groups
        self.is_causal = org_module.is_causal
        self.scaling = org_module.scaling

        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)

        self.q_norm = org_module.q_norm
        self.k_norm = org_module.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        observation_mask = kwargs["observation_mask"]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.cac_migrate:
            weight_list = torch.cat([
                self.q_proj.module.weight, self.k_proj.module.weight, self.v_proj.module.weight,
            ])
            extra_dict = {
                "num_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "num_key_value_groups": self.num_key_value_groups,
                "position_embeddings": position_embeddings,
                "head_dim": self.head_dim,
                "attention_mask": attention_mask,
                "observation_mask": observation_mask,
            }
            best_scale = migration(
                hidden_states, weight_list, None, self.a_qconfig, self.w_qconfig, "qkv", extra_dict,
            )
            hidden_states /= best_scale
            self.q_proj.module.weight.data *= best_scale
            self.k_proj.module.weight.data *= best_scale
            self.v_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.cac_migrate:
            weight_list = torch.cat([self.o_proj.module.weight])
            extra_dict = {"observation_mask": observation_mask}
            best_scale = migration(
                attn_output, weight_list, None, self.a_qconfig, self.w_qconfig, "o_proj", extra_dict,
            )
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)
        return attn_output, attn_weights


class QuantizedQwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3VLTextDecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size
        self.self_attn = QuantizedQwen3VLTextAttention(
            org_module.self_attn, w_qconfig, a_qconfig, qinput=False,
        )
        self.mlp = QuantizedQwen3VLTextMLP(
            org_module.mlp, w_qconfig, a_qconfig, qinput=False,
        )
        self.input_layernorm = org_module.input_layernorm
        self.post_attention_layernorm = org_module.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states


class QuantizedQwen3VLTextModel(Qwen3VLTextModel, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3VLTextModel, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput
        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            self.layers.append(
                QuantizedQwen3VLTextDecoderLayer(org_module.layers[i], w_qconfig, a_qconfig, qinput=True)
            )
        self.rotary_emb = org_module.rotary_emb
        self.norm = org_module.norm
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
        observation_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        assert observation_mask is not None

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                observation_mask=observation_mask,
                **kwargs,
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx in range(len(deepstack_visual_embeds))
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


