"""Quantized wrappers for Qwen2-VL smooth calibration.

Only the language decoder stack is wrapped; the vision encoder stays in FP.
"""

import logging

import torch
from torch import nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP,
    Qwen2VLAttention,
    Qwen2VLDecoderLayer,
    Qwen2VLTextModel,
)

from QQQ.smooth.migration.migration_qwen2_vl import migration
from QQQ.smooth.quantization import QuantizedLayer, QuantizedModule, Quantizer

from .qwen2 import QuantizedQwen2MLP

logger = logging.getLogger("QQQ")


class QuantizedQwen2VLMLP(QuantizedQwen2MLP):
    """MLP wrapper that uses the VLM migration module to keep all scales in one list."""

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


class QuantizedQwen2VLAttention(Qwen2VLAttention, QuantizedModule):
    """Quantized Qwen2-VL text attention (q/k/v with biases, o_proj without)."""

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        nn.Module.__init__(self)
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.layer_idx = org_module.layer_idx

        self.attention_dropout = org_module.attention_dropout
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = org_module.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = org_module.is_causal
        self.scaling = org_module.scaling
        self.rope_parameters = getattr(org_module, "rope_parameters", None)

        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)

        layer_types = getattr(self.config, "layer_types", None)
        if layer_types and layer_types[self.layer_idx] == "sliding_attention":
            self.sliding_window = self.config.sliding_window
        else:
            self.sliding_window = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, eager_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        assert not output_attentions
        observation_mask = kwargs.get("observation_mask")
        bsz, q_len, _ = hidden_states.size()
        assert position_embeddings is not None

        if self.cac_migrate:
            weight_list = torch.cat([
                self.q_proj.module.weight, self.k_proj.module.weight, self.v_proj.module.weight,
            ])
            bias_list = torch.cat([
                self.q_proj.module.bias, self.k_proj.module.bias, self.v_proj.module.bias,
            ])
            extra_dict = {
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "num_key_value_groups": self.num_key_value_groups,
                "position_embeddings": position_embeddings,
                "head_dim": self.head_dim,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "observation_mask": observation_mask,
                "mrope_section": self.config.rope_parameters["mrope_section"],
            }
            best_scale = migration(
                hidden_states, weight_list, bias_list, self.a_qconfig, self.w_qconfig, "qkv", extra_dict,
            )
            hidden_states /= best_scale
            self.q_proj.module.weight.data *= best_scale
            self.k_proj.module.weight.data *= best_scale
            self.v_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        if self.cac_migrate:
            weight_list = torch.cat([self.o_proj.module.weight])
            extra_dict = {"observation_mask": observation_mask}
            best_scale = migration(
                attn_output, weight_list, None, self.a_qconfig, self.w_qconfig, "o_proj", extra_dict,
            )
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)
        return attn_output, attn_weights, past_key_value


class QuantizedQwen2VLDecoderLayer(Qwen2VLDecoderLayer, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2VLDecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size
        self.self_attn = QuantizedQwen2VLAttention(
            org_module.self_attn, w_qconfig, a_qconfig, qinput=False,
        )
        self.mlp = QuantizedQwen2VLMLP(
            org_module.mlp, w_qconfig, a_qconfig, qinput=False,
        )
        self.input_layernorm = org_module.input_layernorm
        self.post_attention_layernorm = org_module.post_attention_layernorm

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
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

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class QuantizedQwen2VLTextModel(Qwen2VLTextModel, QuantizedModule):
    """Wraps the language_model portion – replaces decoder layers with quantized versions."""

    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2VLTextModel, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput
        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            self.layers.append(
                QuantizedQwen2VLDecoderLayer(org_module.layers[i], w_qconfig, a_qconfig, qinput=True)
            )
        self.rotary_emb = org_module.rotary_emb
        self.norm = org_module.norm
        self.gradient_checkpointing = False
        self.has_sliding_layers = getattr(org_module, "has_sliding_layers", False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                observation_mask=None, **kwargs):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        device = inputs_embeds.device

        cache_position = kwargs.pop("cache_position", None)
        if cache_position is None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                observation_mask=observation_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


