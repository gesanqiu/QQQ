"""Quantized wrappers for Qwen3 smooth calibration."""

from types import SimpleNamespace
from typing import Callable

import logging
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QQQ.smooth.migration.migration_qwen3 import migration
from QQQ.smooth.quantization import QuantizedLayer, QuantizedModule, Quantizer

logger = logging.getLogger("QQQ")


class QuantizedQwen3MLP(Qwen3MLP, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3MLP, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig

        if hasattr(org_module, "config"):
            self.config = org_module.config
        else:
            self.config = SimpleNamespace()
            self.config.hidden_size = org_module.hidden_size
            self.config.intermediate_size = org_module.intermediate_size
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
            logger.info(
                f"the original min range is {hidden_states.min()}, the original max range is {hidden_states.max()}"
            )

            weight_list = torch.cat([self.gate_proj.module.weight, self.up_proj.module.weight])
            extra_dict = {"observation_mask": observation_mask, "act_fn": self.act_fn}
            best_scale = migration(
                hidden_states,
                weight_list,
                None,
                self.a_qconfig,
                self.w_qconfig,
                "up_and_gate",
                extra_dict,
            )
            hidden_states /= best_scale
            self.gate_proj.module.weight.data *= best_scale
            self.up_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)

        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

        if not self.a_qconfig.disable_down_proj and self.cac_migrate:
            logger.info(
                f"the original min range is {hidden_states.min()}, the original max range is {hidden_states.max()}"
            )
            weight_list = torch.cat([self.down_proj.module.weight])
            extra_dict = {
                "observation_mask": observation_mask,
            }
            best_scale = migration(
                hidden_states,
                weight_list,
                None,
                self.a_qconfig,
                self.w_qconfig,
                "down_proj",
                extra_dict,
            )
            hidden_states /= best_scale
            self.down_proj.module.weight.data *= best_scale
        hidden_states = self.down_proj(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedQwen3Attention(Qwen3Attention, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3Attention, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.act_fake_quant = Quantizer(None, a_qconfig)

        self.layer_type = org_module.config.layer_types[org_module.layer_idx] if hasattr(org_module.config, "layer_types") else None
        self.config = org_module.config
        self.layer_idx = org_module.layer_idx
        self.head_dim = org_module.head_dim
        self.num_key_value_groups = org_module.num_key_value_groups
        self.scaling = org_module.scaling
        self.attention_dropout = org_module.attention_dropout
        self.is_causal = org_module.is_causal
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)
        self.sliding_window = org_module.config.sliding_window if self.layer_type == "sliding_attention" else None
        self.q_norm = org_module.q_norm
        self.k_norm = org_module.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        observation_mask = kwargs["observation_mask"]

        if self.cac_migrate:
            logger.info(
                f"the original min range is {hidden_states.min()}, the original max range is {hidden_states.max()}"
            )
            weight_list = torch.cat(
                [
                    self.q_proj.module.weight,
                    self.k_proj.module.weight,
                    self.v_proj.module.weight,
                ]
            )
            bias_list = None
            if self.q_proj.module.bias is not None:
                bias_list = torch.cat(
                    [
                        self.q_proj.module.bias,
                        self.k_proj.module.bias,
                        self.v_proj.module.bias,
                    ]
                )
            extra_dict = {
                "num_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "num_key_value_groups": self.num_key_value_groups,
                "position_embeddings": position_embeddings,
                "head_dim": self.head_dim,
                "attention_mask": attention_mask,
                "observation_mask": observation_mask,
                "q_norm_weight": self.q_norm.weight,
                "k_norm_weight": self.k_norm.weight,
                "rms_norm_eps": self.config.rms_norm_eps,
            }
            best_scale = migration(
                hidden_states,
                weight_list,
                bias_list,
                self.a_qconfig,
                self.w_qconfig,
                "qkv",
                extra_dict,
            )
            hidden_states /= best_scale
            self.q_proj.module.weight.data *= best_scale
            self.k_proj.module.weight.data *= best_scale
            self.v_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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

        if self.cac_migrate:
            logger.info(f"the original min range is {attn_output.min()}, the original max range is {attn_output.max()}")
            weight_list = torch.cat([self.o_proj.module.weight])
            extra_dict = {
                "observation_mask": observation_mask,
            }
            best_scale = migration(
                attn_output,
                weight_list,
                None,
                self.a_qconfig,
                self.w_qconfig,
                "o_proj",
                extra_dict,
            )
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)
        return attn_output, attn_weights


class QuantizedQwen3DecoderLayer(Qwen3DecoderLayer, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3DecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput

        self.hidden_size = org_module.hidden_size

        self.self_attn = QuantizedQwen3Attention(
            org_module.self_attn,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )

        self.mlp = QuantizedQwen3MLP(
            org_module.mlp,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.input_layernorm = org_module.input_layernorm
        self.post_attention_layernorm = org_module.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states


class QuantizedQwen3Model(Qwen3Model, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen3Model, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput

        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList(
            [
                QuantizedQwen3DecoderLayer(org_module.layers[i], w_qconfig, a_qconfig, qinput=True)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = org_module.norm
        self.rotary_emb = org_module.rotary_emb
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in org_module.config.layer_types

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        observation_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        assert observation_mask is not None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        layer_types = getattr(self.config, "layer_types", None) or ["full_attention"] * self.config.num_hidden_layers

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                observation_mask=observation_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QuantizedQwen3ForCausalLM(Qwen3ForCausalLM, QuantizedModule):
    def __init__(
        self,
        org_module,
        w_qconfig,
        a_qconfig,
        qinput=True,
        backend="academic",
        is_remove_padding=False,
    ):
        super(Qwen3ForCausalLM, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self._no_split_modules = [
            "QuantizedQwen3DecoderLayer",
            "QuantizedQwen3Attention",
            "QuantizedQwen3MLP",
            "QuantizedLayer",
            "QuantizedModule",
        ]
        self.qinput = qinput

        self.model = QuantizedQwen3Model(org_module.model, w_qconfig, a_qconfig, self.qinput, backend=self.backend)
        self.vocab_size = org_module.vocab_size
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.is_remove_padding and attention_mask is not None:
            observation_mask = attention_mask.clone()
        else:
            observation_mask = None

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            observation_mask=observation_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
