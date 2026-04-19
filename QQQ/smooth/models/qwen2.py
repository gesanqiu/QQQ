"""PyTorch QuantizedLLaMA model."""

from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Union

import logging
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QQQ.smooth.migration.migration_qwen2 import migration
from QQQ.smooth.quantization import QuantizedLayer, QuantizedModule, Quantizer

logger = logging.getLogger("QQQ")


class QuantizedQwen2MLP(Qwen2MLP, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2MLP, self).__init__()
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

            # calculate scale
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
            # update scale
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
            # update scale
            hidden_states /= best_scale
            self.down_proj.module.weight.data *= best_scale
        hidden_states = self.down_proj(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedQwen2Attention(Qwen2Attention, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2Attention, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.layer_idx = org_module.layer_idx

        self.head_dim = org_module.head_dim
        self.num_key_value_groups = org_module.num_key_value_groups
        self.scaling = org_module.scaling
        self.attention_dropout = org_module.attention_dropout
        self.is_causal = org_module.is_causal

        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)
        self.sliding_window = getattr(org_module, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        observation_mask = kwargs["observation_mask"]

        # gamma migration
        if self.cac_migrate:
            logger.info(
                f"the original min range is {hidden_states.min()}, the original max range is {hidden_states.max()}"
            )
            # calculate scale
            weight_list = torch.cat(
                [
                    self.q_proj.module.weight,
                    self.k_proj.module.weight,
                    self.v_proj.module.weight,
                ]
            )
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
            }
            # update scale
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
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # out migration
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
            # update scale
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)

        return attn_output, attn_weights


class QuantizedQwen2DecoderLayer(Qwen2DecoderLayer, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2DecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size

        self.self_attn = QuantizedQwen2Attention(
            org_module.self_attn,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )

        self.mlp = QuantizedQwen2MLP(
            org_module.mlp,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.input_layernorm = org_module.input_layernorm
        self.post_attention_layernorm = org_module.post_attention_layernorm
        self.attention_type = org_module.attention_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states


class QuantizedQwen2Model(Qwen2Model, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(Qwen2Model, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput
        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList(
            [QuantizedQwen2DecoderLayer(org_module.layers[i], w_qconfig, a_qconfig, qinput=True) for i in range(self.config.num_hidden_layers)])
        self.norm = org_module.norm
        self.rotary_emb = org_module.rotary_emb
        self.gradient_checkpointing = False
        self.has_sliding_layers = getattr(org_module, "has_sliding_layers", False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        assert observation_mask is not None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                observation_mask=observation_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        next_cache = past_key_values if use_cache else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class QuantizedQwen2ForCausalLM(Qwen2ForCausalLM, QuantizedModule):
    def __init__(
        self,
        org_module,
        w_qconfig,
        a_qconfig,
        qinput=True,
        backend="academic",
        is_remove_padding=False,
    ):
        super(Qwen2ForCausalLM, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self._no_split_modules = [
            "QuantizedQwen2DecoderLayer",
            "QuantizedQwen2Attention",
            "QuantizedQwen2MLP",
            "QuantizedLayer",
            "QuantizedModule",
        ]
        self.qinput = qinput
        self.vocab_size = org_module.vocab_size
        self.model = QuantizedQwen2Model(org_module.model, w_qconfig, a_qconfig, self.qinput, backend=self.backend)
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

    def is_remove_padding(self, is_remove_padding=False):
        self.is_remove_padding = is_remove_padding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.is_remove_padding and attention_mask is not None:
            observation_mask = attention_mask.clone()
        else:
            observation_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
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
