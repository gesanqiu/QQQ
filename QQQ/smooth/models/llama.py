"""PyTorch QuantizedLLaMA model."""

import logging
import math
import warnings
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from QQQ.smooth.migration.migration_llama import migration
from QQQ.smooth.quantization import QuantizedLayer, QuantizedModule, Quantizer

logger = logging.getLogger("QQQ")


class QuantizedLlamaMLP(LlamaMLP, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(LlamaMLP, self).__init__()
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
            self.down_proj = org_module.mlp.down_proj
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


class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(LlamaAttention, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.layer_idx = org_module.layer_idx

        if self.layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = org_module.attention_dropout
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = org_module.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.is_causal = org_module.is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        observation_mask = kwargs["observation_mask"]
        bsz, q_len, _ = hidden_states.size()

        assert position_embeddings is not None, "position_embeddings must be provided"
        cos, sin = position_embeddings

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
            extra_dict = {
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "num_key_value_groups": self.num_key_value_groups,
                "cos_cached": cos,
                "sin_cached": sin,
                "head_dim": self.head_dim,
                "position_ids": position_ids,
                "attention_mask": attention_mask[:, :, cache_position, :q_len]
                if cache_position is not None and attention_mask is not None
                else attention_mask,
                "observation_mask": observation_mask,
            }
            # update scale
            best_scale = migration(
                hidden_states,
                weight_list,
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
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

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
                self.a_qconfig,
                self.w_qconfig,
                "o_proj",
                extra_dict,
            )
            # update scale
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(LlamaDecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.self_attn = QuantizedLlamaAttention(
            org_module.self_attn,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.mlp = QuantizedLlamaMLP(
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
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37."
                " Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
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


class QuantizedLlamaModel(LlamaModel, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(LlamaModel, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput
        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            self.layers.append(QuantizedLlamaDecoderLayer(org_module.layers[i], w_qconfig, a_qconfig, qinput=True))
        self.norm = org_module.norm
        self.rotary_emb = org_module.rotary_emb
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        assert observation_mask is not None
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # embed positions
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                observation_mask=observation_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantizedLlamaForCausalLM(LlamaForCausalLM, QuantizedModule):
    def __init__(
        self,
        org_module,
        w_qconfig,
        a_qconfig,
        qinput=True,
        backend="academic",
        is_remove_padding=False,
    ):
        super(LlamaForCausalLM, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self._no_split_modules = [
            "QuantizedLlamaDecoderLayer",
            "QuantizedLlamaAttention",
            "QuantizedLlamaMLP",
            "QuantizedLayer",
            "QuantizedModule",
        ]
        self.qinput = qinput
        self.vocab_size = org_module.vocab_size
        self.model = QuantizedLlamaModel(org_module.model, w_qconfig, a_qconfig, self.qinput, backend=self.backend)
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

    def is_remove_padding(self, is_remove_padding=False):
        self.is_remove_padding = is_remove_padding

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            observation_mask=observation_mask,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
