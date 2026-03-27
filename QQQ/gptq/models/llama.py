import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from typing import Optional
from ..qlinear import QuantLinear
from ..gptq import *
from ..quant import *
from QQQ.utils import find_layers
from transformers.utils import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def gptq_llama_func(model, dataloader, dev, args, force_to_cpu=False):
    print("Starting GPTQ quantization ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = []
    attention_mask = []
    position_ids = []
    position_embeddings = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs.get("attention_mask"))
            position_ids.append(kwargs.get("position_ids"))
            position_embeddings.append(kwargs.get("position_embeddings"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if force_to_cpu:
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

    outs = [inp.clone() for inp in inps]

    quantizers = {}
    for i, layer in enumerate(layers):
        if layer.input_layernorm.weight.device == torch.device("cpu"):
            layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        inps = [inp.to(cur_device) for inp in inps]
        outs = [out.to(cur_device) for out in outs]
        attention_mask = [
            att_mask.to(cur_device) if att_mask is not None else None
            for att_mask in attention_mask
        ]
        position_ids = [pos_ids.to(cur_device) for pos_ids in position_ids]
        position_embeddings = [
            (pe[0].to(cur_device), pe[1].to(cur_device)) if pe is not None else None
            for pe in position_embeddings
        ]

        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits,
                    perchannel=True,
                    sym=args.sym,
                    mse=args.mse,
                    groupsize=args.groupsize,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                    position_embeddings=position_embeddings[j],
                )
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                scale, zero, g_idx, scale_extra = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = (
                    scale,
                    zero,
                    g_idx,
                    scale_extra,
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask[j],
                position_ids=position_ids[j],
                position_embeddings=position_embeddings[j],
            )

        if force_to_cpu:
            layers[i] = layer.cpu()
            del layer
        else:
            layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


class QuantizedLlamaAttention(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: dict,
        layer_idx: int,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = QuantLinear(
            wbits,
            group_size,
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )


class QuantizedLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=mlp_bias
        )
        self.up_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=mlp_bias
        )
        self.down_proj = QuantLinear(
            wbits, group_size, self.intermediate_size, self.hidden_size, bias=mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, quant_config: dict, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = QuantizedLlamaAttention(
            config=config, quant_config=quant_config, layer_idx=layer_idx
        )
        self.mlp = QuantizedLlamaMLP(config, quant_config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class QuantizedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                QuantizedLlamaDecoderLayer(config, quant_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class QuantizedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, quant_config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = QuantizedLlamaModel(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
