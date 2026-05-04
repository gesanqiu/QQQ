import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils import logging

from QQQ.utils import find_layers

from ..gptq import *
from ..qlinear import QuantLinear
from ..quant import *

logger = logging.get_logger(__name__)


@torch.no_grad()
def gptq_qwen3_func(model, dataloader, dev, args, force_to_cpu=False):
    print("Starting GPTQ quantization (Qwen3) ...")

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
            self.attention_type = getattr(module, "attention_type", "full_attention")

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
        attention_mask = [att_mask.to(cur_device) if att_mask is not None else None for att_mask in attention_mask]
        position_ids = [pos_ids.to(cur_device) for pos_ids in position_ids]
        position_embeddings = [
            (pe[0].to(cur_device), pe[1].to(cur_device)) if pe is not None else None for pe in position_embeddings
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


class QuantizedQwen3Attention(Qwen3Attention):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: dict,
        layer_idx: int,
    ):
        nn.Module.__init__(self)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        ab = config.attention_bias
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=ab,
        )
        self.k_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=ab,
        )
        self.v_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=ab,
        )
        self.o_proj = QuantLinear(
            wbits,
            group_size,
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=ab,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None


class QuantizedQwen3MLP(Qwen3MLP):
    def __init__(self, config: Qwen3Config, quant_config: dict):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.gate_proj = QuantLinear(wbits, group_size, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = QuantLinear(wbits, group_size, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = QuantLinear(wbits, group_size, self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class QuantizedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, quant_config: dict, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = QuantizedQwen3Attention(config, quant_config, layer_idx)

        self.mlp = QuantizedQwen3MLP(config, quant_config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types else "full_attention"


class QuantizedQwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config, quant_config: dict):
        super(Qwen3Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                QuantizedQwen3DecoderLayer(config, quant_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in getattr(config, "layer_types", [])
        self.post_init()


class QuantizedQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config, quant_config: dict):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = QuantizedQwen3Model(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
