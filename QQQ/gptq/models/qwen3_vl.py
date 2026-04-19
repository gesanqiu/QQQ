import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from transformers.utils import logging

from QQQ.utils import find_layers

from ..gptq import *
from ..qlinear import QuantLinear
from ..quant import *

logger = logging.get_logger(__name__)


@torch.no_grad()
def gptq_qwen3_vl_func(model, dataloader, dev, args, force_to_cpu=False):
    """GPTQ quantization for Qwen3-VL, targeting only language_model layers."""
    print("Starting GPTQ quantization (Qwen3-VL, language-only) ...")

    text_config = model.config.text_config
    use_cache = text_config.use_cache
    text_config.use_cache = False
    layers = model.model.language_model.layers

    model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(dev)
    model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to(dev)
    model.model.language_model.norm = model.model.language_model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = []
    attention_mask_list = []
    position_ids_list = []
    position_embeddings_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask_list.append(kwargs.get("attention_mask"))
            position_ids_list.append(kwargs.get("position_ids"))
            position_embeddings_list.append(kwargs.get("position_embeddings"))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if isinstance(batch, dict):
                batch_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model(**batch_dev)
            else:
                model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if force_to_cpu:
        layers[0] = layers[0].cpu()
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.cpu()
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.cpu()
        model.model.language_model.norm = model.model.language_model.norm.cpu()
        torch.cuda.empty_cache()

    outs = [inp.clone() for inp in inps]

    quantizers = {}
    for i, layer in enumerate(layers):
        if layer.input_layernorm.weight.device == torch.device("cpu"):
            layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        inps = [inp.to(cur_device) for inp in inps]
        outs = [out.to(cur_device) for out in outs]
        attention_mask_list = [
            a.to(cur_device) if a is not None else None for a in attention_mask_list
        ]
        position_ids_list = [
            p.to(cur_device) if p is not None else None for p in position_ids_list
        ]
        position_embeddings_list = [
            (pe[0].to(cur_device), pe[1].to(cur_device)) if pe is not None else None
            for pe in position_embeddings_list
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
                    args.wbits, perchannel=True, sym=args.sym, mse=args.mse, groupsize=args.groupsize,
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
                    attention_mask=attention_mask_list[j],
                    position_ids=position_ids_list[j],
                    position_embeddings=position_embeddings_list[j],
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
                quantizers["model.language_model.layers.%d.%s" % (i, name)] = (
                    scale, zero, g_idx, scale_extra,
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask_list[j],
                position_ids=position_ids_list[j],
                position_embeddings=position_embeddings_list[j],
            )

        if force_to_cpu:
            layers[i] = layer.cpu()
            del layer
        else:
            layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    text_config.use_cache = use_cache
    return quantizers


class QuantizedQwen3VLTextMLP(Qwen3VLTextMLP):
    def __init__(self, config, quant_config):
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


class QuantizedQwen3VLTextAttention(Qwen3VLTextAttention):
    def __init__(self, config, quant_config, layer_idx):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(wbits, group_size, config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = QuantLinear(wbits, group_size, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = QuantLinear(wbits, group_size, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = QuantLinear(wbits, group_size, config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class QuantizedQwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config, quant_config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = QuantizedQwen3VLTextAttention(config, quant_config, layer_idx)
        self.mlp = QuantizedQwen3VLTextMLP(config, quant_config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class QuantizedQwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, config, quant_config):
        super(Qwen3VLTextModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            QuantizedQwen3VLTextDecoderLayer(config, quant_config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class QuantizedQwen3VLModel(Qwen3VLModel):
    def __init__(self, config, quant_config):
        super(Qwen3VLModel, self).__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = QuantizedQwen3VLTextModel(config.text_config, quant_config)
        self.rope_deltas = None
        self.post_init()


class QuantizedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config, quant_config):
        super(Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = QuantizedQwen3VLModel(config, quant_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
