"""Migration module for Qwen2-VL and Qwen3-VL.

Inherits scale search from migration_qwen2 but overrides `qkv_function`
to use `apply_multimodal_rotary_pos_emb` (3-D RoPE sections) instead of
the standard 1-D `apply_rotary_pos_emb`.  Also handles the Qwen3-VL case
where q/k/v have no biases.
"""

import logging

import torch
import torch.nn as nn

from ..quantization.observer import MinMaxObserver
from ..quantization.quant_utils import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
)
from .migration_qwen2 import (
    Migrator1DRangeSearch,
    Migrator1DRangeSearchAWQ,
    Migrator1DRangeSearchSQ,
    MigratorBase,
)

logger = logging.getLogger("QQQ")
scale_list = []
search_class = None


def set_search_class(smooth_method):
    class_map = {
        "os+": VLMMigrator1DRangeSearch,
        "awq": VLMMigrator1DRangeSearchAWQ,
        "sq": VLMMigrator1DRangeSearchSQ,
    }
    global search_class
    search_class = class_map[smooth_method]


def migration(act, weight, bias, a_qconfig, w_qconfig, module_type, extra_dict=None):
    if search_class is None:
        raise ValueError("search_class need to be set before migration!")
    migrator = search_class(act, weight, bias, a_qconfig, w_qconfig, module_type, extra_dict)
    best_scale = migrator()
    scale_list.append(best_scale)
    return best_scale


class VLMMigratorMixin:
    """Overrides qkv_function for VLM multimodal rotary position embeddings."""

    def qkv_function(self, input, weight, bias=None):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            apply_multimodal_rotary_pos_emb,
        )
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        B, N, C = input.shape
        head_dim = self.extra_dict["head_dim"]
        qkv = torch.matmul(input, weight.T)
        if bias is not None:
            qkv = qkv + bias
        sz_q = self.extra_dict["num_heads"] * head_dim
        sz_kv = self.extra_dict["num_key_value_heads"] * head_dim
        q = qkv[:, :, :sz_q].view(B, N, self.extra_dict["num_heads"], head_dim).transpose(1, 2)
        k = qkv[:, :, sz_q: sz_q + sz_kv].view(B, N, self.extra_dict["num_key_value_heads"], head_dim).transpose(1, 2)
        v = qkv[:, :, sz_q + sz_kv:].view(B, N, self.extra_dict["num_key_value_heads"], head_dim).transpose(1, 2)

        cos, sin = self.extra_dict["position_embeddings"]
        mrope_section = self.extra_dict.get("mrope_section")
        if mrope_section is not None:
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        else:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.extra_dict["num_key_value_groups"])
        v = repeat_kv(v, self.extra_dict["num_key_value_groups"])

        if q.device.type == "cuda" and self.extra_dict["attention_mask"] is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.extra_dict["attention_mask"],
            dropout_p=0.0,
            is_causal=self.extra_dict["attention_mask"] is None and N > 1,
        )
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        obs_mask = self.extra_dict.get("observation_mask")
        if obs_mask is not None:
            return output[obs_mask == 1].to(torch.float32)
        return output.to(torch.float32)


class VLMMigrator1DRangeSearch(VLMMigratorMixin, Migrator1DRangeSearch):
    pass


class VLMMigrator1DRangeSearchAWQ(VLMMigratorMixin, Migrator1DRangeSearchAWQ):
    pass


class VLMMigrator1DRangeSearchSQ(VLMMigratorMixin, Migrator1DRangeSearchSQ):
    pass
