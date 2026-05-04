"""Migration helpers for Qwen3 (dense LM): q/k/v projections plus head-dim q_norm/k_norm before RoPE."""

import logging

import torch

from .migration_qwen2 import (
    Migrator1DRangeSearch,
    Migrator1DRangeSearchAWQ,
    Migrator1DRangeSearchSQ,
    MigratorBase,
)

logger = logging.getLogger("QQQ")
scale_list = []
search_class = None


def _rms_norm_heads(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm on last dim; x shape (batch, seq, num_heads, head_dim)."""
    input_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return (weight * x_f.to(input_dtype))


def set_search_class(smooth_method):
    class_map = {
        "os+": Qwen3Migrator1DRangeSearch,
        "awq": Qwen3Migrator1DRangeSearchAWQ,
        "sq": Qwen3Migrator1DRangeSearchSQ,
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


class Qwen3MigratorMixin:
    """Matches Qwen3Attention: proj -> reshape -> q_norm/k_norm on head_dim -> RoPE -> SDPA."""

    def qkv_function(self, input, weight, bias=None):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        B, N, C = input.shape
        head_dim = self.extra_dict["head_dim"]
        qkv = torch.matmul(input, weight.T)
        if bias is not None:
            qkv = qkv + bias
        sz_q = self.extra_dict["num_heads"] * head_dim
        sz_kv = self.extra_dict["num_key_value_heads"] * head_dim
        q = qkv[:, :, :sz_q].view(B, N, self.extra_dict["num_heads"], head_dim)
        k = qkv[:, :, sz_q : sz_q + sz_kv].view(B, N, self.extra_dict["num_key_value_heads"], head_dim)
        v = qkv[:, :, sz_q + sz_kv :].view(B, N, self.extra_dict["num_key_value_heads"], head_dim)

        eps = float(self.extra_dict["rms_norm_eps"])
        q = _rms_norm_heads(q, self.extra_dict["q_norm_weight"], eps)
        k = _rms_norm_heads(k, self.extra_dict["k_norm_weight"], eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.extra_dict["position_embeddings"]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.extra_dict["num_key_value_groups"])
        v = repeat_kv(v, self.extra_dict["num_key_value_groups"])
        if q.device.type == "cuda" and self.extra_dict["attention_mask"] is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.extra_dict["attention_mask"],
            dropout_p=0.0,
            is_causal=self.extra_dict["attention_mask"] is None and N > 1,
        )
        output = output.transpose(1, 2).contiguous().view(B, N, C)
        return output[self.extra_dict["observation_mask"] == 1].to(torch.float32)


class Qwen3Migrator1DRangeSearch(Qwen3MigratorMixin, Migrator1DRangeSearch):
    pass


class Qwen3Migrator1DRangeSearchAWQ(Qwen3MigratorMixin, Migrator1DRangeSearchAWQ):
    pass


class Qwen3Migrator1DRangeSearchSQ(Qwen3MigratorMixin, Migrator1DRangeSearchSQ):
    pass
