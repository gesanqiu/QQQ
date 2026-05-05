# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
import logging
import typing

import torch
import tqdm

logger = logging.getLogger(__name__)

from QQQ.utils import (
    free_memory,
    get_embeddings,
    get_language_config,
    get_lm_head,
    get_pre_head_layernorm,
    get_transformer_layers,
    str2torch_device,
)

from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix


def duplicate_lm_head_if_tied(model) -> None:
    """
    QuaRot applies the orthogonal map separately to the embedding table and to the output projection,
    so we need to duplicate the lm_head from the embeddings if tie_word_embeddings is True.
    """
    lang_config = get_language_config(model)
    if not getattr(lang_config, "tie_word_embeddings", False):
        return
    embeddings = get_embeddings(model)
    if not embeddings:
        logger.warning("tie_word_embeddings True but no embedding modules found; skipping lm_head duplicate")
        return
    lm_head = get_lm_head(model)
    emb = embeddings[0]
    if lm_head.weight.shape != emb.weight.shape:
        logger.warning(
            "lm_head %s vs embedding %s shape mismatch; skipping duplicate",
            lm_head.weight.shape,
            emb.weight.shape,
        )
        return

    lm_head.weight = torch.nn.Parameter(emb.weight.detach().clone())

    lang_config.tie_word_embeddings = False
    root_cfg = model.config
    root_cfg.tie_word_embeddings = False
    text_cfg = getattr(root_cfg, "text_config", None)
    if text_cfg is not None:
        text_cfg.tie_word_embeddings = False

    logger.info(
        "Duplicated lm_head from embeddings and set tie_word_embeddings=False"
    )


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def reset_ln(ln):
    W_norm = ln.weight.data
    ln.weight.data = torch.ones_like(W_norm)


def fuse_layer_norms(model):
    layers = get_transformer_layers(model)

    for layer in layers:
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    fuse_ln_linear(get_pre_head_layernorm(model), [get_lm_head(model)])
    reset_ln(get_pre_head_layernorm(model))
    return model


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, Q, device) -> None:
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, device) -> None:
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, device) -> None:
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, device):
    for W in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, device):
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q, device) -> None:
    W = get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


@torch.inference_mode()
def rotate_model(model, rotation_config, args, Q=None):
    device = str2torch_device(args.device)
    lang_config = get_language_config(model)
    num_heads = lang_config.num_attention_heads
    model_dim = lang_config.hidden_size
    head_dim = getattr(lang_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_dim // num_heads

    Q = get_orthogonal_matrix(model_dim, rotation_config.rotate_mode, device) if Q is None else Q
    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device)
    free_memory()
    layers = get_transformer_layers(model)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
        rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q
