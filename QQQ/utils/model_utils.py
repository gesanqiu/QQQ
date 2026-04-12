import functools

import torch
import torch.nn as nn
import transformers
from accelerate.big_modeling import (
    dispatch_model,
    get_balanced_memory,
    infer_auto_device_map,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)

from .utils import str2torch_device, str2torch_dtype

_SUPPORTED_MODEL_TYPES = {"llama", "qwen2", "qwen2_vl", "qwen3_vl"}


def is_vlm(config):
    """Detect VLM by checking for vision_config in the model config."""
    return hasattr(config, "vision_config") and config.vision_config is not None


def build_model_and_tokenizer(model_path, tokenizer_path, dtype: str, trust_remote_code: bool = True):
    model_path = model_path.rstrip("/")
    tokenizer_path = tokenizer_path.rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {
        "dtype": str2torch_dtype(dtype),
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code, **kwargs)
    return model, tokenizer


def build_vlm_and_processor(model_path, processor_path, dtype: str, trust_remote_code: bool = True):
    model_path = model_path.rstrip("/")
    processor_path = processor_path.rstrip("/")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=trust_remote_code)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    kwargs = {
        "dtype": str2torch_dtype(dtype),
        "device_map": "auto",
    }
    model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=trust_remote_code, **kwargs)
    return model, processor


def get_model_type(config):
    model_type = getattr(config, "model_type", None)
    if model_type in _SUPPORTED_MODEL_TYPES:
        return model_type
    raise ValueError(
        f"Model type '{model_type}' is not supported. "
        f"Supported types: {_SUPPORTED_MODEL_TYPES}"
    )


def prepare_for_inference(model, device, dtype):
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    model.to(str2torch_dtype(dtype))
    if device == "cuda" and torch.cuda.device_count() > 1:
        max_memory = get_balanced_memory(
            model,
            no_split_module_classes=model._no_split_modules,
            dtype=str2torch_dtype(dtype),
        )
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory,
            dtype=str2torch_dtype(dtype),
        )
        print(device_map)
        dispatch_model(model, device_map=device_map)
    else:
        model.to(str2torch_device(device))
    model.eval()
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def get_model_config(model_path: str, trust_remote_code: bool = True, revision: str | None = None) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if not trust_remote_code and "requires you to execute the configuration file" in str(e):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return config


def get_language_model(model):
    """Extract the language model backbone from any supported HuggingFace model.

    VLMs: model.model.language_model (decoder stack under the vision-language wrapper)
    LLMs: model.model (decoder stack directly)
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model
    if hasattr(inner, "backbone"):
        return inner.backbone
    return inner


def set_language_model(model, new_module):
    """Replace the language model backbone in-place."""
    inner = model.model
    if hasattr(inner, "language_model"):
        inner.language_model = new_module
    elif hasattr(inner, "backbone"):
        inner.backbone = new_module
    else:
        model.model = new_module


def get_language_config(model):
    """Get the config for the language model part.

    For VLMs with a separate text_config, returns that; otherwise the top-level config.
    """
    config = model.config
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


def get_transformer_layers(model, model_type=None):
    return list(get_language_model(model).layers)


def get_lm_head(model, model_type=None):
    return model.lm_head


def get_pre_head_layernorm(model, model_type=None):
    # NOTE(HandH1998): only support RMSnorm
    return get_language_model(model).norm


def get_embeddings(model, model_type=None) -> list[torch.nn.Module]:
    return [get_language_model(model).embed_tokens]


def remove_empty_parameters(model):
    state_dict = {}
    for k, v in model.state_dict().items():
        if v.numel() > 0:
            state_dict[k] = v
    return state_dict
