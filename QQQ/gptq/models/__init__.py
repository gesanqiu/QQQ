from .llama import QuantizedLlamaForCausalLM, gptq_llama_func
from .qwen2 import QuantizedQwen2ForCausalLM, gptq_qwen2_func
from .qwen2_5_vl import QuantizedQwen2_5_VLForConditionalGeneration, gptq_qwen2_5_vl_func
from .qwen2_vl import QuantizedQwen2VLForConditionalGeneration, gptq_qwen2_vl_func
from .qwen3_vl import QuantizedQwen3VLForConditionalGeneration, gptq_qwen3_vl_func

_GPTQ_MODEL_FUNC = {
    "llama": gptq_llama_func,
    "qwen2": gptq_qwen2_func,
    "qwen2_vl": gptq_qwen2_vl_func,
    "qwen2_5_vl": gptq_qwen2_5_vl_func,
    "qwen3_vl": gptq_qwen3_vl_func,
}

_QUANTIZED_MODEL_CLASS = {
    "llama": QuantizedLlamaForCausalLM,
    "qwen2": QuantizedQwen2ForCausalLM,
    "qwen2_vl": QuantizedQwen2VLForConditionalGeneration,
    "qwen2_5_vl": QuantizedQwen2_5_VLForConditionalGeneration,
    "qwen3_vl": QuantizedQwen3VLForConditionalGeneration,
}


def get_gptq_model_func(model_type):
    if model_type in _GPTQ_MODEL_FUNC:
        return _GPTQ_MODEL_FUNC[model_type]
    else:
        raise NotImplementedError


def get_quantized_model_class(model_type):
    if model_type in _QUANTIZED_MODEL_CLASS:
        return _QUANTIZED_MODEL_CLASS[model_type]
    else:
        raise NotImplementedError
