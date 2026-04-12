from QQQ.smooth.models import (
    QuantizedLlamaModel,
    QuantizedQwen2Model,
    QuantizedQwen2VLTextModel,
    QuantizedQwen3VLTextModel,
)
from QQQ.smooth.quantization.observer import ObserverBase
from QQQ.utils import get_language_model, get_model_type, prepare_for_inference, set_language_model

_QUANTIZED_MODEL_MAP = {
    "llama": QuantizedLlamaModel,
    "qwen2": QuantizedQwen2Model,
    "qwen2_vl": QuantizedQwen2VLTextModel,
    "qwen3_vl": QuantizedQwen3VLTextModel,
}


def quantize_model(fp_model, config_quant, args):
    fp_model.eval()
    model_type = get_model_type(fp_model.config)
    model_cls = _QUANTIZED_MODEL_MAP[model_type]

    language_model = get_language_model(fp_model)
    quantized = model_cls(
        language_model,
        config_quant.w_qconfig,
        config_quant.a_qconfig,
        qinput=False,
    )
    set_language_model(fp_model, quantized)
    model = fp_model

    for name, module in model.named_modules():
        if isinstance(module, ObserverBase) and "act" in name:
            module.set_name(name)
    model = prepare_for_inference(model, args.device, args.dtype)
    return model
