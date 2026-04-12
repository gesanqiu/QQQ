import argparse

import torch
from transformers import AutoProcessor, AutoTokenizer

from QQQ.gptq.models import get_quantized_model_class
from QQQ.utils import (
    get_model_type,
    get_model_config,
    is_vlm,
    setup_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path contains model weight and quant config",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path contains tokenizer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="You are right, But Genshin Impact is",
        help="prompts",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Image path for VLM models",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    config = get_model_config(args.model_path)
    quant_config = config.quantization_config
    del config.quantization_config
    quant_model_class = get_quantized_model_class(get_model_type(config))
    model = quant_model_class.from_pretrained(
        args.model_path,
        config=config,
        quant_config=quant_config,
        device_map="sequential",
        dtype=torch.float16,
    )

    if is_vlm(config):
        processor = AutoProcessor.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        messages = [{"role": "user", "content": []}]
        if args.image:
            messages[0]["content"].append({"type": "image", "image": args.image})
        messages[0]["content"].append({"type": "text", "text": args.prompt})
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        output_ids_trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
        outputs = processor.batch_decode(output_ids_trimmed, skip_special_tokens=True)
        print(outputs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            args.prompt,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(outputs)
