from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml_config, update_config_from_args


def dtype_from_config(config):
    if bool(config.get("bf16", True)):
        return torch.bfloat16
    if bool(config.get("fp16", False)):
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(config, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype_from_config(config),
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_story(model, tokenizer, prompt: str, config):
    input_text = config["inference_template"].format(prompt=prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            top_k=int(config["top_k"]),
            do_sample=bool(config["do_sample"]),
            repetition_penalty=float(config["repetition_penalty"]),
            num_return_sequences=int(config["num_return_sequences"]),
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--repetition_penalty", type=float)

    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = update_config_from_args(config, args)

    model, tokenizer = load_model_and_tokenizer(config, args.adapter_path)
    story = generate_story(model, tokenizer, args.prompt, config)

    print(story)


if __name__ == "__main__":
    main()
