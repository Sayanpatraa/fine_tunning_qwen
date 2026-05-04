from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--merged_model_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    merged_dir = Path(args.merged_model_dir or config["merged_model_dir"])
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
