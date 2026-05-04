from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml_config, update_config_from_args


def dtype_from_string(name: str):
    name = str(name).lower()

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def load_policy_model(config: dict, adapter_path: str | None):
    quantization_config = None

    if bool(config.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype_from_string(config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    model.config.use_cache = False

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        return model, None

    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )

    return model, peft_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--preference_data_dir", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--dpo_beta", type=float)
    parser.add_argument("--dpo_num_train_epochs", type=float)
    parser.add_argument("--dpo_learning_rate", type=float)
    parser.add_argument("--dpo_per_device_train_batch_size", type=int)
    parser.add_argument("--dpo_gradient_accumulation_steps", type=int)

    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = update_config_from_args(config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, peft_config = load_policy_model(config, args.adapter_path)

    preference_dataset = load_from_disk(args.preference_data_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("dpo_num_train_epochs", 1)),
        per_device_train_batch_size=int(config.get("dpo_per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("dpo_gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("dpo_learning_rate", 5e-5)),
        logging_steps=int(config.get("dpo_logging_steps", 25)),
        save_steps=int(config.get("dpo_save_steps", 250)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        optim=config.get("optim", "paged_adamw_8bit"),
    )

    # TRL has changed DPOTrainer signatures across versions.
    # This supports both older and newer installations.
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            beta=float(config.get("dpo_beta", 0.1)),
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=int(config.get("dpo_max_prompt_length", 512)),
            max_length=int(config.get("dpo_max_length", 2048)),
        )
    except TypeError:
        from trl import DPOConfig

        dpo_args = DPOConfig(
            output_dir=str(output_dir),
            beta=float(config.get("dpo_beta", 0.1)),
            num_train_epochs=float(config.get("dpo_num_train_epochs", 1)),
            per_device_train_batch_size=int(config.get("dpo_per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(config.get("dpo_gradient_accumulation_steps", 8)),
            learning_rate=float(config.get("dpo_learning_rate", 5e-5)),
            logging_steps=int(config.get("dpo_logging_steps", 25)),
            save_steps=int(config.get("dpo_save_steps", 250)),
            save_total_limit=int(config.get("save_total_limit", 2)),
            bf16=bool(config.get("bf16", True)),
            fp16=bool(config.get("fp16", False)),
            report_to=config.get("report_to", "none"),
            remove_unused_columns=False,
            optim=config.get("optim", "paged_adamw_8bit"),
            max_prompt_length=int(config.get("dpo_max_prompt_length", 512)),
            max_length=int(config.get("dpo_max_length", 2048)),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=preference_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"DPO-updated adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
