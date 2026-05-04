from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import (
    add_common_training_args,
    load_yaml_config,
    save_config_snapshot,
    update_config_from_args,
)


def dtype_from_string(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def subset_dataset(dataset, max_samples):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), int(max_samples))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser = add_common_training_args(parser)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = update_config_from_args(config, args)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)

    dataset = load_from_disk(config["processed_data_dir"])
    train_dataset = subset_dataset(dataset["train"], config.get("max_train_samples"))
    eval_dataset = subset_dataset(dataset["validation"], config.get("max_eval_samples"))

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        logging_steps=int(config["logging_steps"]),
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        seed=int(config["seed"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        optim=config["optim"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to=config["report_to"],
        bf16=bool(config["bf16"]),
        fp16=bool(config["fp16"]),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    # TRL changed its SFTTrainer API across versions.
    # This block keeps the script usable across common installations.
    try:
        trainer = SFTTrainer(
            **trainer_kwargs,
            dataset_text_field="text",
            max_seq_length=int(config["max_seq_length"]),
            packing=bool(config["packing"]),
        )
    except TypeError:
        from trl import SFTConfig

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=int(config["max_seq_length"]),
            packing=bool(config["packing"]),
            num_train_epochs=float(config["num_train_epochs"]),
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            learning_rate=float(config["learning_rate"]),
            warmup_ratio=float(config["warmup_ratio"]),
            weight_decay=float(config["weight_decay"]),
            max_grad_norm=float(config["max_grad_norm"]),
            logging_steps=int(config["logging_steps"]),
            eval_steps=int(config["eval_steps"]),
            save_steps=int(config["save_steps"]),
            save_total_limit=int(config["save_total_limit"]),
            seed=int(config["seed"]),
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
            optim=config["optim"],
            lr_scheduler_type=config["lr_scheduler_type"],
            report_to=config["report_to"],
            bf16=bool(config["bf16"]),
            fp16=bool(config["fp16"]),
            eval_strategy="steps",
            save_strategy="steps",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=sft_config,
        )

    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
