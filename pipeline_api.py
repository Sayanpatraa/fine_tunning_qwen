from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer, SFTTrainer

from src.adversarial_utils import (
    build_preference_pairs,
    extract_assistant_completion,
    save_preference_dataset,
    score_story,
)
from src.config_utils import save_config_snapshot


LogFn = Optional[Callable[[str], None]]


def log_message(log_fn: LogFn, message: str) -> None:
    if log_fn is not None:
        log_fn(message)
    else:
        print(message)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f.readlines()]


def build_split(
    source_path: Path,
    target_path: Path,
    prompt_template: str,
    max_samples: int | None = None,
) -> Dataset:
    prompts = read_lines(source_path)
    stories = read_lines(target_path)

    n = min(len(prompts), len(stories))
    prompts = prompts[:n]
    stories = stories[:n]

    if max_samples is not None:
        prompts = prompts[:max_samples]
        stories = stories[:max_samples]

    rows = []

    for prompt, story in zip(prompts, stories):
        if not prompt or not story:
            continue

        rows.append(
            {
                "prompt": prompt,
                "story": story,
                "text": prompt_template.format(prompt=prompt, story=story),
            }
        )

    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def prepare_dataset_from_config(config: dict, log_fn: LogFn = None) -> DatasetDict:
    raw_dir = Path(config["raw_data_dir"])
    processed_dir = Path(config["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    log_message(log_fn, f"Reading raw dataset from: {raw_dir}")

    dataset = DatasetDict(
        {
            "train": build_split(
                raw_dir / config["train_source_file"],
                raw_dir / config["train_target_file"],
                config["prompt_template"],
                config.get("max_train_samples"),
            ),
            "validation": build_split(
                raw_dir / config["valid_source_file"],
                raw_dir / config["valid_target_file"],
                config.get("prompt_template"),
                config.get("max_eval_samples"),
            ),
            "test": build_split(
                raw_dir / config["test_source_file"],
                raw_dir / config["test_target_file"],
                config.get("prompt_template"),
                config.get("max_test_samples"),
            ),
        }
    )

    dataset.save_to_disk(str(processed_dir))

    log_message(log_fn, f"Saved processed dataset to: {processed_dir}")
    log_message(log_fn, str(dataset))

    return dataset


def dtype_from_string(name: str):
    name = str(name).lower()

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def dtype_from_config(config: dict):
    if bool(config.get("bf16", True)):
        return torch.bfloat16
    if bool(config.get("fp16", False)):
        return torch.float16
    return torch.float32


def subset_dataset(dataset, max_samples):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), int(max_samples))))


def make_quantization_config(config: dict):
    if not bool(config.get("load_in_4bit", True)):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_string(
            config.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=bool(
            config.get("bnb_4bit_use_double_quant", True)
        ),
    )


def load_tokenizer(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def make_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )


def train_sft_from_config(config: dict, log_fn: LogFn = None) -> str:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)

    log_message(log_fn, "Loading processed dataset...")
    dataset = load_from_disk(config["processed_data_dir"])

    train_dataset = subset_dataset(dataset["train"], config.get("max_train_samples"))
    eval_dataset = subset_dataset(dataset["validation"], config.get("max_eval_samples"))

    log_message(log_fn, f"Train samples: {len(train_dataset)}")
    log_message(log_fn, f"Eval samples: {len(eval_dataset)}")

    tokenizer = load_tokenizer(config)

    log_message(log_fn, f"Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=make_quantization_config(config),
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    model.config.use_cache = False

    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_config = make_lora_config(config)

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

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=training_args,
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

    log_message(log_fn, "Starting SFT training...")
    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_message(log_fn, f"Saved LoRA adapter to: {output_dir}")

    del trainer
    del model
    cleanup_cuda()

    return str(output_dir)


def load_generation_model(config: dict, adapter_path: str | None = None):
    tokenizer = load_tokenizer(config)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=make_quantization_config(config),
        torch_dtype=None if bool(config.get("load_in_4bit", True)) else dtype_from_config(config),
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_with_loaded_model(model, tokenizer, config: dict, prompt: str) -> str:
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
    return extract_assistant_completion(decoded)


def generate_story_from_config(
    config: dict,
    prompt: str,
    adapter_path: str | None = None,
    log_fn: LogFn = None,
) -> str:
    log_message(log_fn, "Loading model for generation...")
    model, tokenizer = load_generation_model(config, adapter_path)
    completion = generate_with_loaded_model(model, tokenizer, config, prompt)

    del model
    cleanup_cuda()

    return completion


def sample_prompts(config: dict, n_prompts: int, seed: int) -> list[str]:
    dataset = load_from_disk(config["processed_data_dir"])
    train_ds = dataset["train"]

    rng = random.Random(seed)
    idxs = list(range(len(train_ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_prompts, len(idxs))]

    return [train_ds[i]["prompt"] for i in idxs]


def generate_candidates_with_loaded_model(
    model,
    tokenizer,
    prompt: str,
    config: dict,
    n_candidates: int,
) -> list[str]:
    input_text = config["inference_template"].format(prompt=prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    completions = []

    for _ in range(n_candidates):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(config.get("max_new_tokens", 700)),
                temperature=float(config.get("temperature", 0.8)),
                top_p=float(config.get("top_p", 0.9)),
                top_k=int(config.get("top_k", 50)),
                do_sample=bool(config.get("do_sample", True)),
                repetition_penalty=float(config.get("repetition_penalty", 1.08)),
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        completions.append(extract_assistant_completion(decoded))

    return completions


def load_policy_model_for_dpo(config: dict, adapter_path: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=make_quantization_config(config),
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    model.config.use_cache = False

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        return model, None

    return model, make_lora_config(config)


def train_dpo_from_preference_dir(
    config: dict,
    preference_data_dir: str,
    output_dir: str,
    adapter_path: str | None = None,
    log_fn: LogFn = None,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config)
    model, peft_config = load_policy_model_for_dpo(config, adapter_path)
    preference_dataset = load_from_disk(preference_data_dir)

    training_args = TrainingArguments(
        output_dir=str(output_path),
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
            output_dir=str(output_path),
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

    log_message(log_fn, "Starting DPO update...")
    trainer.train()

    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    log_message(log_fn, f"Saved DPO adapter to: {output_path}")

    del trainer
    del model
    cleanup_cuda()

    return str(output_path)


def run_adversarial_self_optimization_from_config(
    config: dict,
    start_adapter_path: str | None = None,
    log_fn: LogFn = None,
    progress_fn: Optional[Callable[[float], None]] = None,
) -> str:
    iterations = int(config.get("self_optimize_iterations", 3))
    prompts_per_iteration = int(config.get("prompts_per_iteration", 128))
    candidates_per_prompt = int(config.get("candidates_per_prompt", 4))

    run_root = Path(config.get("adversarial_output_dir", "models/adversarial_runs"))
    pref_root = Path(config.get("preference_data_dir", "data/preferences"))

    run_root.mkdir(parents=True, exist_ok=True)
    pref_root.mkdir(parents=True, exist_ok=True)

    current_adapter = start_adapter_path or config.get("output_dir")

    total_steps = iterations * prompts_per_iteration
    completed_steps = 0

    for iteration in range(1, iterations + 1):
        log_message(log_fn, f"========== Iteration {iteration}/{iterations} ==========")
        log_message(log_fn, f"Current adapter: {current_adapter}")

        model, tokenizer = load_generation_model(config, current_adapter)

        prompts = sample_prompts(
            config=config,
            n_prompts=prompts_per_iteration,
            seed=int(config.get("seed", 42)) + iteration,
        )

        scored_rows = []

        for prompt_id, prompt in enumerate(prompts):
            completions = generate_candidates_with_loaded_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
                n_candidates=candidates_per_prompt,
            )

            for candidate_id, completion in enumerate(completions):
                critic = score_story(prompt, completion, config)

                scored_rows.append(
                    {
                        "iteration": iteration,
                        "prompt_id": prompt_id,
                        "candidate_id": candidate_id,
                        "prompt": prompt,
                        "completion": completion,
                        "critic_total": critic.total,
                        "prompt_relevance": critic.prompt_relevance,
                        "diversity": critic.diversity,
                        "repetition": critic.repetition,
                        "length_score": critic.length_score,
                        "story_shape": critic.story_shape,
                    }
                )

            completed_steps += 1
            if progress_fn is not None:
                progress_fn(completed_steps / max(1, total_steps))

            if (prompt_id + 1) % 10 == 0 or prompt_id == len(prompts) - 1:
                log_message(log_fn, f"Generated/scored {prompt_id + 1}/{len(prompts)} prompts")

        del model
        cleanup_cuda()

        iter_dir = run_root / f"iteration_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        scored_df = pd.DataFrame(scored_rows)
        scored_csv = iter_dir / "scored_candidates.csv"
        scored_df.to_csv(scored_csv, index=False)

        pair_df = build_preference_pairs(scored_rows)
        pair_csv = iter_dir / "preference_pairs.csv"
        pair_df.to_csv(pair_csv, index=False)

        pref_dir = pref_root / f"iteration_{iteration:02d}"
        save_preference_dataset(pair_df, str(pref_dir))

        metrics = {
            "iteration": iteration,
            "num_scored_candidates": int(len(scored_df)),
            "num_preference_pairs": int(len(pair_df)),
            "mean_critic_score": float(scored_df["critic_total"].mean()),
            "mean_score_margin": float(pair_df["score_margin"].mean()) if len(pair_df) else 0.0,
            "current_adapter_before_update": current_adapter,
        }

        with (iter_dir / "iteration_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        log_message(log_fn, f"Preference pairs saved to: {pair_csv}")
        log_message(log_fn, f"Mean critic score: {metrics['mean_critic_score']:.4f}")
        log_message(log_fn, f"Mean score margin: {metrics['mean_score_margin']:.4f}")

        next_adapter = str(iter_dir / "dpo_adapter")

        current_adapter = train_dpo_from_preference_dir(
            config=config,
            preference_data_dir=str(pref_dir),
            output_dir=next_adapter,
            adapter_path=current_adapter,
            log_fn=log_fn,
        )

    final_path = run_root / "FINAL_ADAPTER_PATH.txt"
    final_path.write_text(str(current_adapter), encoding="utf-8")

    log_message(log_fn, f"Self-optimization complete. Final adapter: {current_adapter}")

    return str(current_adapter)
