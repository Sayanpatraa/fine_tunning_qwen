from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
import sys

import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.adversarial_utils import (
    build_preference_pairs,
    extract_assistant_completion,
    save_preference_dataset,
    score_story,
)
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


def load_generator(config: dict, adapter_path: str | None):
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

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, prompt: str, config: dict, n_candidates: int) -> list[str]:
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
        completion = extract_assistant_completion(decoded)
        completions.append(completion)

    return completions


def sample_prompts(config: dict, n_prompts: int, seed: int) -> list[str]:
    dataset = load_from_disk(config["processed_data_dir"])
    train_ds = dataset["train"]

    rng = random.Random(seed)
    idxs = list(range(len(train_ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_prompts, len(idxs))]

    return [train_ds[i]["prompt"] for i in idxs]


def run_dpo_update(config_path: str, preference_dir: str, adapter_path: str | None, output_dir: str):
    command = [
        sys.executable,
        "scripts/train_dpo.py",
        "--config",
        config_path,
        "--preference_data_dir",
        preference_dir,
        "--output_dir",
        output_dir,
    ]

    if adapter_path:
        command.extend(["--adapter_path", adapter_path])

    print("Running DPO update:")
    print(" ".join(command))

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--start_adapter_path", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--prompts_per_iteration", type=int, default=None)
    parser.add_argument("--candidates_per_prompt", type=int, default=None)

    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = update_config_from_args(config, args)

    iterations = int(args.iterations or config.get("self_optimize_iterations", 3))
    prompts_per_iteration = int(args.prompts_per_iteration or config.get("prompts_per_iteration", 128))
    candidates_per_prompt = int(args.candidates_per_prompt or config.get("candidates_per_prompt", 4))

    run_root = Path(config.get("adversarial_output_dir", "models/adversarial_runs"))
    pref_root = Path(config.get("preference_data_dir", "data/preferences"))
    run_root.mkdir(parents=True, exist_ok=True)
    pref_root.mkdir(parents=True, exist_ok=True)

    current_adapter = args.start_adapter_path or config.get("output_dir")

    for iteration in range(1, iterations + 1):
        print(f"\n========== ITERATION {iteration}/{iterations} ==========")

        model, tokenizer = load_generator(config, current_adapter)

        prompts = sample_prompts(
            config=config,
            n_prompts=prompts_per_iteration,
            seed=int(config.get("seed", 42)) + iteration,
        )

        scored_rows = []

        for prompt_id, prompt in enumerate(prompts):
            completions = generate_candidates(
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

            if (prompt_id + 1) % 10 == 0:
                print(f"Generated and scored {prompt_id + 1}/{len(prompts)} prompts")

        # Free generation model before DPO to reduce VRAM pressure.
        del model
        torch.cuda.empty_cache()

        scored_df = pd.DataFrame(scored_rows)
        iter_dir = run_root / f"iteration_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

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

        next_adapter = str(iter_dir / "dpo_adapter")

        run_dpo_update(
            config_path=args.config,
            preference_dir=str(pref_dir),
            adapter_path=current_adapter,
            output_dir=next_adapter,
        )

        current_adapter = next_adapter

        print(f"Iteration {iteration} complete. New adapter: {current_adapter}")

    final_path = run_root / "FINAL_ADAPTER_PATH.txt"
    final_path.write_text(str(current_adapter), encoding="utf-8")

    print("\nSelf-optimization complete.")
    print(f"Final adapter path: {current_adapter}")


if __name__ == "__main__":
    main()
