from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config_utils import load_yaml_config, update_config_from_args


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

        formatted_text = prompt_template.format(prompt=prompt, story=story)

        rows.append(
            {
                "prompt": prompt,
                "story": story,
                "text": formatted_text,
            }
        )

    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--raw_data_dir", type=str)
    parser.add_argument("--processed_data_dir", type=str)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)

    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = update_config_from_args(config, args)

    raw_dir = Path(config["raw_data_dir"])
    processed_dir = Path(config["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = config["prompt_template"]

    train_ds = build_split(
        raw_dir / config["train_source_file"],
        raw_dir / config["train_target_file"],
        prompt_template,
        config.get("max_train_samples"),
    )

    valid_ds = build_split(
        raw_dir / config["valid_source_file"],
        raw_dir / config["valid_target_file"],
        prompt_template,
        config.get("max_eval_samples"),
    )

    test_ds = build_split(
        raw_dir / config["test_source_file"],
        raw_dir / config["test_target_file"],
        prompt_template,
        config.get("max_test_samples"),
    )

    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": valid_ds,
            "test": test_ds,
        }
    )

    dataset.save_to_disk(str(processed_dir))

    print(dataset)
    print(f"Saved processed dataset to: {processed_dir}")


if __name__ == "__main__":
    main()
