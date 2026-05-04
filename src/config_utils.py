from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return data


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if str(value).strip().lower() in {"none", "null", ""}:
        return None
    return int(value)


def parse_target_modules(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            updated[key] = value

    if "target_modules" in updated:
        updated["target_modules"] = parse_target_modules(updated["target_modules"])

    return updated


def save_config_snapshot(config: Dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / "training_config.yaml"
    json_path = output_dir / "training_config.json"

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trust_remote_code", type=str_to_bool)

    parser.add_argument("--raw_data_dir", type=str)
    parser.add_argument("--processed_data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--merged_model_dir", type=str)

    parser.add_argument("--max_train_samples", type=parse_optional_int)
    parser.add_argument("--max_eval_samples", type=parse_optional_int)
    parser.add_argument("--max_test_samples", type=parse_optional_int)

    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--packing", type=str_to_bool)

    parser.add_argument("--load_in_4bit", type=str_to_bool)
    parser.add_argument("--bnb_4bit_quant_type", type=str)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str)
    parser.add_argument("--bnb_4bit_use_double_quant", type=str_to_bool)

    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--lora_bias", type=str)
    parser.add_argument("--target_modules", type=str)

    parser.add_argument("--num_train_epochs", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--report_to", type=str)
    parser.add_argument("--bf16", type=str_to_bool)
    parser.add_argument("--fp16", type=str_to_bool)

    return parser
