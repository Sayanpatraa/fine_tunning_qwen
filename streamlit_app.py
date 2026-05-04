from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from datasets import load_from_disk

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.pipeline_api import (
    generate_with_loaded_model,
    load_generation_model,
    prepare_dataset_from_config,
    run_adversarial_self_optimization_from_config,
    train_sft_from_config,
)


CONFIG_PATH = BASE_DIR / "configs" / "default_config.yaml"


@st.cache_data(show_spinner=False)
def load_default_config_cached(config_mtime: float) -> dict:
    """
    Cache the default YAML config.

    The file modification time is included so Streamlit invalidates
    the cache when the config file changes.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_dataset_summary_cached(processed_data_dir: str, data_mtime_key: float | None) -> dict:
    """
    Cache lightweight dataset summary.

    Do not cache the full dataset into the UI unless needed; just keep counts.
    """
    path = Path(processed_data_dir)

    if not path.exists():
        return {
            "exists": False,
            "train": 0,
            "validation": 0,
            "test": 0,
        }

    dataset = load_from_disk(str(path))

    return {
        "exists": True,
        "train": len(dataset["train"]) if "train" in dataset else 0,
        "validation": len(dataset["validation"]) if "validation" in dataset else 0,
        "test": len(dataset["test"]) if "test" in dataset else 0,
    }


@st.cache_data(show_spinner=False)
def load_latest_metrics_cached(adversarial_output_dir: str, mtime_key: float | None) -> pd.DataFrame:
    """
    Cache adversarial metrics table.
    """
    root = Path(adversarial_output_dir)

    if not root.exists():
        return pd.DataFrame()

    rows = []

    for metrics_path in sorted(root.glob("iteration_*/iteration_metrics.json")):
        try:
            rows.append(pd.read_json(metrics_path, typ="series").to_dict())
        except Exception:
            continue

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=True)
def load_cached_generation_model(
    model_name: str,
    adapter_path: str,
    trust_remote_code: bool,
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: str,
    bnb_4bit_use_double_quant: bool,
    bf16: bool,
    fp16: bool,
):
    """
    Cache model + tokenizer for generation.

    This avoids reloading the LLM every time the user clicks Generate.
    Cache key includes model name, adapter path, and quantization settings.
    """
    config = {
        "model_name": model_name,
        "trust_remote_code": trust_remote_code,
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_quant_type": bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
        "bf16": bf16,
        "fp16": fp16,
    }

    adapter = adapter_path if adapter_path and Path(adapter_path).exists() else None

    return load_generation_model(config, adapter)


def get_mtime_or_none(path: str) -> float | None:
    p = Path(path)

    if not p.exists():
        return None

    if p.is_file():
        return p.stat().st_mtime

    mtimes = [x.stat().st_mtime for x in p.rglob("*") if x.exists()]
    return max(mtimes) if mtimes else p.stat().st_mtime


def save_run_config(config: dict) -> Path:
    path = BASE_DIR / "configs" / "streamlit_run_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return path


def make_logger():
    log_box = st.empty()
    logs: list[str] = []

    def log_fn(message: str):
        logs.append(str(message))
        log_box.text("\n".join(logs[-160:]))

    return log_fn, logs


def render_cache_controls():
    st.sidebar.header("Cache")

    st.sidebar.caption(
        "Generation models are cached so the UI does not reload the LLM on every click."
    )

    if st.sidebar.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared.")


def sidebar_model_and_paths(config: dict) -> dict:
    st.sidebar.header("Model")

    config["model_name"] = st.sidebar.text_input(
        "Hugging Face model",
        config.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
    )

    config["trust_remote_code"] = st.sidebar.checkbox(
        "trust_remote_code",
        bool(config.get("trust_remote_code", True)),
    )

    st.sidebar.header("Paths")

    config["raw_data_dir"] = st.sidebar.text_input(
        "Raw data dir",
        config.get("raw_data_dir", "data/raw"),
    )

    config["processed_data_dir"] = st.sidebar.text_input(
        "Processed data dir",
        config.get("processed_data_dir", "data/processed"),
    )

    config["output_dir"] = st.sidebar.text_input(
        "SFT / LoRA output dir",
        config.get("output_dir", "models/writing_prompts_lora"),
    )

    config["adversarial_output_dir"] = st.sidebar.text_input(
        "Adversarial output dir",
        config.get("adversarial_output_dir", "models/adversarial_runs"),
    )

    config["preference_data_dir"] = st.sidebar.text_input(
        "Preference data dir",
        config.get("preference_data_dir", "data/preferences"),
    )

    return config


def sidebar_dataset_limits(config: dict) -> dict:
    st.sidebar.header("Dataset Limits")

    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    max_test = config.get("max_test_samples")

    use_train_limit = st.sidebar.checkbox("Limit train samples", max_train is not None)
    config["max_train_samples"] = (
        st.sidebar.number_input("Max train samples", min_value=1, value=int(max_train or 5000), step=500)
        if use_train_limit
        else None
    )

    use_eval_limit = st.sidebar.checkbox("Limit eval samples", max_eval is not None)
    config["max_eval_samples"] = (
        st.sidebar.number_input("Max eval samples", min_value=1, value=int(max_eval or 1000), step=100)
        if use_eval_limit
        else None
    )

    use_test_limit = st.sidebar.checkbox("Limit test samples", max_test is not None)
    config["max_test_samples"] = (
        st.sidebar.number_input("Max test samples", min_value=1, value=int(max_test or 1000), step=100)
        if use_test_limit
        else None
    )

    return config


def training_controls(config: dict) -> dict:
    col1, col2, col3 = st.columns(3)

    with col1:
        config["num_train_epochs"] = st.number_input("Epochs", min_value=0.1, value=float(config.get("num_train_epochs", 1)), step=0.5)
        config["learning_rate"] = st.number_input("Learning rate", min_value=1e-7, value=float(config.get("learning_rate", 0.0002)), format="%.7f")
        config["warmup_ratio"] = st.number_input("Warmup ratio", min_value=0.0, max_value=1.0, value=float(config.get("warmup_ratio", 0.03)), step=0.01)
        config["weight_decay"] = st.number_input("Weight decay", min_value=0.0, value=float(config.get("weight_decay", 0.0)), step=0.001)

    with col2:
        config["per_device_train_batch_size"] = st.number_input("Train batch size", min_value=1, value=int(config.get("per_device_train_batch_size", 1)))
        config["per_device_eval_batch_size"] = st.number_input("Eval batch size", min_value=1, value=int(config.get("per_device_eval_batch_size", 1)))
        config["gradient_accumulation_steps"] = st.number_input("Gradient accumulation", min_value=1, value=int(config.get("gradient_accumulation_steps", 16)))
        config["max_seq_length"] = st.number_input("Max sequence length", min_value=256, value=int(config.get("max_seq_length", 2048)), step=256)

    with col3:
        config["lora_r"] = st.number_input("LoRA rank r", min_value=1, value=int(config.get("lora_r", 32)), step=1)
        config["lora_alpha"] = st.number_input("LoRA alpha", min_value=1, value=int(config.get("lora_alpha", 64)), step=1)
        config["lora_dropout"] = st.number_input("LoRA dropout", min_value=0.0, max_value=0.9, value=float(config.get("lora_dropout", 0.05)), step=0.01)
        config["load_in_4bit"] = st.checkbox("Load in 4-bit", bool(config.get("load_in_4bit", True)))

    target_modules_raw = st.text_input(
        "Target modules comma-separated",
        ",".join(config.get("target_modules", [])),
    )

    config["target_modules"] = [x.strip() for x in target_modules_raw.split(",") if x.strip()]

    col4, col5, col6 = st.columns(3)

    with col4:
        config["gradient_checkpointing"] = st.checkbox("Gradient checkpointing", bool(config.get("gradient_checkpointing", True)))
        config["packing"] = st.checkbox("Packing", bool(config.get("packing", True)))

    with col5:
        config["bf16"] = st.checkbox("bf16", bool(config.get("bf16", True)))
        config["fp16"] = st.checkbox("fp16", bool(config.get("fp16", False)))

    with col6:
        config["logging_steps"] = st.number_input("Logging steps", min_value=1, value=int(config.get("logging_steps", 25)))
        config["eval_steps"] = st.number_input("Eval steps", min_value=1, value=int(config.get("eval_steps", 500)))
        config["save_steps"] = st.number_input("Save steps", min_value=1, value=int(config.get("save_steps", 500)))

    return config


def adversarial_controls(config: dict) -> dict:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        config["self_optimize_iterations"] = st.number_input(
            "Iterations",
            min_value=1,
            value=int(config.get("self_optimize_iterations", 3)),
            step=1,
        )

    with col_b:
        config["prompts_per_iteration"] = st.number_input(
            "Prompts per iteration",
            min_value=1,
            value=int(config.get("prompts_per_iteration", 128)),
            step=16,
        )

    with col_c:
        config["candidates_per_prompt"] = st.number_input(
            "Candidates per prompt",
            min_value=2,
            value=int(config.get("candidates_per_prompt", 4)),
            step=1,
        )

    st.markdown("### Critic weights")

    cw1, cw2, cw3, cw4, cw5 = st.columns(5)

    with cw1:
        config["critic_weight_prompt_relevance"] = st.number_input(
            "Prompt relevance",
            min_value=0.0,
            value=float(config.get("critic_weight_prompt_relevance", 0.30)),
            step=0.05,
        )

    with cw2:
        config["critic_weight_diversity"] = st.number_input(
            "Diversity",
            min_value=0.0,
            value=float(config.get("critic_weight_diversity", 0.20)),
            step=0.05,
        )

    with cw3:
        config["critic_weight_repetition_penalty"] = st.number_input(
            "Repetition control",
            min_value=0.0,
            value=float(config.get("critic_weight_repetition_penalty", 0.20)),
            step=0.05,
        )

    with cw4:
        config["critic_weight_length"] = st.number_input(
            "Length",
            min_value=0.0,
            value=float(config.get("critic_weight_length", 0.15)),
            step=0.05,
        )

    with cw5:
        config["critic_weight_story_shape"] = st.number_input(
            "Story shape",
            min_value=0.0,
            value=float(config.get("critic_weight_story_shape", 0.15)),
            step=0.05,
        )

    st.markdown("### DPO parameters")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        config["dpo_beta"] = st.number_input("DPO beta", min_value=0.001, value=float(config.get("dpo_beta", 0.1)), step=0.01)

    with d2:
        config["dpo_learning_rate"] = st.number_input("DPO learning rate", min_value=1e-7, value=float(config.get("dpo_learning_rate", 0.00005)), format="%.7f")

    with d3:
        config["dpo_gradient_accumulation_steps"] = st.number_input("DPO gradient accumulation", min_value=1, value=int(config.get("dpo_gradient_accumulation_steps", 8)), step=1)

    with d4:
        config["dpo_num_train_epochs"] = st.number_input("DPO epochs", min_value=0.1, value=float(config.get("dpo_num_train_epochs", 1)), step=0.5)

    return config


def generation_controls(config: dict) -> dict:
    col1, col2, col3 = st.columns(3)

    with col1:
        config["max_new_tokens"] = st.number_input("Max new tokens", min_value=10, value=int(config.get("max_new_tokens", 700)), step=50)
        config["temperature"] = st.slider("Temperature", 0.0, 2.0, float(config.get("temperature", 0.8)), 0.05)

    with col2:
        config["top_p"] = st.slider("Top-p", 0.0, 1.0, float(config.get("top_p", 0.9)), 0.01)
        config["top_k"] = st.number_input("Top-k", min_value=1, value=int(config.get("top_k", 50)))

    with col3:
        config["repetition_penalty"] = st.number_input("Repetition penalty", min_value=1.0, value=float(config.get("repetition_penalty", 1.08)), step=0.01)
        config["do_sample"] = st.checkbox("Sample", bool(config.get("do_sample", True)))

    config["num_return_sequences"] = 1

    return config


def main():
    st.set_page_config(page_title="Writing Prompts LLM Fine-Tuning", layout="wide")

    st.title("Writing Prompts LLM Fine-Tuning")
    st.caption("Integrated Streamlit pipeline: dataset prep, SFT, adversarial DPO self-optimization, and cached generation.")

    config_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
    config = load_default_config_cached(config_mtime)

    render_cache_controls()
    config = sidebar_model_and_paths(config)
    config = sidebar_dataset_limits(config)

    dataset_mtime = get_mtime_or_none(config["processed_data_dir"])
    dataset_summary = load_dataset_summary_cached(config["processed_data_dir"], dataset_mtime)

    with st.sidebar.expander("Dataset status", expanded=True):
        if dataset_summary["exists"]:
            st.success("Processed dataset found.")
            st.write(dataset_summary)
        else:
            st.warning("Processed dataset not found.")

    tab_prepare, tab_train, tab_adversarial, tab_generate, tab_metrics, tab_config = st.tabs(
        [
            "Prepare Dataset",
            "Train SFT",
            "Adversarial Self-Optimization",
            "Generate / Test",
            "Metrics",
            "Config Preview",
        ]
    )

    with tab_prepare:
        st.subheader("Prepare prompt-story dataset")

        st.write(
            "This runs inside Streamlit directly. It does not shell out to a CLI script."
        )

        if st.button("Prepare dataset"):
            save_run_config(config)
            log_fn, _ = make_logger()

            with st.spinner("Preparing dataset..."):
                prepare_dataset_from_config(config, log_fn=log_fn)

            st.cache_data.clear()
            st.success("Dataset preparation complete.")

    with tab_train:
        st.subheader("Supervised fine-tuning")
        config = training_controls(config)

        if st.button("Start SFT training"):
            save_run_config(config)
            log_fn, _ = make_logger()

            with st.spinner("Training SFT adapter..."):
                adapter_path = train_sft_from_config(config, log_fn=log_fn)

            st.cache_resource.clear()
            st.success(f"SFT complete. Adapter saved to: {adapter_path}")

    with tab_adversarial:
        st.subheader("Adversarial Self-Optimization")

        st.write(
            "This generates multiple stories per prompt, scores them, creates chosen/rejected pairs, "
            "then applies DPO. It runs directly through Python functions inside Streamlit."
        )

        start_adapter_path = st.text_input(
            "Start adapter path",
            config.get("output_dir", "models/writing_prompts_lora"),
        )

        config = adversarial_controls(config)

        if st.button("Start adversarial self-optimization"):
            save_run_config(config)
            log_fn, _ = make_logger()
            progress = st.progress(0)

            def progress_fn(value: float):
                progress.progress(min(1.0, max(0.0, value)))

            with st.spinner("Running adversarial self-optimization..."):
                final_adapter = run_adversarial_self_optimization_from_config(
                    config=config,
                    start_adapter_path=start_adapter_path,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                )

            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(f"Self-optimization complete. Final adapter: {final_adapter}")

    with tab_generate:
        st.subheader("Generate story")

        adapter_path = st.text_input(
            "Adapter path",
            config.get("output_dir", "models/writing_prompts_lora"),
        )

        config = generation_controls(config)

        prompt = st.text_area(
            "Prompt",
            "A lonely astronaut discovers that the moon has been writing letters to Earth for centuries.",
            height=120,
        )

        use_cached_model = st.checkbox("Use cached model for generation", value=True)

        if st.button("Generate story"):
            save_run_config(config)

            with st.spinner("Generating..."):
                if use_cached_model:
                    model, tokenizer = load_cached_generation_model(
                        model_name=config["model_name"],
                        adapter_path=adapter_path,
                        trust_remote_code=bool(config.get("trust_remote_code", True)),
                        load_in_4bit=bool(config.get("load_in_4bit", True)),
                        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_compute_dtype=config.get("bnb_4bit_compute_dtype", "bfloat16"),
                        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
                        bf16=bool(config.get("bf16", True)),
                        fp16=bool(config.get("fp16", False)),
                    )
                    story = generate_with_loaded_model(model, tokenizer, config, prompt)
                else:
                    from src.pipeline_api import generate_story_from_config

                    story = generate_story_from_config(
                        config=config,
                        prompt=prompt,
                        adapter_path=adapter_path,
                    )

            st.markdown("### Generated story")
            st.write(story)

    with tab_metrics:
        st.subheader("Adversarial run metrics")

        metrics_mtime = get_mtime_or_none(config.get("adversarial_output_dir", "models/adversarial_runs"))
        metrics_df = load_latest_metrics_cached(config.get("adversarial_output_dir", "models/adversarial_runs"), metrics_mtime)

        if metrics_df.empty:
            st.info("No adversarial metrics found yet.")
        else:
            st.dataframe(metrics_df, use_container_width=True)

            if "mean_critic_score" in metrics_df.columns:
                st.line_chart(metrics_df.set_index("iteration")["mean_critic_score"])

            if "mean_score_margin" in metrics_df.columns:
                st.line_chart(metrics_df.set_index("iteration")["mean_score_margin"])

    with tab_config:
        st.subheader("Current config")

        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

        st.download_button(
            "Download current config",
            yaml.safe_dump(config, sort_keys=False),
            file_name="streamlit_run_config.yaml",
            mime="text/yaml",
        )


if __name__ == "__main__":
    main()
