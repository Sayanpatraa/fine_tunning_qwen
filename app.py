import time
from pathlib import Path

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR


st.set_page_config(
    page_title="Qwen Story Generator",
    layout="wide",
)


def adapter_exists(adapter_dir):
    adapter_dir = Path(adapter_dir)

    required_files = [
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
    ]

    return all(path.exists() for path in required_files)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the fine-tuned model once.

    Final model = base Qwen model + your trained LoRA adapter.
    """

    if not adapter_exists(LORA_OUTPUT_DIR):
        raise FileNotFoundError(
            f"LoRA adapter not found at {LORA_OUTPUT_DIR}. "
            "Finish training first or point LORA_OUTPUT_DIR to a checkpoint folder."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This app expects GPU inference."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_OUTPUT_DIR,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_OUTPUT_DIR,
    )

    model.eval()

    return model, tokenizer


def generate_story(
    prompt,
    model,
    tokenizer,
    max_new_tokens=700,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.08,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative writing assistant. "
                "Given a writing prompt, write a coherent, vivid, and complete story."
            ),
        },
        {
            "role": "user",
            "content": f"Write a story based on this prompt:\n\n{prompt}",
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    ).to(model.device)

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    story = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return story.strip(), elapsed


st.title("Qwen Fine-Tuned Story Generator")

st.write(
    "Enter a writing prompt. The app will use your fine-tuned Qwen LoRA model "
    "to generate a story."
)

with st.sidebar:
    st.header("Model")

    st.write("Base model:")
    st.code(BASE_MODEL_NAME)

    st.write("LoRA adapter:")
    st.code(str(LORA_OUTPUT_DIR))

    if torch.cuda.is_available():
        st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        st.error("No GPU detected")

    st.header("Generation settings")

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=100,
        max_value=1500,
        value=700,
        step=50,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.05,
    )

    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    repetition_penalty = st.slider(
        "Repetition penalty",
        min_value=1.0,
        max_value=1.5,
        value=1.08,
        step=0.01,
    )

    if st.button("Clear loaded model cache"):
        st.cache_resource.clear()
        st.success("Cache cleared. Refresh the app.")


default_prompt = (
    "A lonely astronaut receives a handwritten letter from Earth, "
    "but Earth disappeared three years ago."
)

prompt = st.text_area(
    "Writing prompt",
    value=default_prompt,
    height=160,
)

if st.button("Generate Story", type="primary"):
    if not prompt.strip():
        st.error("Please enter a writing prompt.")
    else:
        try:
            with st.spinner("Loading model and generating story..."):
                model, tokenizer = load_model_and_tokenizer()

                story, elapsed = generate_story(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            st.success(f"Story generated in {elapsed:.2f} seconds")

            st.subheader("Generated Story")
            st.write(story)

        except Exception as e:
            st.error("Something failed.")
            st.exception(e)