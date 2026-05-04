import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR


def load_finetuned_model():
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
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


def generate_story(prompt: str, max_new_tokens: int = 700):
    model, tokenizer = load_finetuned_model()

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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return output_text.strip()


if __name__ == "__main__":
    import torch

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    from src.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR


    def load_finetuned_model():
        """
        Loads the base Qwen model plus the trained LoRA adapter.

        Important:
        - We use float16, not bfloat16, because your GPU setup does not support bf16.
        - LORA_OUTPUT_DIR must contain adapter_config.json and adapter_model.safetensors.
        """

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available. This 4-bit LoRA inference script expects a GPU."
            )

        compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("=" * 80)
        print("Loading tokenizer")
        print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            LORA_OUTPUT_DIR,
            trust_remote_code=True,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("=" * 80)
        print("Loading base model")
        print("=" * 80)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print("=" * 80)
        print("Loading LoRA adapter")
        print("=" * 80)

        model = PeftModel.from_pretrained(
            base_model,
            LORA_OUTPUT_DIR,
        )

        model.eval()

        return model, tokenizer


    def generate_story(prompt: str, max_new_tokens: int = 700):
        model, tokenizer = load_finetuned_model()

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

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        output_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return output_text.strip()


    if __name__ == "__main__":
        test_prompt = (
            "A lonely astronaut receives a handwritten letter from Earth, "
            "but Earth disappeared three years ago."
        )

        story = generate_story(test_prompt)

        print("=" * 80)
        print("Generated Story")
        print("=" * 80)
        print(story)