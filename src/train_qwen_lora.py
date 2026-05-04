import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import (
    BASE_MODEL_NAME,
    LORA_OUTPUT_DIR,
    TRAIN_JSONL,
    VAL_JSONL,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LR_SCHEDULER_TYPE,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    FP16,
    BF16,
    SEED,
)


def get_torch_dtype():
    if BF16 and torch.cuda.is_available():
        return torch.bfloat16
    if FP16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer():
    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model():
    print("=" * 80)
    print("Loading Qwen model")
    print("=" * 80)

    compute_dtype = get_torch_dtype()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    return model


def build_lora_config():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return lora_config


def load_training_dataset():
    print("=" * 80)
    print("Loading processed JSONL dataset")
    print("=" * 80)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_JSONL),
            "validation": str(VAL_JSONL),
        },
    )

    print("Original dataset:")
    print(dataset)
    print("Original train columns:", dataset["train"].column_names)

    # IMPORTANT:
    # Newer TRL versions automatically look for prompt/completion columns.
    # Our dataset already has the final formatted training text in the "text" column.
    # So we remove prompt/story to stop TRL from treating this as prompt-completion data.
    columns_to_remove = [
        col for col in dataset["train"].column_names
        if col != "text"
    ]

    dataset = dataset.remove_columns(columns_to_remove)

    print("Cleaned dataset:")
    print(dataset)
    print("Cleaned train columns:", dataset["train"].column_names)

    return dataset


def build_sft_config():
    """
    New TRL API:
    Use SFTConfig instead of passing dataset_text_field and max_seq_length
    directly to SFTTrainer.
    """

    sft_config = SFTConfig(
        output_dir=str(LORA_OUTPUT_DIR),

        # Dataset/text settings
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,

        # Training duration
        num_train_epochs=NUM_TRAIN_EPOCHS,

        # Batch settings
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer and LR scheduler
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim="paged_adamw_8bit",

        # Logging/eval/save
        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision
        bf16=BF16,
        fp16=FP16,

        # Stability
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # No wandb/tensorboard
        report_to="none",

        remove_unused_columns=False,
    )

    return sft_config


def train_qwen_lora():
    print("=" * 80)
    print("Starting Qwen LoRA fine-tuning")
    print("=" * 80)

    tokenizer = load_tokenizer()
    model = load_model()
    dataset = load_training_dataset()
    lora_config = build_lora_config()
    sft_config = build_sft_config()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("=" * 80)
    print("Trainable parameters")
    print("=" * 80)
    trainer.model.print_trainable_parameters()

    print("=" * 80)
    print("Training")
    print("=" * 80)

    trainer.train()

    print("=" * 80)
    print("Saving final LoRA adapter")
    print("=" * 80)

    trainer.model.save_pretrained(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

    print(f"Saved LoRA adapter to: {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    train_qwen_lora()