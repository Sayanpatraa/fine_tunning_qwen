from pathlib import Path


# Project paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# Model configuration


BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # upgraded: A10G has headroom

LORA_OUTPUT_DIR = MODELS_DIR / "writing_prompts_qwen_lora"


# Dataset paths


RAW_DATASET_DIR = RAW_DIR / "writing-prompts"
TRAIN_JSONL = PROCESSED_DIR / "train.jsonl"
VAL_JSONL = PROCESSED_DIR / "val.jsonl"


# Training configuration


SEED = 42

MAX_SEQ_LENGTH = 1024

TRAIN_SIZE_LIMIT =60000
VAL_SIZE_LIMIT = 1000

NUM_TRAIN_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 16 # effective batch size = 32

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"

LOGGING_STEPS = 50
EVAL_STEPS = 500   # less frequent eval = more time training
SAVE_STEPS = 500

SAVE_TOTAL_LIMIT = 2


BF16 = True   # A10G supports bf16 natively
FP16 = False