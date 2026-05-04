import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    RAW_DATASET_DIR,
    TRAIN_JSONL,
    VAL_JSONL,
    SEED,
    TRAIN_SIZE_LIMIT,
    VAL_SIZE_LIMIT,
)


def build_chat_text(prompt: str, story: str) -> str:
    """
    Converts prompt + story into Qwen instruction-tuning format.

    We train the model to write a story from a prompt.
    """

    prompt = str(prompt).strip()
    story = str(story).strip()

    text = (
        "<|im_start|>system\n"
        "You are a creative writing assistant. "
        "Given a writing prompt, write a coherent, vivid, and complete story.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Write a story based on this prompt:\n\n{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{story}\n"
        "<|im_end|>"
    )

    return text


def read_lines(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def load_wp_source_target_files():
    """
    Loads the dataset if it is in .wp_source / .wp_target format.
    """

    train_source = RAW_DATASET_DIR / "train.wp_source"
    train_target = RAW_DATASET_DIR / "train.wp_target"

    valid_source = RAW_DATASET_DIR / "valid.wp_source"
    valid_target = RAW_DATASET_DIR / "valid.wp_target"

    if not train_source.exists() or not train_target.exists():
        return None, None

    print("Detected WritingPrompts source/target file format.")

    train_prompts = read_lines(train_source)
    train_stories = read_lines(train_target)

    if len(train_prompts) != len(train_stories):
        raise ValueError(
            f"Train source/target length mismatch: "
            f"{len(train_prompts)} prompts vs {len(train_stories)} stories"
        )

    train_rows = []
    for prompt, story in zip(train_prompts, train_stories):
        train_rows.append(
            {
                "prompt": prompt,
                "story": story,
                "text": build_chat_text(prompt, story),
            }
        )

    if valid_source.exists() and valid_target.exists():
        valid_prompts = read_lines(valid_source)
        valid_stories = read_lines(valid_target)

        if len(valid_prompts) != len(valid_stories):
            raise ValueError(
                f"Validation source/target length mismatch: "
                f"{len(valid_prompts)} prompts vs {len(valid_stories)} stories"
            )

        val_rows = []
        for prompt, story in zip(valid_prompts, valid_stories):
            val_rows.append(
                {
                    "prompt": prompt,
                    "story": story,
                    "text": build_chat_text(prompt, story),
                }
            )
    else:
        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=0.05,
            random_state=SEED,
        )

    return train_rows, val_rows


def load_tabular_files():
    """
    Fallback loader if the Kaggle dataset appears as CSV/JSON/JSONL.

    It tries to infer prompt/story columns.
    """

    candidate_files = list(RAW_DATASET_DIR.rglob("*.csv"))
    candidate_files += list(RAW_DATASET_DIR.rglob("*.json"))
    candidate_files += list(RAW_DATASET_DIR.rglob("*.jsonl"))

    if not candidate_files:
        return None, None

    print("Detected tabular dataset format.")
    print("Candidate files:")
    for file in candidate_files:
        print(file)

    file_path = candidate_files[0]

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".json":
        df = pd.read_json(file_path)
    elif file_path.suffix == ".jsonl":
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    print("\nColumns found:")
    print(df.columns.tolist())

    possible_prompt_cols = [
        "prompt",
        "writing_prompt",
        "source",
        "wp_source",
        "input",
        "instruction",
    ]

    possible_story_cols = [
        "story",
        "completion",
        "target",
        "wp_target",
        "output",
        "response",
        "text",
    ]

    prompt_col = None
    story_col = None

    for col in possible_prompt_cols:
        if col in df.columns:
            prompt_col = col
            break

    for col in possible_story_cols:
        if col in df.columns and col != prompt_col:
            story_col = col
            break

    if prompt_col is None or story_col is None:
        raise ValueError(
            "Could not automatically detect prompt/story columns. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[prompt_col, story_col]].dropna()
    df = df.rename(columns={prompt_col: "prompt", story_col: "story"})

    rows = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"]).strip()
        story = str(row["story"]).strip()

        if not prompt or not story:
            continue

        rows.append(
            {
                "prompt": prompt,
                "story": story,
                "text": build_chat_text(prompt, story),
            }
        )

    train_rows, val_rows = train_test_split(
        rows,
        test_size=0.05,
        random_state=SEED,
    )

    return train_rows, val_rows


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"Saving {path.name}"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset():
    random.seed(SEED)

    print("=" * 80)
    print("Preparing Writing Prompts dataset")
    print("=" * 80)

    train_rows, val_rows = load_wp_source_target_files()

    if train_rows is None:
        train_rows, val_rows = load_tabular_files()

    if train_rows is None:
        raise FileNotFoundError(
            f"Could not find supported dataset files inside {RAW_DATASET_DIR}"
        )

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    if TRAIN_SIZE_LIMIT is not None:
        train_rows = train_rows[:TRAIN_SIZE_LIMIT]

    if VAL_SIZE_LIMIT is not None:
        val_rows = val_rows[:VAL_SIZE_LIMIT]

    save_jsonl(train_rows, TRAIN_JSONL)
    save_jsonl(val_rows, VAL_JSONL)

    print("\nDataset prepared successfully.")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Train path: {TRAIN_JSONL}")
    print(f"Validation path: {VAL_JSONL}")

    print("\nExample training text:")
    print(train_rows[0]["text"][:1000])


if __name__ == "__main__":
    prepare_dataset()