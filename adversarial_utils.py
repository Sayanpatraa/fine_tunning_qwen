from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
from datasets import Dataset


_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class CriticScore:
    total: float
    prompt_relevance: float
    diversity: float
    repetition: float
    length_score: float
    story_shape: float


def simple_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = set(a)
    b_set = set(b)

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def repetition_ratio(words: list[str], n: int = 3) -> float:
    """
    Estimate how much the story repeats itself.

    Lower is better. The score is based on repeated n-grams.
    """
    if len(words) < n * 2:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)

    repeated = sum(count - 1 for count in counts.values() if count > 1)

    return repeated / max(1, len(ngrams))


def lexical_diversity(words: list[str]) -> float:
    """
    Type-token ratio with a length correction.
    """
    if not words:
        return 0.0

    unique_ratio = len(set(words)) / len(words)

    # Very short outputs can fake high diversity.
    length_factor = clamp(len(words) / 250.0)

    return clamp(unique_ratio * 1.8 * length_factor)


def length_quality(words: list[str], min_words: int, max_words: int) -> float:
    """
    Reward stories that are long enough but not uncontrolled.
    """
    n_words = len(words)

    if n_words < min_words:
        return clamp(n_words / max(1, min_words))

    if n_words <= max_words:
        return 1.0

    excess = n_words - max_words
    return clamp(1.0 - excess / max(1, max_words))


def story_shape_score(text: str) -> float:
    """
    Approximate narrative structure.

    This is not literary truth. It is a cheap critic that rewards:
    - paragraphing,
    - sentence count,
    - dialogue or scene detail,
    - ending punctuation.
    """
    text = text.strip()

    if not text:
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    sentence_score = clamp(len(sentences) / 12.0)
    paragraph_score = clamp(len(paragraphs) / 4.0)

    has_dialogue = 1.0 if '"' in text or "'" in text else 0.0
    has_ending = 1.0 if text[-1] in ".!?" else 0.0

    return clamp(
        0.40 * sentence_score
        + 0.25 * paragraph_score
        + 0.20 * has_dialogue
        + 0.15 * has_ending
    )


def score_story(prompt: str, story: str, config: Dict) -> CriticScore:
    """
    Score one generated story.

    The critic is deliberately modular. You can replace this with:
    - a reward model,
    - an LLM judge,
    - BLEU/ROUGE-style metrics,
    - human preference labels,
    - a classifier trained on high-quality stories.
    """
    prompt_words = simple_words(prompt)
    story_words = simple_words(story)

    prompt_relevance = jaccard_overlap(prompt_words, story_words)
    prompt_relevance = clamp(prompt_relevance * 4.0)

    diversity = lexical_diversity(story_words)

    rep_ratio = repetition_ratio(story_words)
    max_rep = float(config.get("max_repetition_ratio", 0.18))
    repetition = clamp(1.0 - rep_ratio / max_rep)

    length_score = length_quality(
        story_words,
        int(config.get("min_story_words", 120)),
        int(config.get("max_story_words", 900)),
    )

    shape = story_shape_score(story)

    total = (
        float(config.get("critic_weight_prompt_relevance", 0.30)) * prompt_relevance
        + float(config.get("critic_weight_diversity", 0.20)) * diversity
        + float(config.get("critic_weight_repetition_penalty", 0.20)) * repetition
        + float(config.get("critic_weight_length", 0.15)) * length_score
        + float(config.get("critic_weight_story_shape", 0.15)) * shape
    )

    return CriticScore(
        total=float(total),
        prompt_relevance=float(prompt_relevance),
        diversity=float(diversity),
        repetition=float(repetition),
        length_score=float(length_score),
        story_shape=float(shape),
    )


def extract_assistant_completion(decoded_text: str) -> str:
    """
    Try to extract only the assistant response from decoded generation.
    """
    markers = ["<|assistant|>", "Assistant:", "assistant"]
    text = decoded_text

    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1]

    return text.strip()


def build_preference_pairs(scored_rows: list[dict]) -> pd.DataFrame:
    """
    Convert generated candidates into DPO preference pairs.

    For each prompt:
    - highest-scoring candidate becomes chosen,
    - lowest-scoring candidate becomes rejected.
    """
    df = pd.DataFrame(scored_rows)

    if df.empty:
        raise ValueError("No scored rows were provided.")

    rows = []

    for prompt, group in df.groupby("prompt"):
        if len(group) < 2:
            continue

        group = group.sort_values("critic_total", ascending=False)

        best = group.iloc[0]
        worst = group.iloc[-1]

        if best["completion"].strip() == worst["completion"].strip():
            continue

        rows.append(
            {
                "prompt": str(prompt),
                "chosen": str(best["completion"]),
                "rejected": str(worst["completion"]),
                "chosen_score": float(best["critic_total"]),
                "rejected_score": float(worst["critic_total"]),
                "score_margin": float(best["critic_total"] - worst["critic_total"]),
            }
        )

    return pd.DataFrame(rows)


def save_preference_dataset(pair_df: pd.DataFrame, output_dir: str) -> Dataset:
    """
    Save preference pairs in Hugging Face dataset format.
    """
    if pair_df.empty:
        raise ValueError("Preference dataframe is empty. Cannot train DPO.")

    dataset = Dataset.from_pandas(pair_df, preserve_index=False)
    dataset.save_to_disk(output_dir)

    return dataset
