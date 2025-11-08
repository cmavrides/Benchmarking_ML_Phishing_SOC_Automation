"""Preprocess phishing datasets and generate train/val/test splits."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.text_cleaning import normalize_text, strip_html
from src.common.utils import set_random_seed
from src.task_a_benchmark.load_datasets import load_datasets

PROCESSED_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")


def _clean_text(text: str) -> str:
    return normalize_text(strip_html(text))


def preprocess_and_split(output_dir: Path = PROCESSED_DIR, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets, clean text, and create splits saved to disk."""

    set_random_seed(seed)
    df = load_datasets()
    df["clean_text"] = df["text"].map(_clean_text)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, stratify=train_val["label"], random_state=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in zip(SPLITS, (train, val, test), strict=True):
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    return train, val, test


if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_and_split()
    print({"train": len(train_df), "val": len(val_df), "test": len(test_df)})
