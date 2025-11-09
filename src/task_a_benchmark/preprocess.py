"""Preprocess phishing datasets and generate train/val/test splits."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.common.text_cleaning import normalize_text, strip_html
from src.common.utils import set_random_seed
from src.task_a_benchmark.load_datasets import load_datasets

PROCESSED_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")
DEFAULT_CHUNK_SIZE = 20_000


def _clean_text(text: str) -> str:
    return normalize_text(strip_html(text))


def _clean_batch(texts: Iterable[str]) -> list[str]:
    return [_clean_text(text) for text in texts]


def _clean_text_column(series: pd.Series) -> pd.Series:
    """Clean the text column with optional multiprocessing for large datasets."""

    total_rows = len(series)
    if total_rows == 0:
        return series.copy()

    try:
        workers_env = int(os.getenv("PREPROCESS_MAX_WORKERS", "0"))
    except ValueError:
        workers_env = 0
    cpu_count = os.cpu_count() or 1
    workers = workers_env or max(1, cpu_count - 1)

    try:
        chunk_env = int(os.getenv("PREPROCESS_CHUNK_SIZE", "0"))
    except ValueError:
        chunk_env = 0
    chunk_size = chunk_env or DEFAULT_CHUNK_SIZE

    if workers <= 1 or total_rows <= chunk_size:
        return series.map(_clean_text)

    cleaned: list[str | None] = [None] * total_rows
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = series.iloc[start:end].tolist()
            future = executor.submit(_clean_batch, chunk)
            futures[future] = (start, end - start)
        with tqdm(total=len(futures), desc="Cleaning text", unit="chunk") as progress:
            for future in as_completed(futures):
                start, length = futures[future]
                chunk_cleaned = future.result()
                cleaned[start : start + length] = chunk_cleaned
                progress.update(1)

    return pd.Series(cleaned, index=series.index)


def preprocess_and_split(output_dir: Path = PROCESSED_DIR, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets, clean text, and create splits saved to disk."""

    set_random_seed(seed)
    df = load_datasets()
    df["clean_text"] = _clean_text_column(df["text"])
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
