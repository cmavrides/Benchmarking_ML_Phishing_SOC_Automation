"""Preprocessing pipeline for phishing datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_utils.hashing import stable_id
from common_utils.logging_conf import get_logger
from common_utils.text_cleaning import normalize, strip_html

from .schema import UnifiedRecord

LOGGER = get_logger(__name__)


def clean_record(
    subject: str,
    body: str,
    *,
    body_is_html: bool,
    source_dataset: str,
    source_type: str,
    label: int,
    lower: bool,
    strip_html_flag: bool,
) -> Tuple[UnifiedRecord, bool]:
    raw_body = body or ""
    was_html = body_is_html
    cleaned_body = raw_body
    html_stripped = False
    if strip_html_flag or body_is_html:
        cleaned_body, detected_html = strip_html(raw_body)
        was_html = was_html or detected_html
        html_stripped = True
    normalized_body = normalize(cleaned_body, lower=lower)
    subject_clean = normalize(subject or "", lower=lower)
    record_id = stable_id(subject_clean, normalized_body)
    record = UnifiedRecord(
        id=record_id,
        source_dataset=source_dataset,
        source_type=source_type,
        subject=subject or "",
        body=raw_body,
        body_clean=normalized_body,
        body_is_html=was_html or html_stripped,
        label=int(label),
    )
    return record, bool(normalized_body)


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    lower: bool,
    strip_html_flag: bool,
    min_length: int,
    max_length: int,
) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        record, keep = clean_record(
            subject=getattr(row, "subject", ""),
            body=getattr(row, "body", ""),
            body_is_html=bool(getattr(row, "body_is_html", False)),
            source_dataset=getattr(row, "source_dataset"),
            source_type=getattr(row, "source_type", "email"),
            label=int(getattr(row, "label")),
            lower=lower,
            strip_html_flag=strip_html_flag,
        )
        if not keep:
            continue
        if len(record.body_clean) < min_length:
            continue
        if max_length > 0 and len(record.body_clean) > max_length:
            continue
        records.append(asdict(record))
    clean_df = pd.DataFrame(records)
    if clean_df.empty:
        raise ValueError("No records remaining after preprocessing")
    clean_df = clean_df.drop_duplicates(subset="id")
    return clean_df


def split_dataset(
    df: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    stratify_col = df["label"].astype(str) + "_" + df["source_dataset"].astype(str)
    stratify_values = stratify_col.value_counts()
    if (stratify_values < 2).any():
        LOGGER.warning("Falling back to label-only stratification due to limited samples per source")
        stratify_col = df["label"].astype(str)
    if stratify_col.value_counts().min() < 2:
        LOGGER.warning("Insufficient samples for stratified split; using random split without stratification")
        stratify_arg = None
    else:
        stratify_arg = stratify_col
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=stratify_arg,
        random_state=seed,
    )
    temp_strat = temp_df["label"].astype(str) + "_" + temp_df["source_dataset"].astype(str)
    temp_counts = temp_strat.value_counts()
    if (temp_counts < 2).any():
        temp_strat = temp_df["label"].astype(str)
    stratify_temp = temp_strat if temp_strat.value_counts().min() >= 2 else None
    relative_test_size = test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=stratify_temp,
        random_state=seed,
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def save_splits(splits: Dict[str, pd.DataFrame], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        path = processed_dir / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        LOGGER.info("Saved %s split to %s (%d rows)", split_name, path, len(split_df))


def run_preprocess(
    df: pd.DataFrame,
    *,
    lower: bool,
    strip_html_flag: bool,
    min_length: int,
    max_length: int,
    val_size: float,
    test_size: float,
    seed: int,
    save_dir: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    cleaned = preprocess_dataframe(
        df,
        lower=lower,
        strip_html_flag=strip_html_flag,
        min_length=min_length,
        max_length=max_length,
    )
    splits = split_dataset(cleaned, val_size=val_size, test_size=test_size, seed=seed)
    if save_dir is not None:
        save_splits(splits, save_dir)
    return splits
