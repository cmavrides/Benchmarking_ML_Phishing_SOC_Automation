"""Utilities for generating dataset splits."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _prepare_stratification(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = df["label"].to_numpy()
    sources = df.get("source_dataset", pd.Series(["unknown"] * len(df))).to_numpy()
    combined = np.char.add(labels.astype(str), "::" + sources.astype(str))
    return labels, combined


def make_stratified_splits(
    df: pd.DataFrame,
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test using stratification."""

    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    df = df.copy().reset_index(drop=True)
    labels, stratify_groups = _prepare_stratification(df)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(df, stratify_groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_fraction = val_size / (1 - test_size)
    labels_tv, stratify_tv = _prepare_stratification(train_val_df)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(splitter_val.split(train_val_df, stratify_tv))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_df.loc[:, "split"] = "train"
    val_df.loc[:, "split"] = "val"
    test_df.loc[:, "split"] = "test"

    return train_df, val_df, test_df
