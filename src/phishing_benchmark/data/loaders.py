"""Load datasets from local storage or Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from phishing_benchmark.utils import ensure_dir, read_dataframe, write_dataframe

try:  # pragma: no cover - optional dependency during testing
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - handle missing dependency
    load_dataset = None  # type: ignore[assignment]
    _LOAD_DATASET_ERROR = exc
else:
    _LOAD_DATASET_ERROR = None


_RAW_FILENAMES = {
    "zefang_liu": "zefang_liu.parquet",
    "cyradar": "cyradar.parquet",
}


def _require_datasets() -> None:
    if load_dataset is None:  # pragma: no cover - executed when datasets missing
        raise RuntimeError(
            "datasets library is required to download from Hugging Face"
        ) from _LOAD_DATASET_ERROR


def _standardize_columns(
    df: pd.DataFrame,
    *,
    source_dataset: str,
    source_type: str,
    default_subject: str = "",
) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    else:
        df["id"] = df.index.astype(str)
    subject_col = None
    for candidate in ("subject", "title", "headline"):
        if candidate in df.columns:
            subject_col = candidate
            break
    if subject_col:
        df["subject"] = df[subject_col].astype(str)
    else:
        df["subject"] = default_subject

    body_col = None
    for candidate in ("body", "text", "email_text", "content", "message"):
        if candidate in df.columns:
            body_col = candidate
            break
    if body_col is None:
        raise ValueError(f"Could not find body text column for {source_dataset}")

    df["body"] = df[body_col].astype(str)

    if "body_is_html" in df.columns:
        df["body_is_html"] = df["body_is_html"].astype(bool)
    elif "is_html" in df.columns:
        df["body_is_html"] = df["is_html"].astype(bool)
    else:
        df["body_is_html"] = False

    label_col = None
    for candidate in ("label", "labels", "target"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Could not find label column for {source_dataset}")

    raw_labels = df[label_col]
    if raw_labels.dtype.kind in {"U", "S", "O"}:
        normalized = raw_labels.str.lower().map({"phishing": 1, "legitimate": 0, "spam": 1})
        if normalized.isna().any():
            raise ValueError(f"Unexpected string labels for {source_dataset}: {set(raw_labels)}")
        df["label"] = normalized.astype(int)
    else:
        df["label"] = raw_labels.astype(int)

    df["source_dataset"] = source_dataset
    df["source_type"] = source_type
    df["split"] = None
    return df[
        [
            "id",
            "source_dataset",
            "source_type",
            "subject",
            "body",
            "body_is_html",
            "label",
            "split",
        ]
    ]


def _finalize(df: pd.DataFrame, *, raw_dir: Path, dataset_name: str, sample: Optional[int]) -> pd.DataFrame:
    if sample is not None and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=0).reset_index(drop=True)
    path = raw_dir / _RAW_FILENAMES[dataset_name]
    write_dataframe(df, path)
    return df


def load_zefang_liu(raw_dir: Path, *, sample: Optional[int] = None, force: bool = False) -> pd.DataFrame:
    """Load or download the zefang-liu phishing email dataset."""

    ensure_dir(raw_dir)
    path = raw_dir / _RAW_FILENAMES["zefang_liu"]
    if path.exists() and not force:
        return read_dataframe(path)

    _require_datasets()
    dataset = load_dataset("zefang-liu/phishing-email-dataset")
    df = dataset["train"].to_pandas()
    df = _standardize_columns(df, source_dataset="zefang-liu", source_type="email")
    return _finalize(df, raw_dir=raw_dir, dataset_name="zefang_liu", sample=sample)


def load_cyradar(raw_dir: Path, *, sample: Optional[int] = None, force: bool = False) -> pd.DataFrame:
    """Load or download the Cyradar phishing dataset."""

    ensure_dir(raw_dir)
    path = raw_dir / _RAW_FILENAMES["cyradar"]
    if path.exists() and not force:
        return read_dataframe(path)

    _require_datasets()
    dataset = load_dataset("huynq3Cyradar/Phishing_Detection_Dataset")
    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        # Some community datasets expose a single split named "default"
        first_split = next(iter(dataset.keys()))
        df = dataset[first_split].to_pandas()
    df = _standardize_columns(df, source_dataset="cyradar", source_type="text")
    return _finalize(df, raw_dir=raw_dir, dataset_name="cyradar", sample=sample)
