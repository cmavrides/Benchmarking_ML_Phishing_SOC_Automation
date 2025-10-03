"""Loaders for raw phishing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from common_utils.io import read_csv
from common_utils.logging_conf import get_logger

LOGGER = get_logger(__name__)


def _looks_like_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return "</" in text or "<html" in text.lower() or "<body" in text.lower()


def _normalise_label(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    mapping = {
        "phishing": 1,
        "spam": 1,
        "scam": 1,
        "legitimate": 0,
        "ham": 0,
        "0": 0,
        "1": 1,
    }
    if text in mapping:
        return mapping[text]
    raise ValueError(f"Unrecognised label value: {value}")


def _select_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Could not find any of columns {candidates} in dataframe columns {df.columns.tolist()}")


def load_zefang(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    subject_col = None
    for cand in ["subject", "email_subject"]:
        if cand in df.columns:
            subject_col = cand
            break
    if subject_col is None:
        df["subject"] = ""
        subject_col = "subject"

    body_col = _select_column(df, ["body", "text", "content", "email_text", "message"])
    label_col = _select_column(df, ["label", "category", "class"])

    df = df.rename(columns={subject_col: "subject", body_col: "body", label_col: "label"})
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["body_is_html"] = df["body"].apply(_looks_like_html)
    df["label"] = df["label"].apply(_normalise_label)
    df["source_dataset"] = "zefang_liu"
    df["source_type"] = "email"
    return df[["subject", "body", "body_is_html", "label", "source_dataset", "source_type"]]


def load_cyradar(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    subject_col = None
    for cand in ["subject", "title"]:
        if cand in df.columns:
            subject_col = cand
            break
    if subject_col is None:
        df["subject"] = ""
        subject_col = "subject"

    body_col = _select_column(df, ["body", "text", "content", "email", "message"])
    label_col = _select_column(df, ["label", "y", "class", "category"])

    df = df.rename(columns={subject_col: "subject", body_col: "body", label_col: "label"})
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["body_is_html"] = df["body"].apply(_looks_like_html)
    df["label"] = df["label"].apply(_normalise_label)
    df["source_dataset"] = "cyradar"
    df["source_type"] = "email"
    return df[["subject", "body", "body_is_html", "label", "source_dataset", "source_type"]]


LOADERS = {
    "zefang_liu": load_zefang,
    "cyradar": load_cyradar,
}


def load_datasets(raw_dir: Path, use_zefang: bool = True, use_cyradar: bool = True) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if use_zefang:
        path = raw_dir / "zefang_liu.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset at {path}")
        LOGGER.info("Loading Zefang dataset from %s", path)
        frames.append(load_zefang(path))
    if use_cyradar:
        path = raw_dir / "cyradar.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset at {path}")
        LOGGER.info("Loading CyRadar dataset from %s", path)
        frames.append(load_cyradar(path))
    if not frames:
        raise ValueError("No datasets selected for loading")
    combined = pd.concat(frames, ignore_index=True)
    return combined
