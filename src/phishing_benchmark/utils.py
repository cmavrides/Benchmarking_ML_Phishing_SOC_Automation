"""Utility helpers shared across the project."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def dump_json(data: Any, path: Path) -> None:
    """Write JSON data to disk with UTF-8 encoding."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """Load JSON content from disk."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha1_hash(text: str) -> str:
    """Return a SHA-1 hash for the provided text."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available)."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - optional GPU path
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch optional
        pass


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a dataframe from parquet or CSV depending on suffix."""

    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe as parquet or CSV based on suffix."""

    ensure_dir(path.parent)
    if path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def to_serializable(data: Any) -> Any:
    """Convert dataclasses and other objects to JSON-serializable structures."""

    if is_dataclass(data):
        return {k: to_serializable(v) for k, v in asdict(data).items()}
    if isinstance(data, dict):
        return {k: to_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [to_serializable(v) for v in data]
    return data


def save_leaderboard_row(
    path: Path, row: Dict[str, Any], order: Optional[Iterable[str]] = None
) -> None:
    """Append or create a leaderboard CSV with consistent columns."""

    ensure_dir(path.parent)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    if order:
        for column in order:
            if column not in df.columns:
                df[column] = np.nan
        df = df[[col for col in order if col in df.columns] + [c for c in df.columns if c not in order]]

    df = df.drop_duplicates(subset=["model", "run_id"], keep="last", ignore_index=True)
    df.to_csv(path, index=False)
