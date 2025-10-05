"""Common helper utilities for serialization and reproducibility."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def save_pickle(obj: Any, path: os.PathLike[str] | str) -> None:
    """Serialize an object with ``joblib``.

    Args:
        obj: Object to serialize.
        path: Destination file path.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: os.PathLike[str] | str) -> Any:
    """Load a serialized object from disk."""

    return joblib.load(path)


def hash_text(text: str) -> str:
    """Create a deterministic SHA256 hash for the provided text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for Python and NumPy."""

    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, path: os.PathLike[str] | str) -> None:
    """Save data as pretty-printed JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


__all__ = ["save_pickle", "load_pickle", "hash_text", "set_random_seed", "save_json"]
