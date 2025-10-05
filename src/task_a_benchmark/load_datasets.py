"""Load and standardize phishing datasets into a unified schema."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

DATA_DIR = Path("data")
EXPECTED_COLUMNS = {
    "text": ["text", "body", "email", "message"],
    "label": ["label", "is_phishing", "target", "class", "phishing"],
    "id": ["id", "email_id", "message_id"],
}


def _resolve_column(columns: Tuple[str, ...], candidates: list[str], default: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not find a column for {default}. Available columns: {columns}")


def _normalize_dataset(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = tuple(df.columns.str.lower())
    df.columns = cols

    text_col = _resolve_column(cols, EXPECTED_COLUMNS["text"], "text")
    label_col = _resolve_column(cols, EXPECTED_COLUMNS["label"], "label")
    id_col = None
    for candidate in EXPECTED_COLUMNS["id"]:
        if candidate in cols:
            id_col = candidate
            break

    standardized = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col].astype(int),
    })

    if id_col:
        standardized.insert(0, "id", df[id_col].astype(str))
    else:
        standardized.insert(0, "id", standardized.index.astype(str))

    standardized.insert(1, "source", source_name)
    standardized["label"] = standardized["label"].apply(lambda x: 1 if int(x) == 1 else 0)
    return standardized


def load_datasets(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load both datasets and concatenate them into a single DataFrame."""

    paths: Dict[str, Path] = {
        "zefang_liu": data_dir / "zefang_liu.csv",
        "cyradar": data_dir / "cyradar.csv",
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files: " + ", ".join(missing) + ". Place them under the data/ directory."
        )

    frames = [_normalize_dataset(path, name) for name, path in paths.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    return combined[["id", "source", "text", "label"]]


if __name__ == "__main__":
    df = load_datasets()
    print(df.head())
    print(df["label"].value_counts())
