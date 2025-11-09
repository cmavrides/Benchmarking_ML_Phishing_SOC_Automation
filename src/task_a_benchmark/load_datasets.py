"""Load and standardize phishing datasets into a unified schema."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.common.dataset_downloader import ensure_raw_datasets

try:  # Optional dependency used for auto row capping
    import psutil
except ImportError:  # pragma: no cover - psutil always installed via requirements
    psutil = None

DATA_DIR = Path("data")
EXPECTED_COLUMNS = {
    "text": ["text", "body", "email", "message", "email text", "content"],
    "label": ["label", "is_phishing", "target", "class", "phishing", "email type", "type"],
    "id": ["id", "email_id", "message_id"],
}
ROW_LIMIT_CONFIG = {
    "cyradar": {
        "env_var": "CYRADAR_MAX_ROWS",
        "auto_caps": (
            (16, 1_000_000),
            (24, 2_000_000),
            (32, 3_000_000),
        ),
        "default_cap": 1_000_000,
    }
}
AUTO_CAP_DISABLE_ENV = "TASKA_DISABLE_AUTO_ROW_CAP"


def _resolve_column(columns: Tuple[str, ...], candidates: list[str], default: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not find a column for {default}. Available columns: {columns}")


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "all", "full"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid integer value: {value}") from exc
    if parsed <= 0:
        return None
    return parsed


def _determine_row_limit(source_name: str) -> tuple[Optional[int], Optional[str], Optional[str]]:
    config = ROW_LIMIT_CONFIG.get(source_name)
    if not config:
        return None, None, None

    env_var = config.get("env_var")
    env_value = _parse_positive_int(os.getenv(env_var)) if env_var else None
    if env_value:
        return env_value, f"{env_var} override", env_var

    if os.getenv(AUTO_CAP_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return None, None, env_var

    auto_caps = config.get("auto_caps")
    if auto_caps and psutil is not None:
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        for max_ram_gb, cap in auto_caps:
            if total_gb <= max_ram_gb:
                return cap, f"auto-cap for <= {max_ram_gb} GB RAM", env_var

    default_cap = config.get("default_cap")
    if default_cap and psutil is None:
        return default_cap, "default cap (psutil unavailable)", env_var
    return None, None, env_var

    return None, None, env_var


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Convert arbitrary label encodings to {0,1}."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).apply(lambda x: 1 if int(x) == 1 else 0)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "phishing": 1,
        "phishing email": 1,
        "malicious": 1,
        "spam": 1,
        "safe": 0,
        "safe email": 0,
        "legit": 0,
        "legitimate": 0,
        "legitimate email": 0,
        "ham": 0,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        unknown = sorted(set(normalized[mapped.isna()]))
        raise ValueError(f"Unrecognized label values: {unknown[:5]}")
    return mapped.astype(int)


def _normalize_dataset(path: Path, source_name: str) -> pd.DataFrame:
    row_limit, reason, env_var = _determine_row_limit(source_name)
    read_kwargs = {}
    if row_limit:
        read_kwargs["nrows"] = row_limit
        reason_text = reason or "configured limit"
        hint = ""
        if env_var:
            hint = f" (set {env_var} or {AUTO_CAP_DISABLE_ENV}=1 to override)"
        print(f"[load_datasets] Limiting {source_name} to first {row_limit:,} rows ({reason_text}){hint}.")

    df = pd.read_csv(path, **read_kwargs)
    cols = tuple(df.columns.str.lower())
    df.columns = cols

    text_col = _resolve_column(cols, EXPECTED_COLUMNS["text"], "text")
    label_col = _resolve_column(cols, EXPECTED_COLUMNS["label"], "label")
    id_col = None
    for candidate in EXPECTED_COLUMNS["id"]:
        if candidate in cols:
            id_col = candidate
            break

    standardized = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": _normalize_labels(df[label_col]),
        }
    )

    if id_col:
        standardized.insert(0, "id", df[id_col].astype(str))
    else:
        standardized.insert(0, "id", standardized.index.astype(str))

    standardized.insert(1, "source", source_name)
    return standardized


def load_datasets(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load both datasets and concatenate them into a single DataFrame."""

    ensure_raw_datasets(data_dir)

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
