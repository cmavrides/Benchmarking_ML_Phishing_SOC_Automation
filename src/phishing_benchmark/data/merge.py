"""Dataset merging helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def merge_and_dedup(
    frames: Iterable[pd.DataFrame],
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge multiple dataframes and drop duplicates by key columns."""

    key_cols = list(key_cols or ("subject", "body_clean"))
    frames_list: List[pd.DataFrame] = [df.copy() for df in frames]
    if not frames_list:
        return pd.DataFrame(columns=key_cols)

    merged = pd.concat(frames_list, ignore_index=True)
    for column in key_cols:
        if column not in merged.columns:
            merged[column] = ""

    merged = merged.drop_duplicates(subset=list(key_cols), keep="first", ignore_index=True)
    return merged
