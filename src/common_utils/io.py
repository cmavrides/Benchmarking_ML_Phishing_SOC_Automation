"""I/O helpers for reading and writing data safely."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional

import pandas as pd

from .logging_conf import get_logger

LOGGER = get_logger(__name__)


def _try_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        LOGGER.warning("Falling back to latin-1 decoding for %s", path)
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def read_csv(path: str | Path, chunksize: Optional[int] = None, **kwargs: object) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a CSV file with robust encoding handling."""

    path = Path(path)
    if chunksize:
        return pd.read_csv(path, encoding="utf-8", chunksize=chunksize, **kwargs)
    return _try_read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8", **kwargs)


def read_parquet(path: str | Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def chunked_csv_reader(path: str | Path, chunksize: int, **kwargs: object) -> Generator[pd.DataFrame, None, None]:
    path = Path(path)
    for chunk in pd.read_csv(path, encoding="utf-8", chunksize=chunksize, **kwargs):
        yield chunk
