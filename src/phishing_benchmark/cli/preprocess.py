"""CLI for preprocessing phishing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer

if __package__ in (None, ""):
    import sys

    SRC_DIR = Path(__file__).resolve().parents[2]
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from phishing_benchmark.config import DefaultConfig
from phishing_benchmark.data.cleaning import filter_by_length, normalize_text, strip_html
from phishing_benchmark.data.loaders import load_cyradar, load_zefang_liu
from phishing_benchmark.data.merge import merge_and_dedup
from phishing_benchmark.data.splits import make_stratified_splits
from phishing_benchmark.logging_conf import configure_logging
from phishing_benchmark.utils import ensure_dir, read_dataframe, write_dataframe

app = typer.Typer(add_completion=False)


def _load_or_download(name: str, raw_dir: Path):
    path = raw_dir / f"{name}.parquet"
    if path.exists():
        return read_dataframe(path)
    if name == "zefang_liu":
        return load_zefang_liu(raw_dir)
    if name == "cyradar":
        return load_cyradar(raw_dir)
    raise ValueError(f"Unknown dataset {name}")


@app.command()
def main(
    lower: bool = typer.Option(True, help="Apply lowercase normalization"),
    strip_html_flag: bool = typer.Option(True, "--strip-html/--keep-html", help="Strip HTML content"),
    min_length: int = typer.Option(0, help="Minimum body length after cleaning"),
    max_length: int = typer.Option(5000, help="Maximum body length after cleaning"),
    val_size: float = typer.Option(0.1, help="Validation split fraction"),
    test_size: float = typer.Option(0.1, help="Test split fraction"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Preprocess datasets into unified format and generate splits."""

    logger = configure_logging()
    config = DefaultConfig()
    raw_dir = config.raw_dir
    interim_dir = config.interim_dir
    processed_dir = config.processed_dir

    datasets: Dict[str, pd.DataFrame] = {}
    for name in ("zefang_liu", "cyradar"):
        logger.info("Loading dataset %s", name)
        df = _load_or_download(name, raw_dir)
        body_clean: List[str] = []
        body_is_html: List[bool] = []
        for body, html_flag in zip(df["body"], df.get("body_is_html", False)):
            cleaned = str(body or "")
            detected_html = bool(html_flag)
            if strip_html_flag:
                cleaned, detected = strip_html(cleaned)
                detected_html = detected_html or detected
            normalized = normalize_text(cleaned, lower=lower)
            body_clean.append(normalized)
            body_is_html.append(detected_html)
        df = df.assign(body_clean=body_clean, body_is_html=body_is_html)
        mask = list(filter_by_length(df["body_clean"], min_length=min_length, max_length=max_length))
        df = df.loc[mask].reset_index(drop=True)
        datasets[name] = df
        ensure_dir(interim_dir)
        write_dataframe(df, interim_dir / f"{name}_clean.parquet")

    merged = merge_and_dedup(datasets.values())
    train_df, val_df, test_df = make_stratified_splits(
        merged,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ensure_dir(processed_dir)
    write_dataframe(combined, processed_dir / "dataset.parquet")
    write_dataframe(train_df, processed_dir / "train.parquet")
    write_dataframe(val_df, processed_dir / "val.parquet")
    write_dataframe(test_df, processed_dir / "test.parquet")
    logger.info(
        "Preprocessing complete. train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )


if __name__ == "__main__":  # pragma: no cover
    app()
