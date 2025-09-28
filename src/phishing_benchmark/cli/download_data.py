"""CLI for downloading phishing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..config import DefaultConfig
from ..data.loaders import load_cyradar, load_zefang_liu
from ..logging_conf import configure_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    zefang_liu: bool = typer.Option(True, help="Download zefang-liu/phishing-email-dataset"),
    cyradar: bool = typer.Option(True, help="Download huynq3Cyradar/Phishing_Detection_Dataset"),
    sample: Optional[int] = typer.Option(None, help="Sample N rows from each dataset"),
    force: bool = typer.Option(False, help="Overwrite existing raw files"),
) -> None:
    """Download datasets from Hugging Face."""

    logger = configure_logging()
    config = DefaultConfig()
    raw_dir: Path = config.raw_dir
    if zefang_liu:
        logger.info("Downloading zefang-liu dataset")
        load_zefang_liu(raw_dir, sample=sample, force=force)
    if cyradar:
        logger.info("Downloading Cyradar dataset")
        load_cyradar(raw_dir, sample=sample, force=force)
    if not zefang_liu and not cyradar:
        logger.warning("No datasets selected; nothing to download")


if __name__ == "__main__":  # pragma: no cover
    app()
