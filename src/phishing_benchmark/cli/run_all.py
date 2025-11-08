"""Convenience CLI to execute the full benchmarking workflow."""

from __future__ import annotations

from typing import List, Optional

import typer

from ..config import DefaultConfig
from ..logging_conf import configure_logging
from .download_data import main as download_main
from .preprocess import main as preprocess_main
from .train import main as train_main
from .evaluate import main as evaluate_main

app = typer.Typer(add_completion=False)


@app.command()
def main(
    models: Optional[List[str]] = typer.Option(None, help="Models to train and evaluate"),
    seed: int = typer.Option(42, help="Random seed"),
    strip_html: bool = typer.Option(True, help="Strip HTML during preprocessing"),
) -> None:
    """Run download → preprocess → train → evaluate for selected models."""

    logger = configure_logging()
    config = DefaultConfig()
    selected_models = models or config.models

    logger.info("Running full pipeline for models: %s", ", ".join(selected_models))
    download_main()
    preprocess_main(strip_html_flag=strip_html, seed=seed)
    for model in selected_models:
        train_main(model=model)
        evaluate_main(model=model)


if __name__ == "__main__":  # pragma: no cover
    app()
