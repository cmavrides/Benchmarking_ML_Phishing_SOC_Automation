"""Typer CLI for Task A workflows."""

from __future__ import annotations

from pathlib import Path

import typer

from common_utils.config import load_config
from common_utils.logging_conf import get_logger

from . import loaders
from .preprocess import run_preprocess
from .train import train_model
from .evaluate import evaluate_model
from .leaderboard import build_leaderboard

app = typer.Typer(help="Benchmarking workflow for phishing detection.")
LOGGER = get_logger(__name__)


@app.command()
def preprocess(
    config: Path = typer.Option(Path("configs/task_a.yaml"), help="Path to config file."),
    zefang: bool = typer.Option(True, help="Include Zefang Liu dataset."),
    cyradar: bool = typer.Option(True, help="Include CyRadar dataset."),
    lower: bool = typer.Option(True, help="Lowercase text."),
    strip_html_flag: bool = typer.Option(True, "--strip-html/--keep-html", help="Strip HTML from bodies."),
    save_parquet: bool = typer.Option(True, "--save-parquet/--no-save-parquet", help="Persist splits to parquet."),
    min_length: int = typer.Option(5, help="Minimum body length after cleaning."),
    max_length: int = typer.Option(5000, help="Maximum body length after cleaning (0 = unlimited)."),
    val_size: float = typer.Option(0.1, help="Validation set fraction."),
    test_size: float = typer.Option(0.1, help="Test set fraction."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    cfg = load_config(config)
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))

    df = loaders.load_datasets(raw_dir, use_zefang=zefang, use_cyradar=cyradar)
    splits = run_preprocess(
        df,
        lower=lower,
        strip_html_flag=strip_html_flag,
        min_length=min_length,
        max_length=max_length,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        save_dir=processed_dir if save_parquet else None,
    )
    for name, frame in splits.items():
        LOGGER.info("Split %s size: %d", name, len(frame))


@app.command()
def train(
    model: str = typer.Option("lr", help="Model key defined in config."),
    config: Path = typer.Option(Path("configs/task_a.yaml"), help="Path to config file."),
) -> None:
    artifact_path, metrics = train_model(model, config_path=str(config))
    typer.echo(f"Model '{model}' trained. Metrics: {metrics}. Artifacts at {artifact_path}")


@app.command()
def evaluate(
    model: str = typer.Option("lr", help="Model key to evaluate."),
    config: Path = typer.Option(Path("configs/task_a.yaml"), help="Path to config file."),
) -> None:
    result = evaluate_model(model, config_path=str(config))
    typer.echo(f"Evaluation metrics: {result['metrics']}")


@app.command()
def leaderboard(
    config: Path = typer.Option(Path("configs/task_a.yaml"), help="Path to config file."),
) -> None:
    path = build_leaderboard(str(config))
    typer.echo(f"Leaderboard written to {path}")


if __name__ == "__main__":
    app()
