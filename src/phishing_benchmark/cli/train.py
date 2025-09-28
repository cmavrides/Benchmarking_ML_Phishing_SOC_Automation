"""CLI for training phishing detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import classification_report

if __package__ in (None, ""):
    import sys

    SRC_DIR = Path(__file__).resolve().parents[2]
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from phishing_benchmark.config import DefaultConfig
from phishing_benchmark.features.vectorizers import get_tfidf
from phishing_benchmark.logging_conf import configure_logging
from phishing_benchmark.models.classical import get_model as get_classical_model
from phishing_benchmark.models.transformers import TransformerConfig, train_transformer
from phishing_benchmark.utils import ensure_dir, read_dataframe

app = typer.Typer(add_completion=False)


def _load_splits(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = read_dataframe(processed_dir / "dataset.parquet")
    train_df = dataset.loc[dataset["split"] == "train"].reset_index(drop=True)
    val_df = dataset.loc[dataset["split"] == "val"].reset_index(drop=True)
    test_df = dataset.loc[dataset["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


@app.command()
def main(
    model: str = typer.Option("lr", help="Model to train (lr, svm, nb, rf, xgb, bilstm, distilbert, roberta)"),
    max_features: int = typer.Option(100000, help="Max TF-IDF features"),
    ngram_min: int = typer.Option(1, help="Min n-gram"),
    ngram_max: int = typer.Option(2, help="Max n-gram"),
    epochs: int = typer.Option(3, help="Epochs for neural models"),
    batch_size: int = typer.Option(16, help="Batch size for neural models"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    checkpoint_dir: Optional[Path] = typer.Option(None, help="Directory to store checkpoints"),
) -> None:
    """Train a selected model on the processed dataset."""

    logger = configure_logging()
    config = DefaultConfig()
    processed_dir = config.processed_dir
    checkpoint_dir = checkpoint_dir or (Path("models") / model)
    ensure_dir(checkpoint_dir)

    train_df, val_df, test_df = _load_splits(processed_dir)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("phishing-benchmark")

    with mlflow.start_run(run_name=f"train_{model}") as run:
        logger.info("Training model: %s", model)
        mlflow.log_params({"model": model})
        if model in {"lr", "svm", "nb", "rf", "xgb"}:
            vectorizer = get_tfidf(
                max_features=max_features,
                ngram_range=(ngram_min, ngram_max),
            )
            X_train = vectorizer.fit_transform(train_df["body_clean"])
            y_train = train_df["label"].to_numpy()
            X_val = vectorizer.transform(val_df["body_clean"])
            y_val = val_df["label"].to_numpy()

            estimator = get_classical_model(model)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_val)
            report = classification_report(y_val, predictions, output_dict=True, zero_division=0)
            mlflow.log_metrics({"val_f1": report["weighted avg"]["f1-score"], "val_accuracy": report["accuracy"]})
            joblib.dump({"model": estimator, "vectorizer": vectorizer}, checkpoint_dir / "model.joblib")
            logger.info("Model saved to %s", checkpoint_dir / "model.joblib")
        elif model in {"distilbert", "roberta"}:
            transformer_config = TransformerConfig(
                model_name="distilbert-base-uncased" if model == "distilbert" else "roberta-base",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=str(checkpoint_dir),
            )
            metrics = train_transformer(
                train_df["body_clean"],
                train_df["label"],
                val_df["body_clean"],
                val_df["label"],
                transformer_config,
            )
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{key}", value)
        else:
            raise typer.BadParameter(f"Unsupported model: {model}")

        mlflow.log_artifact(str(checkpoint_dir))
        logger.info("Training finished. Run ID: %s", run.info.run_id)


if __name__ == "__main__":  # pragma: no cover
    app()
