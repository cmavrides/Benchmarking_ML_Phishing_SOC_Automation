"""CLI to evaluate trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

if __package__ in (None, ""):
    import sys

    SRC_DIR = Path(__file__).resolve().parents[2]
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from phishing_benchmark.config import DefaultConfig
from phishing_benchmark.eval.metrics import compute_binary_metrics
from phishing_benchmark.eval.reporting import (
    export_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    update_leaderboard,
)
from phishing_benchmark.logging_conf import configure_logging
from phishing_benchmark.utils import ensure_dir, read_dataframe

app = typer.Typer(add_completion=False)


def _load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    dataset = read_dataframe(processed_dir / "dataset.parquet")
    return dataset.loc[dataset["split"] == split].reset_index(drop=True)


@app.command()
def main(
    model: str = typer.Option("lr", help="Model name to evaluate"),
    split: str = typer.Option("test", help="Dataset split to evaluate"),
    checkpoint_dir: Optional[Path] = typer.Option(None, help="Checkpoint directory"),
) -> None:
    """Evaluate a trained model and generate reports."""

    logger = configure_logging()
    config = DefaultConfig()
    processed_dir = config.processed_dir
    reports_dir = config.reports_dir
    ensure_dir(reports_dir)
    checkpoint_dir = checkpoint_dir or (Path("models") / model)

    df = _load_split(processed_dir, split)
    texts = df["body_clean"].tolist()
    labels = df["label"].to_numpy()

    if model in {"lr", "svm", "nb", "rf", "xgb"}:
        bundle = joblib.load(checkpoint_dir / "model.joblib")
        estimator = bundle["model"]
        vectorizer = bundle["vectorizer"]
        transformed = vectorizer.transform(texts)
        if hasattr(estimator, "predict_proba"):
            preds_proba = estimator.predict_proba(transformed)
            scores = preds_proba[:, 1]
        else:
            decision = estimator.decision_function(transformed)
            scores = 1 / (1 + np.exp(-decision))
    elif model in {"distilbert", "roberta"}:
        model_name = "distilbert-base-uncased" if model == "distilbert" else "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir if checkpoint_dir.exists() else model_name)
        clf_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir if checkpoint_dir.exists() else model_name
        )
        clf_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf_model.to(device)
        scores = []
        with torch.no_grad():
            for text in texts:
                tokens = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                logits = clf_model(**tokens).logits
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                scores.append(prob)
        scores = np.array(scores)
    else:
        raise typer.BadParameter(f"Unsupported model: {model}")

    metrics_dict = compute_binary_metrics(labels, scores)
    metrics_path = reports_dir / f"metrics_{model}.json"
    export_metrics(metrics_dict, metrics_path)
    preds = (scores >= 0.5).astype(int)
    plot_confusion_matrix(labels, preds, reports_dir / "plots" / f"{model}_confusion_matrix.png")
    plot_pr_curve(labels, scores, reports_dir / "plots" / f"{model}_pr_curve.png")
    update_leaderboard(reports_dir / "leaderboard.csv", metrics_dict, model_name=model, run_id="manual")
    logger.info("Evaluation complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":  # pragma: no cover
    app()
