"""Evaluation utilities for Task A models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd

from common_utils.config import load_config
from common_utils.logging_conf import get_logger
from common_utils.metrics import (
    classification_report,
    compute_binary_metrics,
    confusion_matrix_plot,
    pr_curve_plot,
    roc_curve_plot,
)

LOGGER = get_logger(__name__)


def _compose_text(df: pd.DataFrame) -> pd.Series:
    return (df["subject"].fillna("") + " \n" + df["body_clean"].fillna("")).str.strip()


def _load_artifacts(artifact_path: Path) -> Dict[str, Any]:
    config_path = artifact_path / "preproc_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing preproc_config.json in {artifact_path}")
    config = json.loads(config_path.read_text())
    model_file = artifact_path / "model.pkl"
    vectorizer_file = artifact_path / "vectorizer.pkl"
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return {"model": model, "vectorizer": vectorizer, "config": config}


def evaluate_model(
    model_name: str,
    *,
    config_path: str,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path, overrides)
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    reports_dir = Path(paths.get("reports_dir", "reports"))
    figures_dir = Path(paths.get("figures_dir", reports_dir / "figures"))
    results_dir = Path(paths.get("results_dir", reports_dir / "results"))
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts")) / model_name

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    text_series = _compose_text(test_df)

    artifacts = _load_artifacts(artifacts_dir)
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]

    X_test = vectorizer.transform(text_series.tolist())
    y_true = test_df["label"].to_numpy()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
        if y_scores.ndim > 1:
            y_scores = y_scores[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred.astype(float)

    metrics = compute_binary_metrics(y_true, y_pred, y_scores)
    report = classification_report(y_true, y_pred)

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (results_dir / f"{model_name}_report.txt").write_text(report)

    confusion_matrix_plot(y_true, y_pred, figures_dir / f"{model_name}_confusion.png")
    if y_scores is not None and np.isfinite(y_scores).all():
        roc_curve_plot(y_true, y_scores, figures_dir / f"{model_name}_roc.png")
        pr_curve_plot(y_true, y_scores, figures_dir / f"{model_name}_pr.png")

    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("enable", True):
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "phishing-benchmark"))
        with mlflow.start_run(run_name=f"eval-{model_name}"):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    LOGGER.info("Evaluation complete for %s", model_name)
    return {"metrics": metrics, "report": report}
