"""Aggregate MLflow runs into a leaderboard."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd

from common_utils.config import load_config
from common_utils.logging_conf import get_logger

LOGGER = get_logger(__name__)


def build_leaderboard(config_path: str, overrides: Dict[str, object] | None = None) -> Path:
    config = load_config(config_path, overrides)
    paths = config.get("paths", {})
    reports_dir = Path(paths.get("reports_dir", "reports"))
    results_dir = Path(paths.get("results_dir", reports_dir / "results"))
    leaderboard_path = results_dir / "leaderboard.csv"

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    experiment_name = mlflow_cfg.get("experiment_name", "phishing-benchmark")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        LOGGER.warning("No MLflow runs found for experiment %s", experiment_name)
        results_dir.mkdir(parents=True, exist_ok=True)
        leaderboard_path.write_text("run_id,model,val_f1,val_pr_auc\n")
        return leaderboard_path

    def _get(col: str) -> pd.Series:
        return runs.get(col, pd.Series(dtype=float))

    leaderboard = pd.DataFrame(
        {
            "run_id": runs["run_id"],
            "model": runs.get("params.model", runs.get("tags.mlflow.runName", "")),
            "val_f1": _get("metrics.val_f1"),
            "val_pr_auc": _get("metrics.val_pr_auc"),
            "test_f1": _get("metrics.test_f1"),
            "test_pr_auc": _get("metrics.test_pr_auc"),
        }
    )

    sort_cols = []
    if not leaderboard["test_f1"].isna().all():
        sort_cols.append("test_f1")
    sort_cols.append("val_f1")
    if not leaderboard["test_pr_auc"].isna().all():
        sort_cols.append("test_pr_auc")
    sort_cols.append("val_pr_auc")
    sort_cols = [col for col in sort_cols if col in leaderboard.columns]

    leaderboard = leaderboard.sort_values(by=sort_cols, ascending=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_path, index=False)
    LOGGER.info("Saved leaderboard to %s", leaderboard_path)
    return leaderboard_path
