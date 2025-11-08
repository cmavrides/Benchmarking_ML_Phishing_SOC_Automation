"""Evaluation helpers for phishing detection models."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn import metrics

MetricDict = Dict[str, float]


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float] | None = None) -> MetricDict:
    """Compute accuracy, precision, recall, F1, and ROC-AUC.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Optional predicted probabilities for ROC-AUC.

    Returns:
        Dictionary of metric name to value.
    """

    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))
    metrics_dict: MetricDict = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics_dict["roc_auc"] = float("nan")

        try:
            metrics_dict["average_precision"] = metrics.average_precision_score(y_true, y_prob)
        except ValueError:
            metrics_dict["average_precision"] = float("nan")

        try:
            metrics_dict["brier_score"] = metrics.brier_score_loss(y_true, y_prob)
        except ValueError:
            metrics_dict["brier_score"] = float("nan")

    return metrics_dict


def print_metrics(name: str, metrics_dict: MetricDict) -> None:
    """Pretty-print evaluation metrics."""

    summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items())
    print(f"[{name}] {summary}")


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confusion matrix and class labels."""

    labels = np.array([0, 1])
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


__all__ = ["compute_classification_metrics", "print_metrics", "confusion_matrix"]
