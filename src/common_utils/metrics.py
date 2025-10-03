"""Metric helpers for binary phishing detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from .logging_conf import get_logger

LOGGER = get_logger(__name__)


MetricDict = Dict[str, float]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray | None = None) -> MetricDict:
    """Compute a suite of binary classification metrics."""

    results: MetricDict = {}
    results["accuracy"] = metrics.accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    results.update({"precision": precision, "recall": recall, "f1": f1})

    macro_precision, macro_recall, macro_f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    results.update(
        {
            "precision_macro": macro_precision,
            "recall_macro": macro_recall,
            "f1_macro": macro_f1,
        }
    )

    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            results["roc_auc"] = metrics.roc_auc_score(y_true, y_scores)
            results["pr_auc"] = metrics.average_precision_score(y_true, y_scores)
        except ValueError as exc:
            LOGGER.warning("Skipping probability metrics: %s", exc)
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")

    return results


def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> Path:
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Legit", "Phishing"])
    ax.set_yticks([0, 1], labels=["Legit", "Phishing"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def roc_curve_plot(y_true: np.ndarray, y_scores: np.ndarray, path: str | Path) -> Path:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def pr_curve_plot(y_true: np.ndarray, y_scores: np.ndarray, path: str | Path) -> Path:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    pr_auc = metrics.auc(recall, precision)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return metrics.classification_report(y_true, y_pred, zero_division=0)
