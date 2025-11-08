"""Reporting utilities for evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from ..utils import dump_json, ensure_dir, save_leaderboard_row


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curve(y_true, y_scores, path: Path) -> None:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(recall, precision, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_metrics(metrics_dict: Dict[str, float], path: Path) -> None:
    dump_json(metrics_dict, path)


def update_leaderboard(
    path: Path,
    metrics_dict: Dict[str, float],
    model_name: str,
    run_id: str,
    extras: Dict[str, float] | None = None,
) -> None:
    row = {"model": model_name, "run_id": run_id}
    row.update(metrics_dict)
    if extras:
        row.update(extras)
    order = ["model", "run_id", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    save_leaderboard_row(path, row, order=order)
