"""Evaluation metrics for phishing detection."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics
from sklearn.utils.extmath import softmax


def compute_binary_metrics(y_true, y_scores) -> Dict[str, float]:
    """Compute binary classification metrics from true labels and predicted probabilities/logits."""

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        y_prob = softmax(y_scores, axis=1)[:, 1]
    else:
        y_prob = y_scores.ravel()

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": metrics.average_precision_score(y_true, y_prob),
    }
