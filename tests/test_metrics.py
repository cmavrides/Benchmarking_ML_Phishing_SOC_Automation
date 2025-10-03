import numpy as np

from common_utils.metrics import compute_binary_metrics


def test_metrics_compute_expected_scores():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.4])
    metrics = compute_binary_metrics(y_true, y_pred, y_scores)
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert round(metrics["recall"], 2) == 0.5
    assert "roc_auc" in metrics
