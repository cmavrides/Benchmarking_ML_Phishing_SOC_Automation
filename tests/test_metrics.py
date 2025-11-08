import numpy as np

from phishing_benchmark.eval.metrics import compute_binary_metrics


def test_compute_binary_metrics_basic():
    y_true = np.array([0, 1, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    metrics = compute_binary_metrics(y_true, y_scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] > 0.9
