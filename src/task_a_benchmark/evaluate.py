"""Evaluate trained models on the test split and produce reports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.metrics import (compute_classification_metrics, confusion_matrix,
                                print_metrics)
from src.common.utils import load_pickle, save_json

PROCESSED_DIR = Path("data/processed")
ML_ARTIFACT_DIR = Path("artifacts/ml")
DISTILBERT_DIR = Path("artifacts/distilbert")
RESULTS_PATH = Path("reports/results/metrics.json")
FIGURE_DIR = Path("reports/figures")


def _load_test_split() -> pd.DataFrame:
    path = PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError("Test split not found. Run preprocess.py first.")
    return pd.read_csv(path)


def _evaluate_ml_models(test_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray | Dict[str, float]]]:
    results: Dict[str, Dict[str, np.ndarray | Dict[str, float]]] = {}
    if not ML_ARTIFACT_DIR.exists():
        return results

    for model_path in ML_ARTIFACT_DIR.glob("*.joblib"):
        name = model_path.stem
        pipeline = load_pickle(model_path)
        y_true = test_df["label"].values
        y_pred = pipeline.predict(test_df["clean_text"])
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(test_df["clean_text"])[:, 1]
            else:
                clf = pipeline.named_steps.get("clf")
                if hasattr(clf, "predict_proba"):
                    y_prob = pipeline.predict_proba(test_df["clean_text"])[:, 1]
                elif hasattr(clf, "decision_function"):
                    decision = pipeline.decision_function(test_df["clean_text"])
                    y_prob = 1 / (1 + np.exp(-decision))
                else:
                    y_prob = None
        except Exception:
            y_prob = None
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        print_metrics(name, metrics)
        results[name] = {"metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}

    return results


def _evaluate_transformer(test_df: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray] | Tuple[Dict[str, float], None]:
    if not DISTILBERT_DIR.exists():
        return {}, None

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.eval()

    texts = test_df["clean_text"].tolist()
    y_true = test_df["label"].values
    probs: List[float] = []
    preds: List[int] = []

    for i in tqdm(range(0, len(texts), 16), desc="DistilBERT eval"):
        batch = texts[i:i + 16]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        probs.extend(batch_probs.tolist())
        preds.extend((np.array(batch_probs) > 0.5).astype(int).tolist())

    y_pred = np.array(preds)
    metrics = compute_classification_metrics(y_true, y_pred, probs)
    print_metrics("distilbert", metrics)
    return metrics, y_pred


def _plot_metrics(metrics_summary: Dict[str, Dict[str, float]]) -> None:
    if not metrics_summary:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    models = list(metrics_summary.keys())
    f1_scores = [metrics_summary[m].get("f1", np.nan) for m in models]

    plt.figure(figsize=(8, 4))
    plt.bar(models, f1_scores, color="steelblue")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison (F1 on Test Set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_f1_scores.png")
    plt.close()


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm, labels = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"confusion_matrix_{title.lower().replace(' ', '_')}.png")
    plt.close()


def evaluate_all() -> Dict[str, Dict[str, float]]:
    test_df = _load_test_split()
    ml_results = _evaluate_ml_models(test_df)
    distilbert_metrics, distilbert_preds = _evaluate_transformer(test_df)

    metrics_summary: Dict[str, Dict[str, float]] = {name: info["metrics"] for name, info in ml_results.items()}
    if distilbert_metrics:
        metrics_summary["distilbert"] = distilbert_metrics

    if metrics_summary:
        best_model = max(metrics_summary.items(), key=lambda item: item[1].get("f1", 0.0))[0]
        y_true = test_df["label"].values
        if best_model == "distilbert" and distilbert_preds is not None:
            _plot_confusion_matrix(y_true, distilbert_preds, "DistilBERT Test Confusion Matrix")
        elif best_model in ml_results:
            y_pred = ml_results[best_model]["y_pred"]
            _plot_confusion_matrix(y_true, y_pred, f"{best_model} Test Confusion Matrix")

    _plot_metrics(metrics_summary)
    save_json(metrics_summary, RESULTS_PATH)
    return metrics_summary


if __name__ == "__main__":
    evaluate_all()
