"""Evaluate trained models on the test split and produce reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, precision_recall_curve, roc_curve)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.metrics import (compute_classification_metrics, confusion_matrix,
                                print_metrics)
from src.common.utils import load_pickle, save_json

PROCESSED_DIR = Path("data/processed")
ML_ARTIFACT_DIR = Path("artifacts/ml")
DISTILBERT_DIR = Path("artifacts/distilbert")
RESULTS_PATH = Path("reports/results/metrics.json")
SOURCE_RESULTS_PATH = Path("reports/results/source_metrics.json")
FIGURE_DIR = Path("reports/figures")


@dataclass
class ModelEvaluation:
    """Container holding evaluation artefacts for a model."""

    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    metrics: Dict[str, float]


def _load_test_split() -> pd.DataFrame:
    path = PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError("Test split not found. Run preprocess.py first.")
    return pd.read_csv(path).reset_index(drop=True)


def _evaluate_ml_models(test_df: pd.DataFrame) -> List[ModelEvaluation]:
    evaluations: List[ModelEvaluation] = []
    if not ML_ARTIFACT_DIR.exists():
        return evaluations

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
        y_prob_array: Optional[np.ndarray] = None if y_prob is None else np.asarray(y_prob)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob_array)
        print_metrics(name, metrics)
        evaluations.append(
            ModelEvaluation(
                name=name,
                y_true=np.asarray(y_true),
                y_pred=np.asarray(y_pred),
                y_prob=y_prob_array,
                metrics=metrics,
            )
        )

    return evaluations


def _evaluate_transformer(test_df: pd.DataFrame) -> Optional[ModelEvaluation]:
    if not DISTILBERT_DIR.exists():
        return None

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
    prob_array = np.asarray(probs)
    metrics = compute_classification_metrics(y_true, y_pred, prob_array)
    print_metrics("distilbert", metrics)
    return ModelEvaluation(
        name="distilbert",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=prob_array,
        metrics=metrics,
    )


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


def _plot_precision_recall_curves(evaluations: List[ModelEvaluation]) -> None:
    pr_evaluations = [ev for ev in evaluations if ev.y_prob is not None]
    if not pr_evaluations:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for eval_result in pr_evaluations:
        precision, recall, _ = precision_recall_curve(eval_result.y_true, eval_result.y_prob)
        label = eval_result.name
        ap = eval_result.metrics.get("average_precision")
        if ap is not None:
            label += f" (AP={ap:.3f})"
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax, name=label)

    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_curves.png")
    plt.close(fig)


def _plot_roc_curves(evaluations: List[ModelEvaluation]) -> None:
    roc_evaluations = [ev for ev in evaluations if ev.y_prob is not None]
    if not roc_evaluations:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for eval_result in roc_evaluations:
        fpr, tpr, _ = roc_curve(eval_result.y_true, eval_result.y_prob)
        label = eval_result.name
        roc_auc = eval_result.metrics.get("roc_auc")
        if roc_auc is not None:
            label += f" (AUC={roc_auc:.3f})"
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curves.png")
    plt.close(fig)


def _plot_calibration_curves(evaluations: List[ModelEvaluation]) -> None:
    calibrated = [ev for ev in evaluations if ev.y_prob is not None]
    if not calibrated:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for eval_result in calibrated:
        CalibrationDisplay.from_predictions(
            eval_result.y_true,
            eval_result.y_prob,
            n_bins=10,
            name=eval_result.name,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "calibration_curves.png")
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


def _compute_source_breakdown(test_df: pd.DataFrame, evaluations: List[ModelEvaluation]) -> Dict[str, Dict[str, Dict[str, float]]]:
    if "source" not in test_df.columns or not evaluations:
        return {}

    grouped = list(test_df.reset_index(drop=True).groupby("source", sort=False))
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}

    for eval_result in evaluations:
        model_breakdown: Dict[str, Dict[str, float]] = {}
        for source, group in grouped:
            indices = group.index.to_numpy()
            y_true = group["label"].to_numpy()
            y_pred = eval_result.y_pred[indices]
            y_prob = eval_result.y_prob[indices] if eval_result.y_prob is not None else None
            model_breakdown[source] = compute_classification_metrics(y_true, y_pred, y_prob)
        breakdown[eval_result.name] = model_breakdown

    return breakdown


def evaluate_all() -> Dict[str, Dict[str, float]]:
    test_df = _load_test_split()
    ml_evaluations = _evaluate_ml_models(test_df)
    evaluations: List[ModelEvaluation] = list(ml_evaluations)

    distilbert_eval = _evaluate_transformer(test_df)
    if distilbert_eval is not None:
        evaluations.append(distilbert_eval)

    metrics_summary: Dict[str, Dict[str, float]] = {eval_result.name: eval_result.metrics for eval_result in evaluations}

    if metrics_summary:
        best_model_name = max(metrics_summary.items(), key=lambda item: item[1].get("f1", 0.0))[0]
        best_eval = next((ev for ev in evaluations if ev.name == best_model_name), None)
        if best_eval is not None:
            _plot_confusion_matrix(test_df["label"].values, best_eval.y_pred, f"{best_model_name} Test Confusion Matrix")

    _plot_metrics(metrics_summary)
    _plot_precision_recall_curves(evaluations)
    _plot_roc_curves(evaluations)
    _plot_calibration_curves(evaluations)

    source_breakdown = _compute_source_breakdown(test_df, evaluations)
    if source_breakdown:
        save_json(source_breakdown, SOURCE_RESULTS_PATH)

    save_json(metrics_summary, RESULTS_PATH)
    return metrics_summary


if __name__ == "__main__":
    evaluate_all()
