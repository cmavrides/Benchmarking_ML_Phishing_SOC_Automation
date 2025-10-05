"""Train classical ML models using TF-IDF features."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.common.metrics import compute_classification_metrics, print_metrics
from src.common.utils import save_json, save_pickle, set_random_seed

PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/ml")
RESULTS_PATH = Path("reports/results/ml_metrics.json")
BEST_MODEL_PATH = Path("artifacts/best_model.joblib")
BEST_MODEL_META_PATH = Path("artifacts/best_model_meta.json")
def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run preprocess.py first.")
    return pd.read_csv(path)


def _prepare_data() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train_df = _load_split("train")
    val_df = _load_split("val")
    return train_df["clean_text"], train_df["label"], val_df["clean_text"], val_df["label"]


def _build_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models["logistic_regression"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
    ])

    models["linear_svm"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("clf", LinearSVC(class_weight="balanced")),
    ])

    models["random_forest"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1)),
    ])

    models["xgboost"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000)),
        (
            "clf",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                max_depth=6,
                n_estimators=400,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
            ),
        ),
    ])

    return models


def _get_probabilities(pipeline: Pipeline, texts: pd.Series) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    if hasattr(clf, "decision_function"):
        decisions = pipeline.decision_function(texts)
        return 1 / (1 + np.exp(-decisions))
    raise AttributeError("Classifier does not support probability estimates.")


def train_models(seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Train all ML baselines and persist artifacts."""

    set_random_seed(seed)
    X_train, y_train, X_val, y_val = _prepare_data()
    models = _build_models()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_summary: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_f1 = -np.inf

    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        try:
            y_prob = _get_probabilities(pipeline, X_val)
        except AttributeError:
            y_prob = None
        metric_values = compute_classification_metrics(y_val, y_pred, y_prob)
        metrics_summary[name] = metric_values
        print_metrics(name, metric_values)

        save_path = ARTIFACT_DIR / f"{name}.joblib"
        joblib.dump(pipeline, save_path)

        if metric_values.get("f1", 0.0) > best_f1:
            best_f1 = metric_values.get("f1", 0.0)
            best_model_name = name

    save_json(metrics_summary, RESULTS_PATH)

    if best_model_name:
        best_pipeline = models[best_model_name]
        save_pickle(best_pipeline, BEST_MODEL_PATH)
        save_json({"model": best_model_name, "metrics": metrics_summary[best_model_name]}, BEST_MODEL_META_PATH)
        print(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    return metrics_summary


if __name__ == "__main__":
    train_models()
