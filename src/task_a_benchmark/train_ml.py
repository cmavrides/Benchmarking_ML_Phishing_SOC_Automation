"""Train classical ML models using TF-IDF features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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


@dataclass
class ModelConfig:
    """Container describing an ML baseline and its search space."""

    name: str
    pipeline: Pipeline
    param_grid: Dict[str, Iterable[Any]]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run preprocess.py first.")
    return pd.read_csv(path)


def _prepare_data() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train_df = _load_split("train")
    val_df = _load_split("val")
    return train_df["clean_text"], train_df["label"], val_df["clean_text"], val_df["label"]


def _build_model_configs() -> List[ModelConfig]:
    """Define baseline estimators and hyper-parameter grids."""

    configs: List[ModelConfig] = []

    logreg = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
        (
            "clf",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="liblinear",
                dual=False,
            ),
        ),
    ])
    logreg_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_df": [0.85, 0.95],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l1", "l2"],
    }
    configs.append(ModelConfig("logistic_regression", logreg, logreg_grid))

    svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
        (
            "clf",
            CalibratedClassifierCV(
                base_estimator=LinearSVC(class_weight="balanced"),
                cv=3,
                method="sigmoid",
            ),
        ),
    ])
    svm_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__base_estimator__C": [0.25, 0.5, 1.0],
    }
    configs.append(ModelConfig("linear_svm", svm, svm_grid))

    rf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000)),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 30],
        "clf__max_features": ["sqrt", "log2"],
    }
    configs.append(ModelConfig("random_forest", rf, rf_grid))

    xgb = Pipeline([
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
                reg_lambda=1.0,
            ),
        ),
    ])
    xgb_grid = {
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 0.9],
        "clf__colsample_bytree": [0.7, 0.9],
    }
    configs.append(ModelConfig("xgboost", xgb, xgb_grid))

    return configs


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
    model_configs = _build_model_configs()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_summary: Dict[str, Dict[str, float]] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    best_estimators: Dict[str, Pipeline] = {}
    best_model_name = None
    best_f1 = -np.inf

    for config in model_configs:
        print(f"Training {config.name} with hyper-parameter search...")
        search = GridSearchCV(
            config.pipeline,
            param_grid=config.param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_pipeline: Pipeline = search.best_estimator_
        best_estimators[config.name] = best_pipeline
        y_pred = best_pipeline.predict(X_val)
        try:
            y_prob = _get_probabilities(best_pipeline, X_val)
        except AttributeError:
            y_prob = None
        metric_values = compute_classification_metrics(y_val, y_pred, y_prob)
        metrics_summary[config.name] = metric_values
        print_metrics(config.name, metric_values)

        best_params[config.name] = search.best_params_

        save_path = ARTIFACT_DIR / f"{config.name}.joblib"
        joblib.dump(best_pipeline, save_path)

        if metric_values.get("f1", 0.0) > best_f1:
            best_f1 = metric_values.get("f1", 0.0)
            best_model_name = config.name

    save_json({"metrics": metrics_summary, "best_params": best_params}, RESULTS_PATH)

    if best_model_name:
        best_pipeline = best_estimators[best_model_name]
        save_pickle(best_pipeline, BEST_MODEL_PATH)
        save_json(
            {
                "model": best_model_name,
                "metrics": metrics_summary[best_model_name],
                "best_params": best_params[best_model_name],
            },
            BEST_MODEL_META_PATH,
        )
        print(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    return metrics_summary


if __name__ == "__main__":
    train_models()
