"""Classical ML models for phishing detection."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def build_model(name: str, params: Dict[str, Any]) -> Any:
    name = name.lower()
    if name == "lr":
        defaults = {"solver": "liblinear", "class_weight": "balanced", "max_iter": 200}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if name == "svm":
        defaults = {"C": 1.0}
        defaults.update(params)
        return LinearSVC(**defaults)
    if name == "nb":
        defaults = {}
        defaults.update(params)
        return MultinomialNB(**defaults)
    if name == "rf":
        defaults = {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    raise ValueError(f"Unsupported classical model: {name}")
