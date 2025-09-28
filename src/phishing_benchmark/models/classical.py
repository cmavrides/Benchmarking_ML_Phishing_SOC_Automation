"""Classical machine learning models for phishing detection."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(name: str, class_weight: str | None = "balanced", **kwargs: Any):
    """Return a configured classical model by name."""

    name = name.lower()
    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            n_jobs=-1,
            **kwargs,
        )
    if name == "svm":
        return LinearSVC(class_weight=class_weight, **kwargs)
    if name == "nb":
        return MultinomialNB(**kwargs)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            n_jobs=-1,
            **kwargs,
        )
    if name == "xgb":
        params: Dict[str, Any] = {
            "objective": "binary:logistic",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "aucpr",
            "scale_pos_weight": kwargs.pop("scale_pos_weight", 1.0),
            "tree_method": kwargs.pop("tree_method", "hist"),
        }
        params.update(kwargs)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown classical model: {name}")
