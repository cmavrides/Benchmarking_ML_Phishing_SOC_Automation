"""Inference pipeline for the SOC automation prototype."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.common.text_cleaning import normalize_text, strip_html
from src.common.utils import load_pickle

BEST_MODEL_PATH = Path("artifacts/best_model.joblib")
FALLBACK_MODEL_PATH = Path("artifacts/ml/logistic_regression.joblib")

_MODEL_CACHE = None


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if BEST_MODEL_PATH.exists():
        _MODEL_CACHE = load_pickle(BEST_MODEL_PATH)
    elif FALLBACK_MODEL_PATH.exists():
        _MODEL_CACHE = load_pickle(FALLBACK_MODEL_PATH)
    else:
        raise FileNotFoundError(
            "No trained model found. Run 'python src/task_a_benchmark/run_all.py' to train and save a model."
        )
    return _MODEL_CACHE


def classify_text(text: str, is_html: bool = False) -> Dict[str, Any]:
    """Classify text using the trained phishing model."""

    model = _load_model()
    cleaned = strip_html(text) if is_html else text
    cleaned = normalize_text(cleaned)
    prob = model.predict_proba([cleaned])[:, 1] if hasattr(model, "predict_proba") else None

    if prob is None:
        decision = model.decision_function([cleaned])
        prob = 1 / (1 + np.exp(-decision))

    probability = float(prob[0])
    label = int(probability >= 0.5)
    return {"label": label, "score": probability}


def reload_model() -> None:
    """Reset the cached model (useful for hot-reload scenarios)."""

    global _MODEL_CACHE
    _MODEL_CACHE = None


__all__ = ["classify_text", "reload_model"]
