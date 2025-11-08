"""Inference pipeline for the SOC automation prototype."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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
    risk_level = _derive_risk_level(probability)
    confidence = float(abs(probability - 0.5) * 2)
    explanation = _get_feature_explanation(model, cleaned, label, probability)
    recommendations = _generate_recommendations(label, risk_level)

    return {
        "label": label,
        "score": probability,
        "risk_level": risk_level,
        "confidence": confidence,
        "explanations": explanation,
        "recommendations": recommendations,
    }


def reload_model() -> None:
    """Reset the cached model (useful for hot-reload scenarios)."""

    global _MODEL_CACHE
    _MODEL_CACHE = None


__all__ = ["classify_text", "reload_model"]


def _derive_risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "high"
    if probability >= 0.65:
        return "elevated"
    if probability >= 0.45:
        return "moderate"
    return "low"


def _get_feature_explanation(model: Any, text: str, label: int, probability: float) -> Dict[str, Any]:
    if not hasattr(model, "named_steps"):
        return {}

    vectorizer = model.named_steps.get("tfidf")
    if vectorizer is None:
        return {}

    feature_names = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([text]).toarray()[0]
    clf = model.named_steps.get("clf")

    explanation: Dict[str, Any] = {"method": "tfidf"}
    top_indices = np.argsort(vector)[::-1]
    top_terms: List[Dict[str, float]] = [
        {"term": feature_names[idx], "weight": float(vector[idx])}
        for idx in top_indices[:8]
        if vector[idx] > 0
    ]
    if top_terms:
        explanation["top_terms"] = top_terms

    if clf is None:
        return explanation

    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)[0]
        contributions = vector * coef
        pos_indices = np.argsort(contributions)[::-1]
        neg_indices = np.argsort(contributions)

        explanation["method"] = "linear_coefficients"
        explanation["supporting_terms"] = [
            {"term": feature_names[idx], "contribution": float(contributions[idx])}
            for idx in pos_indices[:5]
            if contributions[idx] > 0
        ]
        explanation["mitigating_terms"] = [
            {"term": feature_names[idx], "contribution": float(contributions[idx])}
            for idx in neg_indices[:5]
            if contributions[idx] < 0
        ]

    explanation["rationale"] = _compose_rationale(label, probability, explanation)
    return explanation


def _compose_rationale(label: int, probability: float, explanation: Dict[str, Any]) -> str:
    status = "phishing" if label == 1 else "legitimate"
    confidence = f"{probability * 100:.1f}%"
    terms = explanation.get("supporting_terms") or explanation.get("top_terms", [])
    if not terms:
        return f"Model judges the message as {status} with {confidence} confidence."

    highlights = ", ".join(term["term"] for term in terms[:3])
    return (
        f"Model judges the message as {status} with {confidence} confidence, driven by tokens: {highlights}."
    )


def _generate_recommendations(label: int, risk_level: str) -> List[str]:
    if label == 1:
        recommendations = [
            "Quarantine the message and block the sender domain.",
            "Open a phishing investigation ticket with the SOC platform.",
        ]
        if risk_level in {"high", "elevated"}:
            recommendations.append("Trigger user password reset if credentials may be exposed.")
        return recommendations

    guidance = ["Log the event for monitoring."]
    if risk_level in {"moderate", "elevated"}:
        guidance.append("Consider manual review to confirm legitimacy before releasing.")
    return guidance
