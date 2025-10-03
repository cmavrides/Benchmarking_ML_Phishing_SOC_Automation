"""Classification pipeline for Task B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from common_utils.config import load_config
from common_utils.logging_conf import get_logger
from common_utils.text_cleaning import normalize, strip_html

from .enrich import extract_iocs

LOGGER = get_logger(__name__)


class ClassificationPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.paths = self.config.get("paths", {})
        model_cfg = self.config.get("model", {})
        self.threshold = model_cfg.get("threshold", self.config.get("service", {}).get("threshold", 0.5))
        artifact_dir = Path(model_cfg.get("artifact_dir", "artifacts/best_model"))
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory {artifact_dir} not found. Train Task A first.")
        self.artifact_dir = artifact_dir
        self.model_type = model_cfg.get("type", "classical")
        self.explain_top_n = model_cfg.get("explain_top_n", 10)
        self._load_artifacts(model_cfg)

    def _load_artifacts(self, model_cfg: Dict[str, Any]) -> None:
        if self.model_type == "classical":
            vectorizer_file = self.artifact_dir / model_cfg.get("vectorizer_file", "vectorizer.pkl")
            model_file = self.artifact_dir / model_cfg.get("model_file", "model.pkl")
            config_file = self.artifact_dir / model_cfg.get("config_file", "preproc_config.json")
            self.vectorizer = joblib.load(vectorizer_file)
            self.model = joblib.load(model_file)
            if config_file.exists():
                self.preprocess_cfg = json.loads(config_file.read_text()).get("preprocess", {})
            else:
                self.preprocess_cfg = {}
        else:
            raise NotImplementedError("Transformer serving not yet implemented")

    def _prepare(self, subject: str, body: str, as_html: bool) -> Dict[str, Any]:
        subject = subject or ""
        body = body or ""
        lower = self.preprocess_cfg.get("lower", True)
        strip_flag = self.preprocess_cfg.get("strip_html", True)
        if as_html or strip_flag:
            cleaned_body, was_html = strip_html(body)
        else:
            cleaned_body, was_html = body, False
        normalized_body = normalize(cleaned_body, lower=lower)
        normalized_subject = normalize(subject, lower=lower)
        combined = f"{normalized_subject} \n{normalized_body}".strip()
        return {
            "subject": subject,
            "body": body,
            "body_clean": normalized_body,
            "subject_clean": normalized_subject,
            "combined": combined,
            "was_html": was_html or as_html,
        }

    def score_text(self, subject: str, body: str, as_html: bool = False) -> Dict[str, Any]:
        prepared = self._prepare(subject, body, as_html)
        text = prepared["combined"]
        features = self.vectorizer.transform([text])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0, 1]
        elif hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(features)
            proba = 1 / (1 + np.exp(-raw[0]))
        else:
            proba = float(self.model.predict(features)[0])
        label = int(proba >= self.threshold)
        iocs = extract_iocs(body, max_items=self.config.get("ioc", {}).get("max_items", 20))
        explanations = self._build_explanations()
        return {
            "label": label,
            "score": float(proba),
            "explanations": explanations,
            "iocs": iocs,
            "normalized": {
                "subject": prepared["subject_clean"],
                "body": prepared["body_clean"],
            },
        }

    def _build_explanations(self) -> Dict[str, List[List[Any]]]:
        if not hasattr(self.model, "coef_"):
            return {}
        if not hasattr(self.vectorizer, "get_feature_names_out"):
            return {}
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        top_pos_idx = np.argsort(coefs)[-self.explain_top_n :][::-1]
        top_neg_idx = np.argsort(coefs)[: self.explain_top_n]
        top_positive = [[feature_names[i], float(coefs[i])] for i in top_pos_idx]
        top_negative = [[feature_names[i], float(coefs[i])] for i in top_neg_idx]
        return {"top_positive": top_positive, "top_negative": top_negative}


_pipeline_instance: ClassificationPipeline | None = None


def get_pipeline(config_path: str | None = None) -> ClassificationPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        config_path = config_path or "configs/task_b.yaml"
        _pipeline_instance = ClassificationPipeline(config_path)
    return _pipeline_instance
