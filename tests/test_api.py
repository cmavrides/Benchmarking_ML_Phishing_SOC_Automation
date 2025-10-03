import json
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from task_b_soc import pipeline
from task_b_soc.serving import app


def _build_artifact(tmp_path: Path) -> Path:
    texts = ["hello", "verify account"]
    labels = np.array([0, 1])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(X, labels)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.pkl")
    joblib.dump(vectorizer, artifact_dir / "vectorizer.pkl")
    (artifact_dir / "preproc_config.json").write_text(
        json.dumps({"preprocess": {"lower": True, "strip_html": True}})
    )
    return artifact_dir


def _build_config(tmp_path: Path, artifact_dir: Path) -> Path:
    config = {
        "paths": {},
        "model": {
            "type": "classical",
            "artifact_dir": str(artifact_dir),
            "vectorizer_file": "vectorizer.pkl",
            "model_file": "model.pkl",
            "config_file": "preproc_config.json",
            "explain_top_n": 5,
        },
        "service": {"threshold": 0.5},
        "ioc": {"max_items": 5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))
    return config_path


def test_health_and_classify(tmp_path: Path, monkeypatch):
    artifact_dir = _build_artifact(tmp_path)
    config_path = _build_config(tmp_path, artifact_dir)
    monkeypatch.setenv("TASK_B_CONFIG", str(config_path))
    pipeline._pipeline_instance = None

    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    payload = {"subject": "Alert", "body": "Please verify your account", "as_html": False}
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {"label", "score", "iocs"}
