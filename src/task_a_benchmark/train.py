"""Training utilities for Task A models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from common_utils.config import load_config
from common_utils.logging_conf import get_logger
from common_utils.metrics import compute_binary_metrics

from .features import FeaturePack, build_hashing, build_tfidf
from .models_classical import build_model as build_classical_model
from .models_transformers import train_transformer

LOGGER = get_logger(__name__)


def load_splits(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    splits = {}
    for split in ["train", "val", "test"]:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split {path}")
        splits[split] = pd.read_parquet(path)
    return splits


def _compose_text(df: pd.DataFrame) -> pd.Series:
    return (df["subject"].fillna("") + " \n" + df["body_clean"].fillna("")).str.strip()


def _scores_from_model(model: Any, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return proba
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            return scores
        return scores[:, 1]
    predictions = model.predict(X)
    return predictions.astype(float)


def _vectorise(
    train_texts: pd.Series,
    val_texts: pd.Series,
    test_texts: pd.Series,
    feature_cfg: Dict[str, Any],
    feature_type: str = "tfidf",
) -> FeaturePack:
    if feature_type == "hashing":
        return build_hashing(
            train_texts.tolist(),
            val_texts.tolist(),
            test_texts.tolist(),
            feature_cfg,
        )
    return build_tfidf(
        train_texts.tolist(),
        val_texts.tolist(),
        test_texts.tolist(),
        feature_cfg,
    )


def _train_classical(
    model_name: str,
    params: Dict[str, Any],
    feature_cfg: Dict[str, Any],
    feature_type: str,
    preprocess_cfg: Dict[str, Any],
    splits: Dict[str, pd.DataFrame],
    artifacts_dir: Path,
    seed: int,
    mlflow_enabled: bool,
) -> Tuple[Path, Dict[str, float]]:
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    train_texts = _compose_text(train_df)
    val_texts = _compose_text(val_df)
    test_texts = _compose_text(test_df)

    vector_pack = _vectorise(train_texts, val_texts, test_texts, feature_cfg, feature_type)
    model = build_classical_model(model_name, params)
    model.random_state = seed if hasattr(model, "random_state") else getattr(model, "random_state", None)  # type: ignore[attr-defined]
    model.fit(vector_pack.X_train, train_df["label"].to_numpy())
    val_pred = model.predict(vector_pack.X_val)
    val_scores = _scores_from_model(model, vector_pack.X_val)
    metrics = compute_binary_metrics(val_df["label"].to_numpy(), val_pred, val_scores)

    if mlflow_enabled:
        mlflow.log_params({f"model_{k}": v for k, v in params.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    # Fit on train + val
    combined_texts = pd.concat([train_texts, val_texts]).tolist()
    if feature_type == "hashing":
        final_vectorizer = vector_pack.vectorizer
        X_final = final_vectorizer.transform(combined_texts)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        final_vectorizer = TfidfVectorizer(**feature_cfg)
        X_final = final_vectorizer.fit_transform(combined_texts)
    y_final = pd.concat([train_df["label"], val_df["label"]]).to_numpy()
    final_model = build_classical_model(model_name, params)
    if hasattr(final_model, "random_state"):
        setattr(final_model, "random_state", seed)
    final_model.fit(X_final, y_final)

    artifact_path = artifacts_dir / model_name
    artifact_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_path / "model.pkl")
    joblib.dump(final_vectorizer, artifact_path / "vectorizer.pkl")
    config_payload = {
        "model": model_name,
        "params": params,
        "feature_type": feature_type,
        "feature_config": feature_cfg,
        "seed": seed,
        "preprocess": preprocess_cfg,
    }
    (artifact_path / "preproc_config.json").write_text(json.dumps(config_payload, indent=2))
    LOGGER.info("Saved classical model artifacts to %s", artifact_path)
    return artifact_path, metrics


def _train_transformer_wrapper(
    model_key: str,
    model_cfg: Dict[str, Any],
    splits: Dict[str, pd.DataFrame],
    artifacts_dir: Path,
    seed: int,
    mlflow_enabled: bool,
) -> Tuple[Path, Dict[str, float]]:
    train_df, val_df = splits["train"], splits["val"]
    output_dir = artifacts_dir / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_transformer(
        train_df,
        val_df,
        model_name=model_cfg.get("model_name", model_key),
        output_dir=output_dir,
        epochs=model_cfg.get("epochs", 2),
        batch_size=model_cfg.get("batch_size", 16),
        learning_rate=model_cfg.get("learning_rate", 5e-5),
        seed=seed,
    )
    trainer = result["trainer"]
    metrics = trainer.evaluate()
    if mlflow_enabled:
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
    return output_dir, metrics


def train_model(
    model_name: str,
    *,
    config_path: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[Path, Dict[str, float]]:
    config = load_config(config_path, overrides)
    global_cfg = config.get("paths", {})
    processed_dir = Path(global_cfg.get("processed_dir", "data/processed"))
    artifacts_dir = Path(global_cfg.get("artifacts_dir", "artifacts"))
    best_model_dir = Path(global_cfg.get("best_model_dir", "artifacts/best_model"))

    splits = load_splits(processed_dir)

    random_seed = config.get("preprocess", {}).get("seed", 42)
    np.random.seed(random_seed)
    check_random_state(random_seed)

    mlflow_cfg = config.get("mlflow", {})
    mlflow_enabled = mlflow_cfg.get("enable", True)
    if mlflow_enabled:
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        experiment = mlflow_cfg.get("experiment_name", "phishing-benchmark")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=model_name) if mlflow_enabled else _nullcontext():
        if mlflow_enabled:
            mlflow.log_param("model", model_name)
        if model_name in config.get("models", {}).get("classical", {}):
            params = config["models"]["classical"].get(model_name, {})
            feature_cfg = config.get("features", {}).get("tfidf", {"max_features": 50000, "ngram_range": (1, 2)})
            feature_type = "tfidf"
            artifact_path, metrics = _train_classical(
                model_name,
                params,
                feature_cfg,
                feature_type,
                config.get("preprocess", {}),
                splits,
                artifacts_dir,
                random_seed,
                mlflow_enabled,
            )
        elif model_name in config.get("models", {}).get("transformers", {}):
            model_cfg = config["models"]["transformers"][model_name]
            artifact_path, metrics = _train_transformer_wrapper(
                model_name,
                model_cfg,
                splits,
                artifacts_dir,
                random_seed,
                mlflow_enabled,
            )
        else:
            raise ValueError(f"Model '{model_name}' not found in configuration")

    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(artifact_path, best_model_dir)
    LOGGER.info("Copied %s to best model directory %s", artifact_path, best_model_dir)
    return artifact_path, metrics


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
