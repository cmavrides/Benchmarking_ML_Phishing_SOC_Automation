"""Transformer fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from common_utils.logging_conf import get_logger

LOGGER = get_logger(__name__)


def _build_text(df: pd.DataFrame) -> pd.Series:
    return (df["subject"].fillna("") + " \n" + df["body_clean"].fillna("")).str.strip()


def _to_dataset(df: pd.DataFrame, tokenizer, max_length: int = 512) -> Dataset:
    texts = _build_text(df).tolist()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = Dataset.from_dict({"text": texts, "label": df["label"].tolist()})
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = _to_dataset(train_df, tokenizer)
    val_dataset = _to_dataset(val_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}
