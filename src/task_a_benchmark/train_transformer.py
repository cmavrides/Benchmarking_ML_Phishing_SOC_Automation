"""Fine-tune DistilBERT for phishing detection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.common.metrics import compute_classification_metrics, print_metrics
from src.common.utils import save_json, set_random_seed

PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/distilbert")
RESULTS_PATH = Path("reports/results/distilbert_metrics.json")
MODEL_NAME = "distilbert-base-uncased"


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run preprocess.py first.")
    return pd.read_csv(path)


def _tokenize_function(tokenizer, examples):
    return tokenizer(examples["clean_text"], truncation=True)


def train_transformer(seed: int = 42, num_train_epochs: float = 1.0, batch_size: int = 8) -> Dict[str, float]:
    set_random_seed(seed)
    torch.manual_seed(seed)

    train_df = _load_split("train")
    val_df = _load_split("val")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["clean_text", "label"]])

    train_dataset = train_dataset.map(lambda x: _tokenize_function(tokenizer, x), batched=True)
    val_dataset = val_dataset.map(lambda x: _tokenize_function(tokenizer, x), batched=True)

    for column in ["clean_text", "__index_level_0__"]:
        if column in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(column)
        if column in val_dataset.column_names:
            val_dataset = val_dataset.remove_columns(column)
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return compute_classification_metrics(labels, preds, probs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print_metrics("distilbert", metrics)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(ARTIFACT_DIR)
    tokenizer.save_pretrained(ARTIFACT_DIR)
    save_json(metrics, RESULTS_PATH)

    return metrics


if __name__ == "__main__":
    train_transformer()
