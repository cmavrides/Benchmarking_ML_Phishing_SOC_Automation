"""Transformer fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..utils import set_seed


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    output_dir: str = "checkpoints/transformers"
    seed: int = 42
    fp16: bool = False


def prepare_dataset(texts, labels, tokenizer, max_length: int = 512) -> Dataset:
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encodings["label"] = list(labels)
    return Dataset.from_dict(encodings)


def train_transformer(
    train_texts,
    train_labels,
    eval_texts,
    eval_labels,
    config: TransformerConfig,
) -> Dict[str, float]:
    """Fine-tune a Hugging Face transformer and return evaluation metrics."""

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )

    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer)
    eval_dataset = prepare_dataset(eval_texts, eval_labels, tokenizer)

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=config.fp16,
        report_to=["none"],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
    metrics = trainer.evaluate()
    return metrics
