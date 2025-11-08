"""Simple deep learning baselines using PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils import set_seed


class TextDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class BiRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - neural net forward
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded)
        pooled, _ = torch.max(outputs, dim=1)
        return self.fc(self.dropout(pooled))


@dataclass
class TrainingConfig:
    vocab_size: int
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq
    return padded, torch.stack(labels)


def train_model(
    model: nn.Module,
    train_data: List[List[int]],
    train_labels: List[int],
    val_data: List[List[int]] | None,
    val_labels: List[int] | None,
    config: TrainingConfig,
) -> nn.Module:
    """Train a simple RNN classifier."""

    set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)

    train_dataset = TextDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for _ in range(config.epochs):  # pragma: no cover - training loop (heavy)
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return model
