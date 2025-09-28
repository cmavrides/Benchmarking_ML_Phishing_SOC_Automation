"""Configuration primitives for the phishing benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(slots=True)
class DefaultConfig:
    """Default project configuration.

    Attributes
    ----------
    data_dir:
        Root path for dataset artifacts.
    raw_dir:
        Directory containing raw downloads.
    interim_dir:
        Directory containing cleaned but unsplit data.
    processed_dir:
        Directory containing train/val/test splits.
    reports_dir:
        Directory for evaluation reports and leaderboard files.
    random_seed:
        Global random seed used across modules.
    split_config:
        Default split fractions for validation and test sets.
    models:
        Baseline models executed by the `run_all` CLI.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    random_seed: int = 42
    split_config: Dict[str, float] = field(
        default_factory=lambda: {"val_size": 0.1, "test_size": 0.1}
    )
    models: List[str] = field(
        default_factory=lambda: ["lr", "svm", "nb", "rf", "xgb", "bilstm", "distilbert"]
    )
    tfidf_params: Dict[str, object] = field(
        default_factory=lambda: {"max_features": 100_000, "ngram_range": (1, 2)}
    )
    mlflow_tracking_uri: str = "mlruns"

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.project_root / "reports"
        self.cache_dir = self.project_root / ".cache"

    def dataset_paths(self) -> Dict[str, Path]:
        """Return commonly used dataset paths."""

        return {
            "raw": self.raw_dir,
            "interim": self.interim_dir,
            "processed": self.processed_dir,
        }

    def split_fractions(self) -> Tuple[float, float]:
        """Return validation and test fractions."""

        return (
            float(self.split_config.get("val_size", 0.1)),
            float(self.split_config.get("test_size", 0.1)),
        )
