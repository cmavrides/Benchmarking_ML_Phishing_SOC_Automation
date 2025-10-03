"""Schema definitions for the unified phishing dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class UnifiedRecord:
    id: str
    source_dataset: str
    source_type: str
    subject: str
    body: str
    body_clean: str
    body_is_html: bool
    label: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_dataset": self.source_dataset,
            "source_type": self.source_type,
            "subject": self.subject,
            "body": self.body,
            "body_clean": self.body_clean,
            "body_is_html": self.body_is_html,
            "label": int(self.label),
        }


REQUIRED_COLUMNS = {
    "id",
    "source_dataset",
    "source_type",
    "subject",
    "body",
    "body_clean",
    "body_is_html",
    "label",
}


def validate_schema(record: Dict[str, Any]) -> None:
    missing = REQUIRED_COLUMNS.difference(record)
    if missing:
        raise ValueError(f"Record missing required fields: {sorted(missing)}")
