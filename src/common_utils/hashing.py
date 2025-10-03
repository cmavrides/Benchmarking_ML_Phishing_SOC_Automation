"""Hashing helpers for deterministic identifiers."""

from __future__ import annotations

import hashlib
from typing import Any


def stable_id(*fields: Any) -> str:
    """Generate a deterministic SHA-1 hash from the provided fields."""

    joined = "||".join(str(field or "") for field in fields)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
