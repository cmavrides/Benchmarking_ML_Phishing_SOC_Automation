"""Logging helpers for consistent formatting across the project."""

from __future__ import annotations

import logging
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, logger: Optional[str] = None) -> logging.Logger:
    """Configure root logging and return a logger instance.

    Parameters
    ----------
    level:
        Logging level to set (defaults to :data:`logging.INFO`).
    logger:
        Optional logger name. If omitted, the root logger is configured.
    """

    logging.basicConfig(level=level, format=_LOG_FORMAT)
    return logging.getLogger(logger)
