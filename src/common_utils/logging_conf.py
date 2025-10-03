"""Logging configuration helper."""

from __future__ import annotations

import logging
from typing import Optional

_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with consistent formatting."""

    if name is None:
        name = "phishing-detection-suite"

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger
