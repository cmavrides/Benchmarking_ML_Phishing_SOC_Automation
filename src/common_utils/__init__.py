"""Shared utilities for phishing-detection-suite."""

from .config import load_config
from .logging_conf import get_logger

__all__ = ["load_config", "get_logger"]
