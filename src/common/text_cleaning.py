"""Utility functions for cleaning and normalizing phishing email text."""
from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: Optional[str]) -> str:
    """Remove HTML tags from a string using BeautifulSoup.

    Args:
        text: Raw text that may contain HTML markup.

    Returns:
        The text content with HTML tags stripped. Empty string is returned when
        ``text`` is ``None``.
    """

    if not text:
        return ""

    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ").strip()


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by lowercasing and replacing common tokens.

    The function performs the following steps:

    1. Convert to lowercase.
    2. Replace URLs, email addresses, and numbers with dedicated tokens.
    3. Collapse repeated whitespace.

    Args:
        text: Input string to normalize.

    Returns:
        A normalized string suitable for downstream modeling.
    """

    if not text:
        return ""

    text = text.lower()
    text = _URL_RE.sub(" <URL> ", text)
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    text = _NUMBER_RE.sub(" <NUMBER> ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


__all__ = ["strip_html", "normalize_text"]
