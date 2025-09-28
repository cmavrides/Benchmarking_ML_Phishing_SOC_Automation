"""Utilities for cleaning phishing datasets."""

from __future__ import annotations

import html
import re
from typing import Iterable, Iterator, Tuple

from bs4 import BeautifulSoup


_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_URL_PATTERN = re.compile(
    r"((?:https?://|www\.)[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)",
    flags=re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"\b\d+\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_html(text: str) -> Tuple[str, bool]:
    """Strip HTML tags and scripts from the provided text."""

    if not text:
        return "", False

    if "<" not in text and ">" not in text:
        return text, False

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    cleaned = soup.get_text(separator=" ")
    cleaned = html.unescape(cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned, True


def normalize_text(
    text: str,
    *,
    lower: bool = True,
    replace_urls: bool = True,
    replace_emails: bool = True,
    replace_numbers: bool = True,
) -> str:
    """Normalize text by standardizing case and masking entities."""

    text = text or ""
    if lower:
        text = text.lower()

    if replace_emails:
        text = _EMAIL_PATTERN.sub(" <email> ", text)
    if replace_urls:
        text = _URL_PATTERN.sub(" <url> ", text)
    if replace_numbers:
        text = _NUMBER_PATTERN.sub(" <number> ", text)

    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def filter_by_length(
    texts: Iterable[str], *, min_length: int = 0, max_length: int | None = None
) -> Iterator[bool]:
    """Yield booleans selecting texts within the specified length bounds."""

    for text in texts:
        length = len(text or "")
        if length < min_length:
            yield False
            continue
        if max_length is not None and length > max_length:
            yield False
            continue
        yield True
