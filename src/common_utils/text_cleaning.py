"""Text cleaning utilities shared between tasks."""

from __future__ import annotations

import html
import re
from typing import Tuple

from bs4 import BeautifulSoup

WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


def strip_html(text: str) -> Tuple[str, bool]:
    """Strip HTML content, removing scripts/styles and collapsing whitespace."""

    if not text:
        return "", False

    soup = BeautifulSoup(text, "html.parser")
    was_html = bool(soup.find()) and bool(soup.find_all())

    for element in soup(["script", "style"]):
        element.decompose()

    stripped = soup.get_text(separator=" ")
    stripped = html.unescape(stripped)
    stripped = WHITESPACE_RE.sub(" ", stripped).strip()

    return stripped, was_html


def normalize(
    text: str,
    *,
    lower: bool = True,
    url_token: str = "<URL>",
    num_token: str = "<NUM>",
    email_token: str = "<EMAIL>",
) -> str:
    """Normalise text by optionally lowercasing and masking entities."""

    if not text:
        return ""

    processed = text
    processed = URL_RE.sub(url_token, processed)
    processed = EMAIL_RE.sub(email_token, processed)
    processed = re.sub(r"\d+", num_token, processed)

    processed = WHITESPACE_RE.sub(" ", processed).strip()

    if lower:
        processed = processed.lower()

    return processed
