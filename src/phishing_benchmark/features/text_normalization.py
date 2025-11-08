"""Feature engineering helpers for text normalization."""

from __future__ import annotations

from typing import Iterable, List

from ..data.cleaning import normalize_text


def normalize_corpus(
    texts: Iterable[str],
    lower: bool = True,
    replace_urls: bool = True,
    replace_emails: bool = True,
    replace_numbers: bool = False,
) -> List[str]:
    """Normalize an iterable of texts and return a list."""

    return [
        normalize_text(
            text=text,
            lower=lower,
            replace_urls=replace_urls,
            replace_emails=replace_emails,
            replace_numbers=replace_numbers,
        )
        for text in texts
    ]
