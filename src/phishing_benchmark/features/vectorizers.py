"""Vectorization utilities for text features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class VectorizerBundle:
    """Container for fitted vectorizer and transformation helpers."""

    vectorizer: TfidfVectorizer

    def fit_transform(self, texts: Iterable[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]):
        return self.vectorizer.transform(texts)


def get_tfidf(
    max_features: int = 100_000,
    ngram_range: Tuple[int, int] = (1, 2),
    analyzer: str = "word",
    **kwargs,
) -> VectorizerBundle:
    """Construct a TF-IDF vectorizer with sensible defaults."""

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer=analyzer,
        **kwargs,
    )
    return VectorizerBundle(vectorizer=vectorizer)
