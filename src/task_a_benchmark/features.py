"""Feature builders for Task A models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


@dataclass
class FeaturePack:
    vectorizer: object
    X_train: sparse.spmatrix
    X_val: sparse.spmatrix
    X_test: sparse.spmatrix


def build_tfidf(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    config: Dict[str, object],
) -> FeaturePack:
    vectorizer = TfidfVectorizer(**config)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return FeaturePack(vectorizer, X_train, X_val, X_test)


def build_hashing(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    config: Dict[str, object],
) -> FeaturePack:
    vectorizer = HashingVectorizer(**config)
    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return FeaturePack(vectorizer, X_train, X_val, X_test)
