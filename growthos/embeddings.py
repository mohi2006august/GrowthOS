"""Text embeddings behind a small interface.

The default `HashingEmbedder` needs no model download, network call or paid service: it
hashes word and character n-grams into a fixed-size vector. It captures lexical overlap, not
deep semantics, which is why retrieval is hybrid (BM25 + dense). Swap in a learned model by
implementing `Embedder`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was "
    "were will with we our you your they them not but if so than then into over per".split()
)
_SUFFIXES = ("ing", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and a light suffix strip."""
    return [_stem(word) for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised row per text."""
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 512, char_ngrams: tuple[int, ...] = (3, 4)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._char_ngrams = char_ngrams

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                index, sign = _slot(feature, self.dim)
                # Sublinear term frequency so one repeated word cannot dominate.
                matrix[row, index] += sign * (1 + math.log(count))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms > 0, norms, 1.0)

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features: Counter[str] = Counter(f"w:{token}" for token in tokens)
        features.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        for token in tokens:
            padded = f"^{token}$"
            for n in self._char_ngrams:
                # Character n-grams carry half weight: they add fuzzy matching, not meaning.
                for start in range(len(padded) - n + 1):
                    features[f"c:{padded[start : start + n]}"] += 0.5
        return features


@lru_cache(maxsize=200_000)
def _slot(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of `matrix` with `vector` (rows are pre-normalised)."""
    return matrix @ vector
