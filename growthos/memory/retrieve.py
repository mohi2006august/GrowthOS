"""Hybrid retrieval over experiment memory (FR-02).

Sparse BM25 and dense cosine rankings are merged with Reciprocal Rank Fusion, so a document
that either shares the query's exact terms or sits close in embedding space can surface.
Every hit carries a citation that proposals show to the approver.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection

import numpy as np
from pydantic import BaseModel

from growthos.embeddings import Embedder, cosine, tokenize
from growthos.memory.experiment_store import DocKind, ExperimentNotFound, ExperimentStore, MemoryDoc, outcome_note
from growthos.models import Citation

_RRF_K = 60
_BM25_K1 = 1.5
_BM25_B = 0.75
_SNIPPET_CHARS = 220


class RetrievedDoc(BaseModel):
    doc: MemoryDoc
    score: float
    # Cosine similarity to the query, clipped to [0, 1]; used as evidence weight downstream.
    similarity: float
    dense_rank: int
    sparse_rank: int | None
    citation: Citation


class HybridRetriever:
    def __init__(self, store: ExperimentStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder
        self._version: str | None = None
        self._docs: list[MemoryDoc] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: _Bm25 | None = None

    def search(self, query: str, k: int = 5, kinds: Collection[DocKind] | None = None) -> list[RetrievedDoc]:
        if k <= 0:
            raise ValueError("k must be positive")
        self._refresh()
        if not self._docs or not query.strip():
            return []
        assert self._bm25 is not None

        allowed = [i for i, doc in enumerate(self._docs) if kinds is None or doc.kind in kinds]
        if not allowed:
            return []
        similarities = cosine(self._matrix, self._embedder.embed([query])[0])
        sparse = self._bm25.scores(tokenize(query))

        dense_order = sorted(allowed, key=lambda i: -similarities[i])
        sparse_order = sorted((i for i in allowed if sparse[i] > 0), key=lambda i: -sparse[i])
        dense_rank = {i: rank for rank, i in enumerate(dense_order, start=1)}
        sparse_rank = {i: rank for rank, i in enumerate(sparse_order, start=1)}

        fused = {
            i: 1 / (_RRF_K + dense_rank[i]) + (1 / (_RRF_K + sparse_rank[i]) if i in sparse_rank else 0.0)
            for i in allowed
        }
        top = sorted(allowed, key=lambda i: (-fused[i], self._docs[i].id))[:k]
        return [
            RetrievedDoc(
                doc=self._docs[i],
                score=fused[i],
                similarity=float(np.clip(similarities[i], 0.0, 1.0)),
                dense_rank=dense_rank[i],
                sparse_rank=sparse_rank.get(i),
                citation=self._citation(self._docs[i]),
            )
            for i in top
        ]

    def _refresh(self) -> None:
        version = self._store.index_version()
        if version == self._version:
            return
        self._docs, self._matrix = self._store.docs()
        self._bm25 = _Bm25([tokenize(f"{doc.title} {doc.text}") for doc in self._docs])
        self._version = version

    def _citation(self, doc: MemoryDoc) -> Citation:
        note = _snippet(doc.text)
        if doc.kind == "experiment" and doc.experiment_id:
            try:
                note = outcome_note(self._store.get(doc.experiment_id))
            except ExperimentNotFound:
                pass
        return Citation(doc_id=doc.id, experiment_id=doc.experiment_id, title=doc.title, note=note)


class _Bm25:
    def __init__(self, corpus: list[list[str]]) -> None:
        self._term_counts = [Counter(tokens) for tokens in corpus]
        self._lengths = [len(tokens) for tokens in corpus]
        self._average_length = sum(self._lengths) / len(corpus) if corpus else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        n = len(corpus)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self._term_counts)
        for term in set(query_tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self._term_counts):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                norm = 1 - _BM25_B + _BM25_B * self._lengths[i] / self._average_length
                scores[i] += idf * tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * norm)
        return scores


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[: _SNIPPET_CHARS - 1].rstrip() + "…"
