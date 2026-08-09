"""Baseline C: semantic response cache — the credible baseline the brief's
§13 insists on (embeddings, configurable threshold, response storage,
metadata, similarity hit/miss). No exact-match fast path here deliberately
— that comparison is baseline B; keeping this baseline purely semantic
makes the RQ2/RQ3 contrast (semantic reuse vs. its failure under
parameter variation) legible on its own.

This is where RQ2/RQ3 actually get tested: a semantic hit reuses whatever
answer was cached for the nearest prior request, regardless of whether
that prior request's numeric parameters match this one's. Under
parameter-varying workloads that's exactly the false-hit mechanism H2
predicts — measured, not assumed, by comparing the reused answer against
this request's own ground truth in the benchmark harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..matching.embedder import Embedder, nearest


@dataclass
class SemanticLookupResult:
    hit: bool
    answer: str | None = None
    similarity: float = -1.0
    matched_request: str | None = None


@dataclass
class _Entry:
    text: str
    vector: np.ndarray
    answer: str


class SemanticCache:
    def __init__(self, embedder: Embedder | None = None, threshold: float = 0.75) -> None:
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self._entries: list[_Entry] = []

    def lookup(self, text: str) -> SemanticLookupResult:
        if not self._entries:
            return SemanticLookupResult(hit=False)
        query = self.embedder.encode_one(text)
        corpus = np.stack([e.vector for e in self._entries])
        idx, sim = nearest(query, corpus)
        if idx == -1 or sim < self.threshold:
            return SemanticLookupResult(hit=False, similarity=sim)
        entry = self._entries[idx]
        return SemanticLookupResult(
            hit=True, answer=entry.answer, similarity=sim, matched_request=entry.text
        )

    def store(self, text: str, answer: str) -> None:
        vector = self.embedder.encode_one(text)
        self._entries.append(_Entry(text=text, vector=vector, answer=answer))

    def __len__(self) -> int:
        return len(self._entries)
