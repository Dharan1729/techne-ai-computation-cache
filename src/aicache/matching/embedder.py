"""Thin wrapper around sentence-transformers for cosine-similarity matching.

Same model choice this author's other project (Kairos) used —
``sentence-transformers/all-MiniLM-L6-v2`` — for consistency with prior
findings: that repo's own benchmark (see ``semantic caching/docker-tests``
there) found a recall cliff for this exact model between similarity
threshold 0.80 and 0.85, which is exactly the kind of threshold-dependence
this project's semantic-cache baseline (§13 of the brief) is supposed to
test explicitly rather than assume.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Lazily loads the model on first use — importing this module should
    not require a network call or the model download."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def cosine_sim(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """corpus is (N, D), query is (D,), embeddings assumed pre-normalized."""
    if corpus.size == 0:
        return np.zeros(0)
    return corpus @ query


def nearest(query: np.ndarray, corpus: np.ndarray) -> tuple[int, float]:
    """Returns (index, similarity) of the best match, or (-1, -1.0) if corpus empty."""
    sims = cosine_sim(query, corpus)
    if sims.size == 0:
        return -1, -1.0
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])
