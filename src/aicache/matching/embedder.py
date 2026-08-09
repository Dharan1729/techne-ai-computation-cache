"""Thin wrapper around sentence-transformers for cosine-similarity matching.

Default model: ``redis/langcache-embed-v1`` — a model Redis trained
specifically for semantic caching (not a generic sentence-similarity
model repurposed for this task), free to download and run entirely
locally (no API key, no network call after the first download). Chosen
over a generic model — this project initially used
``sentence-transformers/all-MiniLM-L6-v2`` (still available via the
``model_name`` param) — after a direct comparison: on a genuine
paraphrase pair from this project's own workload, MiniLM scored cosine
similarity in the 0.3-0.5 range (see docs/experiment-log.md, H6's
discussion of the identification-threshold tension that caused), while
langcache-embed-v1 scored 0.77 on the same pair. RedisVL's own
`SemanticCache` (see `benchmarks/external/redisvl_adapter.py`) defaults
to this same model, which is also what makes the RedisVL comparison in
`docs/experiment-log.md` an apples-to-apples "caching algorithm" test
rather than a confounded "algorithm + embedding model" test — see that
adapter's module docstring for how GPTCache's own default (a generic
SBERT model) differs, and why that difference is called out explicitly
rather than papered over.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "redis/langcache-embed-v1"


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
