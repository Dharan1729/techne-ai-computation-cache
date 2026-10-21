"""Embedders the router can use. All expose ``encode(list[str]) -> (N, D)`` and
``encode_one(str) -> (D,)`` with L2-normalised rows, so a dot product is cosine
similarity.

* ``SentenceTransformerEmbedder``: the default, a local model. Needs the
  ``embeddings`` extra.
* ``OpenAICompatibleEmbedder``: any server that speaks ``POST /v1/embeddings``
  (OpenAI, Ollama, vLLM, and others). No local model.
* ``HashingEmbedder``: no dependencies and no download. Character n-gram
  hashing, so it only sees spelling overlap, not meaning. Each embedder needs
  its own ``min_similarity``: run ``aicache calibrate`` after switching.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "redis/langcache-embed-v1"


class EmbedderLike(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_one(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class SentenceTransformerEmbedder:
    """Loads the model on first use, so importing this costs nothing."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "the sentence-transformers embedder needs the 'embeddings' extra: "
                    "pip install 'ai-computation-cache[embeddings]'"
                ) from exc
            # Prefer the local cache. Loading normally makes network checks against the
            # Hugging Face Hub even for a cached model, and on a flaky network that can block
            # a restart indefinitely. Only reach out if the model is not on disk yet.
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._load().encode(list(texts), normalize_embeddings=True))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder:
    """Word and character-trigram hashing into ``dim`` buckets."""

    _WORD = re.compile(r"[a-z0-9%$+*/:.-]+")

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = self._WORD.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"^{w}$"
            feats += [f"c:{padded[i:i + 3]}" for i in range(max(1, len(padded) - 2))]
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for r, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                out[r, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
                out[r, (h >> 20) % self.dim] += 1.0
        return _normalise(out)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class OpenAICompatibleEmbedder:
    """Calls ``POST {base_url}/embeddings``. ``base_url`` should include ``/v1``."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout_s: float = 30.0, transport: Any = None) -> None:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                    timeout=timeout_s, transport=transport)
        self.model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.post("/embeddings", json={"model": self.model, "input": list(texts)})
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return _normalise(np.array([d["embedding"] for d in data], dtype=float))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def make_embedder(kind: str = "sentence-transformers", *, model: str | None = None,
                  base_url: str | None = None, api_key_env: str | None = None) -> EmbedderLike:
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model or DEFAULT_MODEL)
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "openai":
        if not base_url or not model:
            raise ValueError("embedder type 'openai' needs base_url and model")
        key = os.environ.get(api_key_env) if api_key_env else None
        return OpenAICompatibleEmbedder(base_url, model, key)
    raise ValueError(f"unknown embedder type {kind!r}; use sentence-transformers, hashing or openai")
