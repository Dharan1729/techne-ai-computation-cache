"""Baseline B: exact-match response cache.

REQUEST -> EXACT SAME REQUEST -> REUSE RESPONSE. Included as the floor
baseline (RQ1) precisely because it's expected to do almost nothing on a
workload of paraphrased, parameter-varying requests — that gap is the
motivation for semantic caching in the first place.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class ExactCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def lookup(self, text: str) -> str | None:
        return self._store.get(_normalize(text))

    def store(self, text: str, answer: str) -> None:
        self._store[_normalize(text)] = answer

    def __len__(self) -> int:
        return len(self._store)
