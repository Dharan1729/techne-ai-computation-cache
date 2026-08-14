"""A generic memoize-with-expiry cache — deliberately modeled on
LangGraph's real ``CachePolicy(ttl=...)`` mechanism, not invented from
scratch. Verified directly against the actual ``langgraph`` library
(see ``docs/differentiation.md`` §3): a node hashes its input, memoizes
its output, and returns the memoized value until the TTL elapses,
regardless of whether the underlying data changed in the meantime.

This is Level 4's "graduated freshness" primitive — distinct from
Level 3's binary always-fresh/memoized-forever split
(``aicache/cache/workflow_computation.py``). Some sub-fetches (a
transaction's live status) must never be cached at all; others (a
batch-computed fraud score, ``aicache/enterprise/db.py``) are
legitimately safe to reuse for a bounded window in a real system — a
TTL cache is the right primitive for the second case, not the first,
and Level 4's whole point is quantifying exactly what that bound costs
in correctness as the underlying data's own volatility changes (see
``docs/level4.md``).

``now_fn`` is injectable so experiments can control time directly
(instant TTL expiry in a test, or a fast-forwarded clock in a sweep)
rather than requiring the test suite to actually sleep for real
seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable


@dataclass
class TTLCache:
    ttl_seconds: float | None  # None = memoize forever (Level 3's E, as a special case)
    now_fn: Callable[[], float] = time.monotonic
    _store: dict[Hashable, tuple[float, Any]] = field(default_factory=dict)

    def get_or_compute(self, key: Hashable, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
        """Returns (value, was_cache_hit)."""
        now = self.now_fn()
        if key in self._store:
            stored_at, value = self._store[key]
            if self.ttl_seconds is None or (now - stored_at) <= self.ttl_seconds:
                return value, True
        value = compute_fn()
        self._store[key] = (now, value)
        return value, False

    def clear(self) -> None:
        self._store.clear()
