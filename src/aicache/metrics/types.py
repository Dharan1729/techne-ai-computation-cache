"""Per-request outcome taxonomy (brief §12) and result record.

    NO_CACHE     — baseline A: every request calls the LLM, by design.
    HIT_VALID    — a cache/computation reuse whose answer matches ground truth.
    HIT_INVALID  — a reuse whose answer does NOT match ground truth (the
                   failure mode §12/§13 exist to measure — most relevant to
                   the semantic-response-cache baseline under changing
                   parameters, the central RQ3 concern).
    MISS         — no reusable entry found; fell through to a fresh LLM call
                   (exact/semantic cache) or fresh direct computation
                   (computation cache with successful identification).
    FALLBACK     — the computation cache specifically: task identification
                   or parameter binding/validation failed, so the request
                   fell back to a full LLM call rather than risk an
                   unvalidated execution.

"Correct" (on the RequestResult) is independent of "valid hit": it's
whether the served answer matches the workload's ground truth, checked
for every outcome, not just hits — this is what lets false-hit rate and
overall correctness be measured directly rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    NO_CACHE = "no_cache"
    HIT_VALID = "hit_valid"
    HIT_INVALID = "hit_invalid"
    MISS = "miss"
    FALLBACK = "fallback"


@dataclass
class RequestResult:
    request_id: str
    baseline: str
    outcome: Outcome
    correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    llm_called: bool
    task_type: str | None = None
    similarity: float | None = None
    extra: dict = field(default_factory=dict)
