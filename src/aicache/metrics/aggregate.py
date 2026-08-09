"""Aggregate per-request results into a per-baseline summary.

Deliberately reports more than hit rate (brief §17 / H7: hit rate alone
doesn't correlate with inference-cost savings). Token/cost/latency
figures live alongside hit-rate figures so a reader can see, e.g., a
baseline with a mediocre hit rate but strong cost savings (or the
reverse) instead of just the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Outcome, RequestResult


@dataclass
class BaselineSummary:
    baseline: str
    request_count: int
    hit_rate: float
    valid_hit_rate: float
    invalid_hit_rate: float  # aka false-hit rate
    miss_rate: float
    fallback_rate: float
    llm_call_rate: float
    correctness_rate: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    outcome_counts: dict[str, int] = field(default_factory=dict)


def summarize(results: list[RequestResult], baseline: str | None = None) -> BaselineSummary:
    rows = [r for r in results if baseline is None or r.baseline == baseline]
    n = len(rows)
    if n == 0:
        raise ValueError("no results to summarize")

    name = baseline or rows[0].baseline
    outcome_counts: dict[str, int] = {}
    for r in rows:
        outcome_counts[r.outcome.value] = outcome_counts.get(r.outcome.value, 0) + 1

    def frac(outcome: Outcome) -> float:
        return outcome_counts.get(outcome.value, 0) / n

    hit_valid = frac(Outcome.HIT_VALID)
    hit_invalid = frac(Outcome.HIT_INVALID)

    latencies = np.array([r.latency_ms for r in rows])
    costs = np.array([r.cost_usd for r in rows])

    return BaselineSummary(
        baseline=name,
        request_count=n,
        hit_rate=hit_valid + hit_invalid,
        valid_hit_rate=hit_valid,
        invalid_hit_rate=hit_invalid,
        miss_rate=frac(Outcome.MISS) + frac(Outcome.NO_CACHE),
        fallback_rate=frac(Outcome.FALLBACK),
        llm_call_rate=sum(1 for r in rows if r.llm_called) / n,
        correctness_rate=sum(1 for r in rows if r.correct) / n,
        total_input_tokens=sum(r.input_tokens for r in rows),
        total_output_tokens=sum(r.output_tokens for r in rows),
        total_tokens=sum(r.input_tokens + r.output_tokens for r in rows),
        total_cost_usd=float(costs.sum()),
        avg_cost_usd=float(costs.mean()),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        outcome_counts=outcome_counts,
    )


def summarize_all(results: list[RequestResult]) -> dict[str, BaselineSummary]:
    baselines = sorted({r.baseline for r in results})
    return {b: summarize(results, b) for b in baselines}
