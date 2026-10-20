"""The four experimental conditions (brief §5 / §15 A-D).

Each function takes the same workload and LLM backend and returns one
``RequestResult`` per request, so the harness can run all four back to
back on identical inputs and compare directly. Each baseline owns exactly
one cache instance so requests are processed in workload order — cache
state (what's been seen so far) accumulates the way it would in a real
serving loop, which is what makes "does reuse show up as requests
accumulate" a meaningful thing to plot.
"""

from __future__ import annotations

import os
import time

from aicache.cache.computation import ComputationCache
from aicache.cache.exact import ExactCache
from aicache.cache.semantic import SemanticCache
from aicache.llm.base import LLMClient
from aicache.llm.pricing import estimate_cost_usd
from aicache.metrics.correctness import is_correct
from aicache.metrics.types import Outcome, RequestResult
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier

from workloads.types import WorkloadItem


def _task_hint(item: WorkloadItem) -> dict:
    return {"operation": item.task_type, "params": item.bound_params}


def _llm_result(
    item: WorkloadItem, llm: LLMClient, baseline: str, outcome: Outcome, extra: dict | None = None
) -> tuple[RequestResult, str]:
    resp = llm.complete(item.text, task_hint=_task_hint(item))
    cost = estimate_cost_usd(resp.model_name, resp.input_tokens, resp.output_tokens)
    correct = is_correct(resp.text, item.expected_answer)
    result = RequestResult(
        request_id=item.request_id,
        baseline=baseline,
        outcome=outcome,
        correct=correct,
        latency_ms=resp.latency_ms,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=cost,
        llm_called=True,
        task_type=item.task_type,
        extra=extra or {},
    )
    return result, resp.text


# --------------------------------------------------------------- Baseline A


def run_no_cache(workload: list[WorkloadItem], llm: LLMClient) -> list[RequestResult]:
    """Request -> LLM -> Answer, every time. The floor for cost/latency."""
    return [_llm_result(item, llm, "no_cache", Outcome.NO_CACHE)[0] for item in workload]


# --------------------------------------------------------------- Baseline B


def run_exact_cache(workload: list[WorkloadItem], llm: LLMClient) -> list[RequestResult]:
    cache = ExactCache()
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        cached = cache.lookup(item.text)
        lookup_ms = (time.perf_counter() - t0) * 1000

        if cached is not None:
            correct = is_correct(cached, item.expected_answer)
            results.append(
                RequestResult(
                    request_id=item.request_id,
                    baseline="exact_cache",
                    outcome=Outcome.HIT_VALID if correct else Outcome.HIT_INVALID,
                    correct=correct,
                    latency_ms=lookup_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    llm_called=False,
                    task_type=item.task_type,
                )
            )
            continue

        r, resp_text = _llm_result(item, llm, "exact_cache", Outcome.MISS)
        r.latency_ms += lookup_ms
        results.append(r)
        # Store exactly what the LLM said, so a future byte-identical
        # request reuses this same answer (and nothing else does).
        cache.store(item.text, resp_text)
    return results


# --------------------------------------------------------------- Baseline C


def run_semantic_cache(
    workload: list[WorkloadItem], llm: LLMClient, threshold: float = 0.75, embedder: Embedder | None = None
) -> list[RequestResult]:
    """Semantic response cache — reuses the nearest prior request's answer
    whenever similarity clears `threshold`, regardless of whether that
    prior request's *parameters* match this one's. This is the mechanism
    RQ2/RQ3 are about: measured here via `is_correct`, not assumed."""
    cache = SemanticCache(embedder=embedder or Embedder(), threshold=threshold)
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        lookup = cache.lookup(item.text)
        lookup_ms = (time.perf_counter() - t0) * 1000

        if lookup.hit:
            correct = is_correct(lookup.answer or "", item.expected_answer)
            results.append(
                RequestResult(
                    request_id=item.request_id,
                    baseline="semantic_cache",
                    outcome=Outcome.HIT_VALID if correct else Outcome.HIT_INVALID,
                    correct=correct,
                    latency_ms=lookup_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    llm_called=False,
                    task_type=item.task_type,
                    similarity=lookup.similarity,
                    extra={"matched_request": lookup.matched_request},
                )
            )
            continue

        resp = llm.complete(item.text, task_hint=_task_hint(item))
        cost = estimate_cost_usd(resp.model_name, resp.input_tokens, resp.output_tokens)
        correct = is_correct(resp.text, item.expected_answer)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="semantic_cache",
                outcome=Outcome.MISS,
                correct=correct,
                latency_ms=resp.latency_ms + lookup_ms,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=cost,
                llm_called=True,
                task_type=item.task_type,
                similarity=lookup.similarity,
            )
        )
        cache.store(item.text, resp.text)
    return results


# --------------------------------------------------------------- Baseline D


def run_computation_cache(
    workload: list[WorkloadItem],
    llm: LLMClient,
    identifier: TaskIdentifier | None = None,
) -> list[RequestResult]:
    """Identify -> retrieve reusable computation -> bind fresh params ->
    validate -> execute directly. No LLM call on a hit; falls back to the
    LLM whenever identification or binding can't be validated."""
    # AICACHE_NO_CUES=1 disables the lexical-cue guard, for the EXP-016 ablation only.
    cache = ComputationCache(
        identifier=identifier or TaskIdentifier(),
        use_lexical_cues=os.environ.get("AICACHE_NO_CUES") != "1",
    )
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        lookup = cache.lookup(item.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if lookup.hit:
            correct = is_correct(str(lookup.answer), item.expected_answer)
            results.append(
                RequestResult(
                    request_id=item.request_id,
                    baseline="computation_cache",
                    outcome=Outcome.HIT_VALID if correct else Outcome.HIT_INVALID,
                    correct=correct,
                    latency_ms=elapsed_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    llm_called=False,
                    task_type=lookup.task_type,
                    similarity=lookup.similarity,
                )
            )
            continue

        r, _ = _llm_result(
            item, llm, "computation_cache", Outcome.FALLBACK, extra={"reason": lookup.reason}
        )
        r.latency_ms += elapsed_ms
        r.similarity = lookup.similarity
        results.append(r)
    return results
