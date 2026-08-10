"""Level 3's five conditions (brief: baselines A-D primary, E as a
freshness ablation — see docs/level3.md).

Structural difference from Level 1/2 (``benchmarks/baselines.py``):
every condition here needs the *fetch* step, not just the computation
cache. A, B, and C all "ask an LLM" on a miss — but the LLM cannot
possibly answer correctly without the current live data (the question
is about a specific customer/transaction the model has no way to know
about on its own), so every LLM call is preceded by a fetch, exactly the
way a real RAG/tool-using agent would need to look up the relevant
records before reasoning over them. This is not a caching optimization
for A/B/C — it is a prerequisite for producing *any* plausible answer,
cache or no cache. What differs between A/B/C/D/E is what gets reused:
A/B/C potentially reuse a full text *answer*; D never reuses anything
but the *plan* (always fetches fresh); E deliberately reuses fetched
*data* too, as a controlled demonstration of why D doesn't.
"""

from __future__ import annotations

import time

from aicache.cache.exact import ExactCache
from aicache.cache.semantic import SemanticCache
from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import EnterpriseDB
from aicache.llm.base import LLMClient
from aicache.llm.pricing import estimate_cost_usd
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.types import Outcome, RequestResult
from aicache.metrics.workflow_correctness import is_workflow_correct

from workloads.level3.types import Level3WorkloadItem


def _fetch_for_item(item: Level3WorkloadItem, db: EnterpriseDB) -> tuple[dict | None, float, str | None]:
    spec = WORKFLOW_TASK_REGISTRY[item.task_type]
    outcome = spec.fetch(item.entities, item.numbers, db)
    return outcome.values, outcome.latency_ms, outcome.error


def _task_hint(item: Level3WorkloadItem, fetched: dict) -> dict:
    return {
        "operation": item.task_type,
        "kind": "workflow",
        "params": {**item.entities, **item.numbers, **fetched},
    }


def _ask_llm(
    item: Level3WorkloadItem, llm: LLMClient, db: EnterpriseDB
) -> tuple[str, float, int, int, float, float]:
    """Fetch + call the LLM. Returns (served_text, fetch_ms, input_tokens,
    output_tokens, llm_latency_ms, cost_usd)."""
    fetched, fetch_ms, error = _fetch_for_item(item, db)
    if fetched is None:
        # Shouldn't happen for workload items generated as valid — kept
        # as a real, reported branch rather than an assertion, since
        # "the fetch itself can fail" is a real system property.
        return f"[fetch failed: {error}]", fetch_ms, 0, 0, 0.0, 0.0
    resp = llm.complete(item.text, task_hint=_task_hint(item, fetched))
    cost = estimate_cost_usd(resp.model_name, resp.input_tokens, resp.output_tokens)
    return resp.text, fetch_ms, resp.input_tokens, resp.output_tokens, resp.latency_ms, cost


# --------------------------------------------------------------- Baseline A


def run_no_cache(
    workload: list[Level3WorkloadItem], llm: LLMClient, db: EnterpriseDB
) -> list[RequestResult]:
    results = []
    for item in workload:
        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_workflow_correct(item.task_type, text, item.expected_result)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="no_cache",
                outcome=Outcome.NO_CACHE,
                correct=correct,
                latency_ms=fetch_ms + llm_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                llm_called=True,
                task_type=item.task_type,
                extra={"fetch_ms": fetch_ms, "llm_ms": llm_ms},
            )
        )
    return results


# --------------------------------------------------------------- Baseline B


def run_exact_cache(
    workload: list[Level3WorkloadItem], llm: LLMClient, db: EnterpriseDB
) -> list[RequestResult]:
    cache = ExactCache()
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        cached = cache.lookup(item.text)
        lookup_ms = (time.perf_counter() - t0) * 1000

        if cached is not None:
            correct = is_workflow_correct(item.task_type, cached, item.expected_result)
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

        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_workflow_correct(item.task_type, text, item.expected_result)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="exact_cache",
                outcome=Outcome.MISS,
                correct=correct,
                latency_ms=lookup_ms + fetch_ms + llm_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                llm_called=True,
                task_type=item.task_type,
                extra={"fetch_ms": fetch_ms, "llm_ms": llm_ms},
            )
        )
        cache.store(item.text, text)
    return results


# --------------------------------------------------------------- Baseline C


def run_semantic_cache(
    workload: list[Level3WorkloadItem],
    llm: LLMClient,
    db: EnterpriseDB,
    threshold: float = 0.75,
    embedder: Embedder | None = None,
) -> list[RequestResult]:
    """The higher-stakes version of Level 1/2's RQ2/RQ3 mechanism: a
    semantic hit reuses whatever answer was cached for the nearest prior
    *text*, regardless of which customer/transaction that answer was
    actually about. Reusing "$500 refund approved" for a different
    customer_id is not just a wrong number — it's a wrong decision about
    a specific person's money, the vivid version of the failure mode
    Level 1/2 could only demonstrate with plain arithmetic."""
    cache = SemanticCache(embedder=embedder or Embedder(), threshold=threshold)
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        lookup = cache.lookup(item.text)
        lookup_ms = (time.perf_counter() - t0) * 1000

        if lookup.hit:
            correct = is_workflow_correct(item.task_type, lookup.answer or "", item.expected_result)
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

        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_workflow_correct(item.task_type, text, item.expected_result)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="semantic_cache",
                outcome=Outcome.MISS,
                correct=correct,
                latency_ms=lookup_ms + fetch_ms + llm_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                llm_called=True,
                task_type=item.task_type,
                similarity=lookup.similarity,
                extra={"fetch_ms": fetch_ms, "llm_ms": llm_ms},
            )
        )
        cache.store(item.text, text)
    return results


# ----------------------------------------------------- Baselines D and E


def run_workflow_computation_cache(
    workload: list[Level3WorkloadItem],
    llm: LLMClient,
    db: EnterpriseDB,
    identifier: TaskIdentifier | None = None,
    use_stale_cache: bool = False,
) -> list[RequestResult]:
    """D (``use_stale_cache=False``) or E (``use_stale_cache=True``) —
    see aicache/cache/workflow_computation.py's module docstring for
    what the flag means. Baseline name is set by the caller
    (``benchmarks/level3/run_experiment.py``) after this returns, same
    pattern as the Level 1/2 semantic-cache threshold sweep re-labeling
    its own results."""
    cache = WorkflowComputationCache(
        db, identifier or TaskIdentifier(registry=WORKFLOW_TASK_REGISTRY), use_stale_cache
    )
    results: list[RequestResult] = []
    for item in workload:
        lookup = cache.lookup(item.text)
        total_ms = lookup.latency.total_ms

        if lookup.hit:
            correct = is_workflow_correct(item.task_type, lookup.served_text or "", item.expected_result)
            results.append(
                RequestResult(
                    request_id=item.request_id,
                    baseline="workflow_computation_cache",
                    outcome=Outcome.HIT_VALID if correct else Outcome.HIT_INVALID,
                    correct=correct,
                    latency_ms=total_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    llm_called=False,
                    task_type=lookup.task_type,
                    similarity=lookup.similarity,
                    extra={
                        "used_stale_fetch": lookup.used_stale_fetch,
                        "identify_ms": lookup.latency.identify_ms,
                        "extract_ms": lookup.latency.extract_ms,
                        "validate_ms": lookup.latency.validate_ms,
                        "fetch_ms": lookup.latency.fetch_ms,
                        "execute_ms": lookup.latency.execute_ms,
                    },
                )
            )
            continue

        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_workflow_correct(item.task_type, text, item.expected_result)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="workflow_computation_cache",
                outcome=Outcome.FALLBACK,
                correct=correct,
                latency_ms=total_ms + fetch_ms + llm_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                llm_called=True,
                task_type=lookup.task_type,
                similarity=lookup.similarity,
                extra={"reason": lookup.reason, "fetch_ms": fetch_ms, "llm_ms": llm_ms},
            )
        )
    return results
