"""Level 4's baselines — A/C carried forward from Level 3 unchanged in
mechanism (only pointed at the new task's rendering/correctness
modules), plus the two Level 4-specific conditions:

    F — primary: computation cache, everything always-fresh (fraud_cache=None)
    G — ablation: computation cache, fraud sub-fetch TTL-cached (fraud_cache=TTLCache(...))

Both F and G reuse `WorkflowComputationCache` completely unchanged —
Level 4's branching + graduated freshness live entirely inside the
`return_decision` task spec's `fetch` closure
(`aicache/computation/graph_tasks.py`), not in a new pipeline. Exact
cache (B) is omitted here — Level 3's own results already establish its
structural inadequacy for parameterized traffic (`docs/level3.md`
EXP-009: 3.6% hit rate); nothing about Level 4's branching changes that
finding, so re-running it would be a checkbox, not new evidence.
"""

from __future__ import annotations

import time

from aicache.cache.semantic import SemanticCache
from aicache.cache.ttl_cache import TTLCache
from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.computation.graph_tasks import GRAPH_TASK_REGISTRY, build_graph_registry
from aicache.enterprise.db import EnterpriseDB
from aicache.llm.base import LLMClient
from aicache.llm.graph_rendering import render_graph_answer
from aicache.llm.pricing import estimate_cost_usd
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.graph_correctness import is_graph_correct
from aicache.metrics.types import Outcome, RequestResult

from workloads.level4.types import Level4WorkloadItem


def _fetch_for_item(item: Level4WorkloadItem, db: EnterpriseDB) -> tuple[dict | None, float, str | None]:
    spec = GRAPH_TASK_REGISTRY[item.task_type]
    outcome = spec.fetch(item.entities, item.numbers, db)
    return outcome.values, outcome.latency_ms, outcome.error


def _task_hint(item: Level4WorkloadItem, fetched: dict) -> dict:
    return {
        "operation": item.task_type,
        "kind": "graph",
        "params": {**item.entities, **item.numbers, **fetched},
    }


def _ask_llm(item: Level4WorkloadItem, llm: LLMClient, db: EnterpriseDB) -> tuple[str, float, int, int, float, float]:
    fetched, fetch_ms, error = _fetch_for_item(item, db)
    if fetched is None:
        return f"[fetch failed: {error}]", fetch_ms, 0, 0, 0.0, 0.0
    resp = llm.complete(item.text, task_hint=_task_hint(item, fetched))
    cost = estimate_cost_usd(resp.model_name, resp.input_tokens, resp.output_tokens)
    return resp.text, fetch_ms, resp.input_tokens, resp.output_tokens, resp.latency_ms, cost


def run_no_cache(workload: list[Level4WorkloadItem], llm: LLMClient, db: EnterpriseDB) -> list[RequestResult]:
    results = []
    for item in workload:
        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_graph_correct(item.task_type, text, item.expected_result)
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
                extra={"fetch_ms": fetch_ms, "llm_ms": llm_ms, "branch": item.branch},
            )
        )
    return results


def run_semantic_cache(
    workload: list[Level4WorkloadItem],
    llm: LLMClient,
    db: EnterpriseDB,
    threshold: float = 0.65,
    embedder: Embedder | None = None,
) -> list[RequestResult]:
    cache = SemanticCache(embedder=embedder or Embedder(), threshold=threshold)
    results: list[RequestResult] = []
    for item in workload:
        t0 = time.perf_counter()
        lookup = cache.lookup(item.text)
        lookup_ms = (time.perf_counter() - t0) * 1000

        if lookup.hit:
            correct = is_graph_correct(item.task_type, lookup.answer or "", item.expected_result)
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
                    extra={"matched_request": lookup.matched_request, "branch": item.branch},
                )
            )
            continue

        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_graph_correct(item.task_type, text, item.expected_result)
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
                extra={"fetch_ms": fetch_ms, "llm_ms": llm_ms, "branch": item.branch},
            )
        )
        cache.store(item.text, text)
    return results


def run_graph_computation_cache(
    workload: list[Level4WorkloadItem],
    llm: LLMClient,
    db: EnterpriseDB,
    embedder: Embedder,
    threshold: float,
    fraud_cache: TTLCache | None,
) -> list[RequestResult]:
    """F (`fraud_cache=None`) or G (`fraud_cache=<a TTLCache>`) — caller
    relabels `r.baseline` afterward, same pattern as Level 3's D/E and
    the Level 1/2 threshold sweep before it. Builds its own
    `TaskIdentifier` bound to a registry closed over `fraud_cache` — F
    and G need genuinely different `fetch` closures under the same
    task_type key (see `graph_tasks.build_graph_registry`), so they
    can't share one identifier instance the way D/E could share one
    `WorkflowComputationCache` at Level 3."""
    registry = build_graph_registry(fraud_cache)
    identifier = TaskIdentifier(embedder=embedder, threshold=threshold, registry=registry)
    cache = WorkflowComputationCache(db, identifier, use_stale_cache=False, render_fn=render_graph_answer)

    results: list[RequestResult] = []
    for item in workload:
        lookup = cache.lookup(item.text)
        total_ms = lookup.latency.total_ms

        if lookup.hit:
            correct = is_graph_correct(item.task_type, lookup.served_text or "", item.expected_result)
            results.append(
                RequestResult(
                    request_id=item.request_id,
                    baseline="graph_computation_cache",
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
                        "identify_ms": lookup.latency.identify_ms,
                        "extract_ms": lookup.latency.extract_ms,
                        "validate_ms": lookup.latency.validate_ms,
                        "fetch_ms": lookup.latency.fetch_ms,
                        "execute_ms": lookup.latency.execute_ms,
                        "branch": item.branch,
                        "predicted_branch": (lookup.result or {}).get("branch"),
                    },
                )
            )
            continue

        text, fetch_ms, in_tok, out_tok, llm_ms, cost = _ask_llm(item, llm, db)
        correct = is_graph_correct(item.task_type, text, item.expected_result)
        results.append(
            RequestResult(
                request_id=item.request_id,
                baseline="graph_computation_cache",
                outcome=Outcome.FALLBACK,
                correct=correct,
                latency_ms=total_ms + fetch_ms + llm_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                llm_called=True,
                task_type=lookup.task_type,
                similarity=lookup.similarity,
                extra={"reason": lookup.reason, "fetch_ms": fetch_ms, "llm_ms": llm_ms, "branch": item.branch},
            )
        )
    return results
