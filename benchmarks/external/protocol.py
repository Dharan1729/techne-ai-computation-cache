"""Shared JSON protocol between this repo's main process and the
external-system runner scripts (runner_gptcache.py, runner_redisvl.py),
which execute inside GPTCache's and RedisVL's own dedicated virtualenvs
as subprocesses — kept out-of-process specifically to avoid pulling
GPTCache's dependency tree (torch, onnxruntime, faiss-cpu) or RedisVL's
into this project's own environment, where a version conflict with
sentence-transformers/numpy could break the rest of the benchmark.

Protocol: the main process precomputes, for every workload item, what
the (simulated) LLM would answer — using the exact same SimulatedLLM
every other baseline in this repo uses — and hands that to the runner
alongside the request text. This mirrors exactly what
`benchmarks/baselines.py::run_semantic_cache` does in-process: on a
cache miss, call the LLM once, store its answer, move on. The runner
never calls an LLM itself; it only decides hit/miss and, on a hit,
returns whatever text the external cache actually served — which is
what makes an "invalid hit" (GPTCache/RedisVL serving stale text for a
different parameter draw) something this harness can detect exactly the
way it detects it for this repo's own semantic-cache baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from aicache.llm.base import LLMClient
from aicache.llm.pricing import estimate_cost_usd
from aicache.llm.simulated import SimulatedLLM
from aicache.metrics.correctness import is_correct
from aicache.metrics.types import Outcome, RequestResult

from workloads.types import WorkloadItem


def build_export(workload: list[WorkloadItem], llm: LLMClient | None = None) -> list[dict]:
    """Precompute the "what would the LLM have said" text/cost/latency
    for every item, once, in-process — so every external adapter (and a
    future one) works off the identical precomputed answers rather than
    each re-deriving them independently."""
    llm = llm or SimulatedLLM(seed=0)
    export = []
    for item in workload:
        resp = llm.complete(
            item.text, task_hint={"operation": item.task_type, "params": item.bound_params}
        )
        export.append(
            {
                "request_id": item.request_id,
                "text": item.text,
                "task_type": item.task_type,
                "expected_answer": item.expected_answer,
                "llm_answer_text": resp.text,
                "llm_input_tokens": resp.input_tokens,
                "llm_output_tokens": resp.output_tokens,
                "llm_model_name": resp.model_name,
                "llm_latency_ms": resp.latency_ms,
            }
        )
    return export


def results_from_runner_output(
    runner_results: list[dict], baseline_name: str
) -> list[RequestResult]:
    """Convert a runner script's JSON output (one dict per request, each
    carrying the precomputed LLM fields back through unchanged for
    convenience) into this repo's own RequestResult type, exactly the
    same way `benchmarks/baselines.py` does for the in-process
    semantic-cache baseline."""
    out: list[RequestResult] = []
    for r in runner_results:
        if r["hit"]:
            served = r.get("served_text") or ""
            correct = is_correct(served, r["expected_answer"])
            out.append(
                RequestResult(
                    request_id=r["request_id"],
                    baseline=baseline_name,
                    outcome=Outcome.HIT_VALID if correct else Outcome.HIT_INVALID,
                    correct=correct,
                    latency_ms=r["lookup_latency_ms"],
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    llm_called=False,
                    task_type=r.get("task_type"),
                    similarity=r.get("similarity_or_distance"),
                )
            )
        else:
            cost = estimate_cost_usd(
                r["llm_model_name"], r["llm_input_tokens"], r["llm_output_tokens"]
            )
            correct = is_correct(r["llm_answer_text"], r["expected_answer"])
            out.append(
                RequestResult(
                    request_id=r["request_id"],
                    baseline=baseline_name,
                    outcome=Outcome.MISS,
                    correct=correct,
                    latency_ms=r["llm_latency_ms"] + r["lookup_latency_ms"],
                    input_tokens=r["llm_input_tokens"],
                    output_tokens=r["llm_output_tokens"],
                    cost_usd=cost,
                    llm_called=True,
                    task_type=r.get("task_type"),
                    similarity=r.get("similarity_or_distance"),
                )
            )
    return out
