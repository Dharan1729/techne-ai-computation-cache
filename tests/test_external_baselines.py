"""Tests for the optional GPTCache/RedisVL comparison baselines.

Skipped automatically (not failed) when the required venv or a reachable
Redis instance isn't present — these are real external systems this repo
doesn't own or vendor, unlike everything else in aicache/. See
benchmarks/external/*_adapter.py for the availability checks these tests
reuse."""

from __future__ import annotations

import pytest

from aicache.llm.simulated import SimulatedLLM
from aicache.metrics.correctness import is_correct
from benchmarks.external import gptcache_adapter, redisvl_adapter
from workloads.generators import generate_workload

_gptcache_ok, _gptcache_reason = gptcache_adapter.is_available()
_redisvl_ok, _redisvl_reason = redisvl_adapter.is_available()


@pytest.mark.skipif(not _gptcache_ok, reason=_gptcache_reason)
def test_gptcache_adapter_runs_and_reports_correctness():
    workload = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=3)
    results = gptcache_adapter.run_gptcache(workload, SimulatedLLM(seed=0), threshold=0.75)
    assert len(results) == len(workload)
    assert all(r.baseline == "gptcache" for r in results)
    # A miss should always be correct — it's served by the precomputed
    # ground-truth-correct simulated LLM answer, not by the cache.
    misses = [r for r in results if r.outcome.value == "miss"]
    assert all(r.correct for r in misses)


@pytest.mark.skipif(not _redisvl_ok, reason=_redisvl_reason)
def test_redisvl_adapter_runs_and_reports_correctness():
    workload = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=3)
    results = redisvl_adapter.run_redisvl(workload, SimulatedLLM(seed=0), similarity_threshold=0.75)
    assert len(results) == len(workload)
    assert all(r.baseline == "redisvl" for r in results)
    misses = [r for r in results if r.outcome.value == "miss"]
    assert all(r.correct for r in misses)


@pytest.mark.skipif(
    not (_gptcache_ok and _redisvl_ok), reason="both gptcache and redisvl must be available"
)
def test_gptcache_and_redisvl_use_different_embedding_models_and_diverge():
    """The point of running both: they use different embedding models
    by default, and that shows up as a measurable difference in false-hit
    behavior at the same nominal threshold — not just a hypothesis, a
    property this test checks directly on every run."""
    workload = generate_workload(param_sets_per_task=4, paraphrases_per_param_set=3, seed=7)
    llm = SimulatedLLM(seed=0)
    gc = gptcache_adapter.run_gptcache(workload, llm, threshold=0.75)
    rv = redisvl_adapter.run_redisvl(workload, llm, similarity_threshold=0.75)

    gc_invalid_rate = sum(1 for r in gc if r.outcome.value == "hit_invalid") / len(gc)
    rv_invalid_rate = sum(1 for r in rv if r.outcome.value == "hit_invalid") / len(rv)

    # GPTCache's generic-SBERT default is expected to show a materially
    # higher false-hit rate than RedisVL's purpose-trained default at
    # the same nominal threshold (see docs/experiment-log.md for the
    # measured magnitude on the full workload).
    assert gc_invalid_rate > rv_invalid_rate
