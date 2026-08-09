"""Integration tests proving the four baselines behave as the research
questions predict, end to end (workload -> baseline -> RequestResult),
using the deterministic SimulatedLLM backend so results are reproducible
without network access."""

from __future__ import annotations

from aicache.llm.simulated import SimulatedLLM
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.aggregate import summarize
from aicache.metrics.types import Outcome

from benchmarks.baselines import (
    run_computation_cache,
    run_exact_cache,
    run_no_cache,
    run_semantic_cache,
)
from workloads.generators import generate_workload


def _small_workload():
    return generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=11)


def test_no_cache_always_calls_llm_and_is_correct():
    workload = _small_workload()
    results = run_no_cache(workload, SimulatedLLM(seed=0))
    assert all(r.outcome == Outcome.NO_CACHE for r in results)
    assert all(r.llm_called for r in results)
    assert all(r.correct for r in results)  # ground-truth executor == "model"


def test_exact_cache_never_hits_on_paraphrased_workload():
    """RQ1: exact caching should do essentially nothing when every
    request is a paraphrase with different parameters."""
    workload = _small_workload()
    results = run_exact_cache(workload, SimulatedLLM(seed=0))
    hits = [r for r in results if r.outcome in (Outcome.HIT_VALID, Outcome.HIT_INVALID)]
    assert len(hits) == 0


def test_semantic_cache_hits_are_not_all_correct(embedder):
    """RQ2/RQ3, the central empirical finding: at a threshold loose
    enough to produce real reuse, some of that reuse serves a stale,
    wrong answer. This test only requires that SOME invalid hit occurs
    somewhere in a reasonably large, repeated-structure workload — not a
    specific rate, since that's threshold- and workload-dependent (see
    docs/experiment-log.md for the measured rate at threshold 0.75)."""
    workload = generate_workload(param_sets_per_task=4, paraphrases_per_param_set=3, seed=11)
    results = run_semantic_cache(workload, SimulatedLLM(seed=0), threshold=0.75, embedder=embedder)
    invalid = [r for r in results if r.outcome == Outcome.HIT_INVALID]
    valid = [r for r in results if r.outcome == Outcome.HIT_VALID]
    assert len(valid) + len(invalid) > 0, "expected at least some semantic hits on this workload"
    assert len(invalid) > 0, (
        "expected at least one false hit under parameter variation — if this "
        "starts failing, either the workload got easier or the caching bug "
        "this baseline is supposed to demonstrate got fixed (which would be "
        "a different kind of interesting)"
    )


def test_computation_cache_hits_are_always_correct(embedder):
    """H3/H4: unlike the semantic cache, every computation-cache hit
    freshly executes the identified operation on freshly-bound
    parameters, so (identification and binding both succeeding) implies
    correctness, by construction."""
    workload = _small_workload()
    identifier = TaskIdentifier(embedder=embedder, threshold=0.6)
    results = run_computation_cache(workload, SimulatedLLM(seed=0), identifier=identifier)
    hits = [r for r in results if r.outcome == Outcome.HIT_VALID]
    invalid_hits = [r for r in results if r.outcome == Outcome.HIT_INVALID]
    assert len(hits) > 0, "expected at least one computation-cache hit"
    assert len(invalid_hits) == 0, (
        "a computation-cache hit went wrong — see docs/experiment-log.md "
        "for the known failure mode (misidentified task) this would indicate"
    )


def test_computation_cache_never_calls_llm_on_a_hit(embedder):
    workload = _small_workload()
    identifier = TaskIdentifier(embedder=embedder, threshold=0.6)
    results = run_computation_cache(workload, SimulatedLLM(seed=0), identifier=identifier)
    for r in results:
        if r.outcome == Outcome.HIT_VALID:
            assert not r.llm_called
            assert r.input_tokens == 0 and r.output_tokens == 0
            assert r.cost_usd == 0.0


def test_summarize_outcome_counts_sum_to_request_count():
    workload = _small_workload()
    results = run_no_cache(workload, SimulatedLLM(seed=0))
    summary = summarize(results)
    assert sum(summary.outcome_counts.values()) == summary.request_count == len(workload)
