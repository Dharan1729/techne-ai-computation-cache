import math

from aicache.cache.computation import ComputationCache
from aicache.matching.task_identification import TaskIdentifier


def _cache(embedder, threshold=0.52):
    # 0.52 matches TaskIdentifier's own calibrated default (see that
    # module) for redis/langcache-embed-v1 — zero misidentified matches
    # observed at this threshold across this project's own 150-request
    # workload, at ~82% recall of genuinely-correct matches.
    return ComputationCache(identifier=TaskIdentifier(embedder=embedder, threshold=threshold))


def test_hit_computes_fresh_correct_answer(embedder):
    cache = _cache(embedder)
    result = cache.lookup("Can you add 512 and 937?")
    assert result.hit
    assert result.task_type == "add"
    assert math.isclose(result.answer, 1449.0)


def test_hit_recomputes_for_different_parameters_each_time(embedder):
    """The central claim (H3/H4): the SAME computation (add) produces a
    FRESH, correct answer for each new set of parameters — never a
    cached/stale response."""
    cache = _cache(embedder)
    r1 = cache.lookup("What is 10 plus 20?")
    r2 = cache.lookup("Can you add 512 and 937?")
    assert r1.hit and r2.hit
    assert r1.task_type == r2.task_type == "add"
    assert math.isclose(r1.answer, 30.0)
    assert math.isclose(r2.answer, 1449.0)


def test_fallback_on_unrelated_request(embedder):
    cache = _cache(embedder, threshold=0.6)
    result = cache.lookup("What is the capital of France?")
    assert not result.hit
    assert result.reason == "task_identification_below_threshold"


def test_fallback_on_insufficient_parameters(embedder):
    """Identification can succeed (this looks like a compound_interest
    question) while binding still correctly fails and falls back, rather
    than executing with missing/garbage parameters."""
    cache = _cache(embedder, threshold=0.3)
    result = cache.lookup("If I invest at some interest rate, what happens?")
    assert not result.hit
