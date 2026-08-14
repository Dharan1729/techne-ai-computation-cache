from aicache.cache.ttl_cache import TTLCache
from aicache.llm.simulated import SimulatedLLM
from benchmarks.level4.baselines import run_graph_computation_cache
from benchmarks.level4.run_experiment import LEVEL4_IDENTIFICATION_THRESHOLD
from workloads.level4.generator import generate_level4_workload

THRESHOLD = LEVEL4_IDENTIFICATION_THRESHOLD


def test_fresh_baseline_is_100pct_correct_on_a_small_workload(embedder):
    db, items = generate_level4_workload(combinations_per_branch=2, paraphrases_per_combination=2, seed=5, db_seed=5)
    llm = SimulatedLLM(seed=0)
    results = run_graph_computation_cache(items, llm, db.clone(), embedder, THRESHOLD, fraud_cache=None)
    assert all(r.correct for r in results)
    assert all(r.outcome.value != "hit_invalid" for r in results)


def test_ttl_ablation_matches_fresh_baseline_when_db_never_mutates(embedder):
    """Mirrors Level 3's D/E-identical-when-static finding: absent any
    mutation, memoizing the fraud sub-fetch produces the same answers as
    always re-fetching it."""
    db, items = generate_level4_workload(combinations_per_branch=2, paraphrases_per_combination=2, seed=5, db_seed=5)
    llm = SimulatedLLM(seed=0)
    fresh = run_graph_computation_cache(items, llm, db.clone(), embedder, THRESHOLD, fraud_cache=None)
    ttl_ablation = run_graph_computation_cache(
        items, llm, db.clone(), embedder, THRESHOLD, fraud_cache=TTLCache(ttl_seconds=None)
    )
    assert [r.correct for r in fresh] == [r.correct for r in ttl_ablation]
