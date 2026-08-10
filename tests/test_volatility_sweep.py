import random

from benchmarks.level3.volatility_sweep import _build_probe_db, _run_rate_point
from aicache.matching.embedder import Embedder


def test_probe_db_produces_guaranteed_eligible_pairs():
    db, pairs = _build_probe_db(5)
    assert len(pairs) == 5
    for cust_id, txn_id in pairs:
        assert db.customers[cust_id]["tier"] == "gold"
        assert db.transactions[txn_id]["status"] == "completed"
        assert db.transactions[txn_id]["customer_id"] == cust_id


def test_zero_mutation_rate_is_correct_for_every_baseline(embedder: Embedder):
    db, pairs = _build_probe_db(6)
    point = _run_rate_point(0.0, db, pairs, embedder, random.Random(0))
    assert point.correctness("semantic_cache") == 1.0
    assert point.correctness("computation_cache_fresh") == 1.0
    assert point.correctness("computation_cache_stale_ablation") == 1.0


def test_full_mutation_rate_breaks_semantic_and_stale_but_not_fresh(embedder: Embedder):
    db, pairs = _build_probe_db(6)
    point = _run_rate_point(1.0, db, pairs, embedder, random.Random(0))
    assert point.correctness("semantic_cache") == 0.0
    assert point.correctness("computation_cache_fresh") == 1.0
    assert point.correctness("computation_cache_stale_ablation") == 0.0
