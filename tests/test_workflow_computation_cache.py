from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import generate_enterprise_db
from aicache.matching.task_identification import TaskIdentifier

# Matches LEVEL3_IDENTIFICATION_THRESHOLD in benchmarks/level3/run_experiment.py.
THRESHOLD = 0.45


def _identifier(embedder):
    return TaskIdentifier(embedder=embedder, threshold=THRESHOLD, registry=WORKFLOW_TASK_REGISTRY)


def _eligible_refund_db():
    db = generate_enterprise_db(seed=1, n_customers=10, n_transactions=20)
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id].update(status="completed", day=990, amount=500.0)
    db.customers[cust_id]["tier"] = "gold"
    return db, cust_id, txn_id


def test_hit_serves_a_correct_answer(embedder):
    db, cust_id, txn_id = _eligible_refund_db()
    cache = WorkflowComputationCache(db, _identifier(embedder))
    result = cache.lookup(f"Is customer {cust_id} eligible for a refund on transaction {txn_id}?")
    assert result.hit
    assert result.task_type == "refund_eligibility"
    assert result.result["eligible"] is True


def test_fallback_on_unrelated_request(embedder):
    db, _, _ = _eligible_refund_db()
    cache = WorkflowComputationCache(db, _identifier(embedder))
    result = cache.lookup("What is the capital of France?")
    assert not result.hit


def test_fresh_cache_reflects_mutation_but_stale_ablation_does_not(embedder):
    """The core Level 3 claim: D (use_stale_cache=False) always re-fetches
    and correctly reflects a database mutation; E (use_stale_cache=True)
    is deliberately stale and keeps serving the pre-mutation answer."""
    db, cust_id, txn_id = _eligible_refund_db()
    text = f"Is customer {cust_id} eligible for a refund on transaction {txn_id}?"
    identifier = _identifier(embedder)
    cache_d = WorkflowComputationCache(db, identifier, use_stale_cache=False)
    cache_e = WorkflowComputationCache(db, identifier, use_stale_cache=True)

    before_d = cache_d.lookup(text)
    before_e = cache_e.lookup(text)
    assert before_d.result["eligible"] is True
    assert before_e.result["eligible"] is True

    db.mutate_transaction_status(txn_id, "refunded")

    after_d = cache_d.lookup(text)
    after_e = cache_e.lookup(text)
    assert after_d.result["eligible"] is False  # fresh fetch sees the mutation
    assert after_e.result["eligible"] is True  # stale ablation still serves the old fetch
    assert after_e.used_stale_fetch is True
    assert after_e.latency.fetch_ms == 0.0


def test_latency_breakdown_has_all_five_components(embedder):
    db, cust_id, txn_id = _eligible_refund_db()
    cache = WorkflowComputationCache(db, _identifier(embedder))
    result = cache.lookup(f"Is customer {cust_id} eligible for a refund on transaction {txn_id}?")
    assert result.latency.identify_ms >= 0
    assert result.latency.extract_ms >= 0
    assert result.latency.validate_ms >= 0
    assert result.latency.fetch_ms > 0  # a real simulated DB round-trip happened
    assert result.latency.execute_ms >= 0
    assert result.latency.total_ms == sum(
        [
            result.latency.identify_ms,
            result.latency.extract_ms,
            result.latency.validate_ms,
            result.latency.fetch_ms,
            result.latency.execute_ms,
        ]
    )
