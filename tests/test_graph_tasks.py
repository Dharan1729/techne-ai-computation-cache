from aicache.cache.ttl_cache import TTLCache
from aicache.computation.graph_tasks import (
    FRAUD_CHECK_AMOUNT_THRESHOLD,
    FRAUD_SCORE_HIGH_RISK,
    GRAPH_TASK_REGISTRY,
    build_graph_registry,
)
from aicache.enterprise.db import generate_enterprise_db


def _db():
    return generate_enterprise_db(seed=1, n_customers=10, n_transactions=10)


def _setup(db, cust_id, txn_id, *, tier="gold", status="completed", day=990, amount=900.0, score=0.1):
    db.customers[cust_id]["tier"] = tier
    db.fraud_scores[cust_id]["score"] = score
    db.transactions[txn_id] = {
        "transaction_id": txn_id, "customer_id": cust_id, "amount": amount,
        "day": day, "status": status, "category": "other",
    }


def _run(spec, cust_id, txn_id, db):
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    assert outcome.values is not None, outcome.error
    return spec.executor(**entities, **outcome.values)


def test_not_eligible_branch():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000000"
    _setup(db, cust_id, txn_id, status="pending")
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    result = _run(spec, cust_id, txn_id, db)
    assert result == {"decision": "DENY", "branch": "not_eligible", "refund_amount": 0.0, "fraud_score": None}


def test_auto_approve_low_value_branch_skips_fraud_check():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000001"
    _setup(db, cust_id, txn_id, amount=FRAUD_CHECK_AMOUNT_THRESHOLD - 1)
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    result = _run(spec, cust_id, txn_id, db)
    assert result["decision"] == "APPROVE"
    assert result["branch"] == "auto_approve_low_value"
    assert result["fraud_score"] is None


def test_fraud_cleared_branch():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000002"
    _setup(db, cust_id, txn_id, amount=FRAUD_CHECK_AMOUNT_THRESHOLD + 1, score=FRAUD_SCORE_HIGH_RISK - 0.1)
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    result = _run(spec, cust_id, txn_id, db)
    assert result["decision"] == "APPROVE"
    assert result["branch"] == "fraud_cleared"
    assert result["fraud_score"] is not None


def test_fraud_flagged_branch():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000003"
    _setup(db, cust_id, txn_id, amount=FRAUD_CHECK_AMOUNT_THRESHOLD + 1, score=FRAUD_SCORE_HIGH_RISK + 0.1)
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    result = _run(spec, cust_id, txn_id, db)
    assert result["decision"] == "DENY"
    assert result["branch"] == "fraud_flagged"


def test_ttl_cache_reuses_stale_fraud_score_within_window_but_not_after():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000004"
    _setup(db, cust_id, txn_id, amount=FRAUD_CHECK_AMOUNT_THRESHOLD + 1, score=0.1)

    clock = [0.0]
    ttl_cache = TTLCache(ttl_seconds=5.0, now_fn=lambda: clock[0])
    spec = build_graph_registry(ttl_cache)["return_decision"]

    before = _run(spec, cust_id, txn_id, db)
    assert before["branch"] == "fraud_cleared"

    db.mutate_fraud_score(cust_id, 0.95)
    clock[0] = 2.0  # within the 5s TTL
    still_stale = _run(spec, cust_id, txn_id, db)
    assert still_stale["branch"] == "fraud_cleared"  # stale on purpose

    clock[0] = 6.0  # past the TTL
    fresh_again = _run(spec, cust_id, txn_id, db)
    assert fresh_again["branch"] == "fraud_flagged"


def test_always_fresh_spec_never_reuses_fraud_score():
    db = _db()
    cust_id, txn_id = next(iter(db.customers)), "TXN-000005"
    _setup(db, cust_id, txn_id, amount=FRAUD_CHECK_AMOUNT_THRESHOLD + 1, score=0.1)
    spec = build_graph_registry(fraud_cache=None)["return_decision"]

    before = _run(spec, cust_id, txn_id, db)
    assert before["branch"] == "fraud_cleared"

    db.mutate_fraud_score(cust_id, 0.95)
    after = _run(spec, cust_id, txn_id, db)
    assert after["branch"] == "fraud_flagged"
