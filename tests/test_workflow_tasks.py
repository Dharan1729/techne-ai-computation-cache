from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import generate_enterprise_db


def _db():
    return generate_enterprise_db(seed=1, n_customers=10, n_transactions=20)


def test_every_task_has_example_phrasings():
    for task_type, spec in WORKFLOW_TASK_REGISTRY.items():
        assert spec.example_phrasings, f"{task_type} has no example phrasings"


def test_every_task_is_always_fresh():
    """Load-bearing at Level 3, unlike Level 1/2 where this field was
    documentation-only — every task must actually re-fetch on demand."""
    for task_type, spec in WORKFLOW_TASK_REGISTRY.items():
        assert spec.freshness_requirements == "always_fresh", task_type


def test_refund_eligibility_within_window():
    db = _db()
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id].update(status="completed", day=990)
    db.customers[cust_id]["tier"] = "gold"  # 60-day window

    spec = WORKFLOW_TASK_REGISTRY["refund_eligibility"]
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    assert outcome.values is not None
    result = spec.executor(**entities, **outcome.values)
    assert result["eligible"] is True
    assert result["reason"] == "within_refund_window"


def test_refund_eligibility_outside_window():
    db = _db()
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id].update(status="completed", day=100)
    db.customers[cust_id]["tier"] = "gold"

    spec = WORKFLOW_TASK_REGISTRY["refund_eligibility"]
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    result = spec.executor(**entities, **outcome.values)
    assert result["eligible"] is False
    assert result["reason"] == "outside_refund_window"


def test_refund_eligibility_not_completed():
    db = _db()
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id].update(status="pending", day=990)

    spec = WORKFLOW_TASK_REGISTRY["refund_eligibility"]
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    result = spec.executor(**entities, **outcome.values)
    assert result["eligible"] is False
    assert result["reason"] == "transaction_not_completed"


def test_refund_eligibility_transaction_not_owned_by_customer():
    db = _db()
    txn_id = next(iter(db.transactions))
    other_cust = next(c for c in db.customers if c != db.transactions[txn_id]["customer_id"])

    spec = WORKFLOW_TASK_REGISTRY["refund_eligibility"]
    outcome = spec.fetch({"customer_id": other_cust, "transaction_id": txn_id}, {}, db)
    assert outcome.values is None
    assert outcome.error == "transaction_does_not_belong_to_customer"


def test_loyalty_discount_applies_tier_percent():
    db = _db()
    cust_id = next(iter(db.customers))
    db.customers[cust_id]["tier"] = "gold"  # 10% per _DEFAULT_POLICIES

    spec = WORKFLOW_TASK_REGISTRY["loyalty_discount"]
    entities = {"customer_id": cust_id}
    numbers = {"order_total": 400.0}
    outcome = spec.fetch(entities, numbers, db)
    result = spec.executor(**entities, **numbers, **outcome.values)
    assert result["discount_percent"] == 10.0
    assert result["discount_amount"] == 40.0
    assert result["final_amount"] == 360.0


def test_shipping_cost_free_for_gold():
    db = _db()
    cust_id = next(iter(db.customers))
    db.customers[cust_id]["tier"] = "gold"  # free_shipping_threshold = 0

    spec = WORKFLOW_TASK_REGISTRY["shipping_cost"]
    entities = {"customer_id": cust_id, "zone": "zone-2"}
    numbers = {"weight": 5.0}
    outcome = spec.fetch(entities, numbers, db)
    result = spec.executor(**entities, **numbers, **outcome.values)
    assert result["free_shipping"] is True
    assert result["cost"] == 0.0


def test_shipping_cost_charged_for_bronze():
    db = _db()
    cust_id = next(iter(db.customers))
    db.customers[cust_id]["tier"] = "bronze"  # free_shipping_threshold = None

    spec = WORKFLOW_TASK_REGISTRY["shipping_cost"]
    entities = {"customer_id": cust_id, "zone": "zone-1"}
    numbers = {"weight": 10.0}
    outcome = spec.fetch(entities, numbers, db)
    result = spec.executor(**entities, **numbers, **outcome.values)
    assert result["free_shipping"] is False
    assert result["cost"] > 0


def test_customer_spend_summary_only_counts_completed():
    db = _db()
    cust_id = next(iter(db.customers))
    for t in db.transactions.values():
        t["customer_id"] = cust_id
    ids = list(db.transactions)
    db.transactions[ids[0]].update(status="completed", amount=100.0)
    db.transactions[ids[1]].update(status="completed", amount=50.0)
    db.transactions[ids[2]].update(status="pending", amount=999.0)

    spec = WORKFLOW_TASK_REGISTRY["customer_spend_summary"]
    entities = {"customer_id": cust_id}
    outcome = spec.fetch(entities, {}, db)
    result = spec.executor(**entities, **outcome.values)
    assert result["transaction_count"] >= 2
    assert result["total_spend"] >= 150.0


def test_transaction_risk_disputed_is_high():
    db = _db()
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id]["status"] = "disputed"

    spec = WORKFLOW_TASK_REGISTRY["transaction_risk"]
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    result = spec.executor(**entities, **outcome.values)
    assert result["risk"] == "high"


def test_transaction_risk_large_bronze_amount_is_medium():
    db = _db()
    txn_id = next(iter(db.transactions))
    cust_id = db.transactions[txn_id]["customer_id"]
    db.transactions[txn_id].update(status="completed", amount=5000.0)
    db.customers[cust_id].update(tier="bronze", account_status="active")

    spec = WORKFLOW_TASK_REGISTRY["transaction_risk"]
    entities = {"customer_id": cust_id, "transaction_id": txn_id}
    outcome = spec.fetch(entities, {}, db)
    result = spec.executor(**entities, **outcome.values)
    assert result["risk"] == "medium"
