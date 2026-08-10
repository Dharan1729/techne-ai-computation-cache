from aicache.enterprise.db import generate_enterprise_db


def test_generate_produces_requested_counts():
    db = generate_enterprise_db(seed=1, n_customers=10, n_transactions=30)
    assert len(db.customers) == 10
    assert len(db.transactions) == 30


def test_fetch_returns_a_copy_not_a_live_reference():
    """The bug this guards against: memoized fetches (the freshness
    ablation) must not be silently mutated by a later mutate_* call just
    because they share the same dict object as the table row."""
    db = generate_enterprise_db(seed=1, n_customers=5, n_transactions=5)
    txn_id = next(iter(db.transactions))
    fetched = db.fetch_transaction(txn_id).value
    db.mutate_transaction_status(txn_id, "refunded")
    assert fetched["status"] != "refunded"


def test_mutations_are_logged():
    db = generate_enterprise_db(seed=1, n_customers=5, n_transactions=5)
    cust_id = next(iter(db.customers))
    db.mutate_customer_tier(cust_id, "gold")
    assert any("tier" in entry for entry in db.mutation_log)


def test_clone_is_independent():
    db = generate_enterprise_db(seed=1, n_customers=5, n_transactions=5)
    clone = db.clone()
    cust_id = next(iter(db.customers))
    db.mutate_customer_tier(cust_id, "gold")
    assert clone.customers[cust_id]["tier"] != "gold"


def test_fetch_customer_not_found():
    db = generate_enterprise_db(seed=1, n_customers=5, n_transactions=5)
    result = db.fetch_customer("CUST-9999")
    assert not result.found
    assert result.value is None


def test_transaction_day_distribution_puts_a_meaningful_share_in_refund_windows():
    """Regression check for the uniform-vs-exponential fix: with the
    recency-skewed distribution, a real share of the database (not 2-5%)
    should land within a typical 14-60 day refund window."""
    db = generate_enterprise_db(seed=42, n_customers=80, n_transactions=200)
    now = 1000
    within_60_days = sum(1 for t in db.transactions.values() if now - t["day"] <= 60)
    assert within_60_days / len(db.transactions) >= 0.08
