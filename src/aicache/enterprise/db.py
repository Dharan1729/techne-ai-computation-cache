"""A small, in-memory, mutable "enterprise" database — customers,
transactions, tier policies. Deliberately not SQLite/Postgres: this repo
has stayed dependency-light throughout, and nothing about the Level 3
research question (does computation reuse survive contact with live,
changing state?) needs a real database engine, only something that can
be read, fetched by ID, and mutated between requests.

The one property that matters more than any schema detail: **mutable**.
A database generated once and never touched again cannot test
freshness — see docs/level3.md for why this was the central design
correction from the Level 1/2-style static workload.

Fetch operations carry a small simulated latency (``_SIMULATED_FETCH_MS``)
so "the fetch still happens on every hit" is visible in the recorded
timing, not silently zero — a real database round-trip is not free, and
pretending it is would overstate computation caching's speedup (see
docs/level3.md's latency-component breakdown).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

_SIMULATED_FETCH_MS = 8.0  # per fetch call; a modest, not-hand-wavy LAN DB round-trip
# Fraud scores model a slower, ML-model-backed tool call (Level 4,
# docs/level4.md) — deliberately costlier than a plain row read, which is
# exactly what makes TTL-caching this specific field operationally
# tempting in a real system (see LangGraph's real CachePolicy(ttl=...),
# docs/prior-art.md) rather than an obviously bad idea the way caching a
# transaction's live status would be.
_SIMULATED_FRAUD_FETCH_MS = 50.0

TIERS = ["bronze", "silver", "gold"]

# Tier -> policy fields. Mutable at runtime via mutate_policy().
_DEFAULT_POLICIES: dict[str, dict] = {
    "bronze": {
        "tier": "bronze",
        "refund_window_days": 14,
        "discount_percent": 0.0,
        "free_shipping_threshold": None,  # never free
        "loyalty_threshold": 500.0,
    },
    "silver": {
        "tier": "silver",
        "refund_window_days": 30,
        "discount_percent": 5.0,
        "free_shipping_threshold": 100.0,
        "loyalty_threshold": 1000.0,
    },
    "gold": {
        "tier": "gold",
        "refund_window_days": 60,
        "discount_percent": 10.0,
        "free_shipping_threshold": 0.0,  # always free
        "loyalty_threshold": 2000.0,
    },
}

_SHIPPING_ZONE_RATES = {"zone-1": 5.0, "zone-2": 9.0, "zone-3": 14.0}

TRANSACTION_STATUSES = ["completed", "pending", "refunded", "disputed"]


@dataclass
class FetchResult:
    value: dict | list | None
    latency_ms: float
    found: bool


@dataclass
class EnterpriseDB:
    customers: dict[str, dict] = field(default_factory=dict)
    transactions: dict[str, dict] = field(default_factory=dict)
    policies: dict[str, dict] = field(default_factory=dict)
    fraud_scores: dict[str, dict] = field(default_factory=dict)
    _mutation_log: list[str] = field(default_factory=list)

    # ---------------------------------------------------------- fetches --

    def fetch_customer(self, customer_id: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FETCH_MS / 1000)
        row = self.customers.get(customer_id)
        # A real query returns a snapshot as of query time, not a live
        # reference into the table — copy the row so a later mutation to
        # the underlying store can never retroactively change data
        # already returned from an earlier fetch. Without this, the
        # freshness ablation (WorkflowComputationCache's memoized-fetch
        # mode) silently isn't stale at all: its memoized dict would be
        # the *same object* the table mutates in place, so "reusing old
        # data" and "reading current data" would be indistinguishable —
        # exactly the bug this comment is here because of; caught before
        # any experiment ran on it (see docs/level3.md).
        return FetchResult(dict(row) if row else None, (time.perf_counter() - t0) * 1000, row is not None)

    def fetch_transaction(self, transaction_id: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FETCH_MS / 1000)
        row = self.transactions.get(transaction_id)
        return FetchResult(dict(row) if row else None, (time.perf_counter() - t0) * 1000, row is not None)

    def fetch_policy(self, tier: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FETCH_MS / 1000)
        row = self.policies.get(tier)
        return FetchResult(dict(row) if row else None, (time.perf_counter() - t0) * 1000, row is not None)

    def fetch_transactions_for_customer(self, customer_id: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FETCH_MS / 1000)
        rows = [dict(t) for t in self.transactions.values() if t["customer_id"] == customer_id]
        return FetchResult(rows, (time.perf_counter() - t0) * 1000, True)

    def fetch_shipping_rate(self, zone: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FETCH_MS / 1000)
        rate = _SHIPPING_ZONE_RATES.get(zone)
        return FetchResult(rate, (time.perf_counter() - t0) * 1000, rate is not None)

    def fetch_fraud_score(self, customer_id: str) -> FetchResult:
        t0 = time.perf_counter()
        time.sleep(_SIMULATED_FRAUD_FETCH_MS / 1000)
        row = self.fraud_scores.get(customer_id)
        return FetchResult(dict(row) if row else None, (time.perf_counter() - t0) * 1000, row is not None)

    # ---------------------------------------------------------- mutations --
    # Every mutation is logged (mutation description) so an experiment
    # replaying a fixed script can report exactly what changed and when —
    # see docs/level3.md's state-mutation experiment.

    def mutate_customer_tier(self, customer_id: str, new_tier: str) -> None:
        old = self.customers[customer_id]["tier"]
        self.customers[customer_id]["tier"] = new_tier
        self._mutation_log.append(f"customer {customer_id}: tier {old} -> {new_tier}")

    def mutate_transaction_status(self, transaction_id: str, new_status: str) -> None:
        old = self.transactions[transaction_id]["status"]
        self.transactions[transaction_id]["status"] = new_status
        self._mutation_log.append(
            f"transaction {transaction_id}: status {old} -> {new_status}"
        )

    def mutate_transaction_amount(self, transaction_id: str, new_amount: float) -> None:
        old = self.transactions[transaction_id]["amount"]
        self.transactions[transaction_id]["amount"] = new_amount
        self._mutation_log.append(
            f"transaction {transaction_id}: amount {old} -> {new_amount}"
        )

    def mutate_policy(self, tier: str, **changes) -> None:
        for k, v in changes.items():
            old = self.policies[tier].get(k)
            self.policies[tier][k] = v
            self._mutation_log.append(f"policy {tier}.{k}: {old} -> {v}")

    def mutate_fraud_score(self, customer_id: str, new_score: float) -> None:
        old = self.fraud_scores[customer_id]["score"]
        self.fraud_scores[customer_id]["score"] = new_score
        self._mutation_log.append(f"fraud_score {customer_id}: {old} -> {new_score}")

    # ---------------------------------------------------------- lifecycle --

    def clone(self) -> "EnterpriseDB":
        """A fresh, independent copy — used so each baseline in an
        experiment replays the same script against its own DB starting
        from the same initial state, rather than fighting over one
        shared mutable object (see benchmarks/level3/harness.py)."""
        return EnterpriseDB(
            customers=copy.deepcopy(self.customers),
            transactions=copy.deepcopy(self.transactions),
            policies=copy.deepcopy(self.policies),
            fraud_scores=copy.deepcopy(self.fraud_scores),
            _mutation_log=[],
        )

    @property
    def mutation_log(self) -> list[str]:
        return list(self._mutation_log)


def generate_enterprise_db(
    seed: int = 42, n_customers: int = 80, n_transactions: int = 200
) -> EnterpriseDB:
    rng = random.Random(seed)
    db = EnterpriseDB(policies=copy.deepcopy(_DEFAULT_POLICIES))

    tier_weights = [0.5, 0.35, 0.15]  # bronze-heavy, gold-rare — realistic-ish skew
    for i in range(n_customers):
        customer_id = f"CUST-{i:04d}"
        db.customers[customer_id] = {
            "customer_id": customer_id,
            "tier": rng.choices(TIERS, weights=tier_weights)[0],
            "region": rng.choice(["us-east", "us-west", "eu", "apac"]),
            "account_status": "active" if rng.random() > 0.05 else "suspended",
            "signup_day": rng.randint(1, 1000),
        }
        # A two-population mixture, not a single tail-heavy distribution:
        # a beta(2,6) tail (mean ~0.25) alone puts under 1% of scores
        # above the 0.7 high-risk cutoff (aicache/computation/graph_tasks.py)
        # — checked numerically before trusting it, the same discipline
        # that caught Level 3's uniform-day/refund-window mismatch
        # (docs/level3.md) — which would make the fraud-check branch a
        # rare accident rather than a reliably exercised code path.
        # Explicit ~12% high-risk pool guarantees branch coverage instead.
        if rng.random() < 0.12:
            score = round(rng.uniform(0.65, 0.95), 4)
        else:
            score = round(rng.betavariate(2, 6), 4)
        db.fraud_scores[customer_id] = {
            "customer_id": customer_id,
            "score": score,
            "computed_day": max(1, 1000 - rng.randint(0, 60)),
        }

    customer_ids = list(db.customers.keys())
    status_weights = [0.7, 0.15, 0.10, 0.05]  # completed-heavy
    for i in range(n_transactions):
        transaction_id = f"TXN-{i:06d}"
        db.transactions[transaction_id] = {
            "transaction_id": transaction_id,
            "customer_id": rng.choice(customer_ids),
            "amount": round(rng.uniform(10, 2000), 2),
            # Recency-skewed, not uniform: age (days since "now"=1000)
            # is exponential with a mean of 150 days, not a uniform draw
            # over the whole 1000-day history — realistic for a live
            # transaction log (recent activity dominates) and load-bearing
            # for this workload specifically. A uniform 1-1000 draw
            # against refund windows of 14-60 days put only ~2.5-5% of
            # the *entire* database within any refund window — not a
            # sampling-luck artifact, a structural skew caught by
            # checking the full-database distribution, not just one
            # workload sample, before trusting the generator (see
            # docs/level3.md).
            "day": max(1, 1000 - round(rng.expovariate(1 / 150))),
            "status": rng.choices(TRANSACTION_STATUSES, weights=status_weights)[0],
            "category": rng.choice(["electronics", "apparel", "home", "grocery", "other"]),
        }

    return db
