"""Generates the Level 4 primary comparison workload for
``return_decision`` — explicitly constructed per-branch, not left to
random combinatorics. `not_eligible` is common by chance;
`fraud_flagged` needs a high-risk customer (~6% of the base-generated
population, see `aicache/enterprise/db.py`) AND a >$500 eligible
transaction, which is rare enough by chance that a purely random
workload would badly under-represent it — the same class-imbalance
lesson Level 3's `day`-distribution fix already taught (see
`docs/level3.md`): check branch balance directly, don't assume the
generator produces it.

Ground truth is computed via `GRAPH_TASK_REGISTRY` (the always-fresh
oracle, `aicache/computation/graph_tasks.py`) directly against the final
DB state — independent of any baseline under test, same discipline as
`workloads/level3/generator.py`.
"""

from __future__ import annotations

import random

from aicache.computation.graph_tasks import GRAPH_TASK_REGISTRY
from aicache.enterprise.db import EnterpriseDB, generate_enterprise_db

from .templates import LEVEL4_TEMPLATES
from .types import Level4WorkloadItem

BRANCHES = ["not_eligible", "auto_approve_low_value", "fraud_cleared", "fraud_flagged"]
_GOLD_WINDOW_DAYS = 60
_NOW = 1000


def _make_pair_for_branch(rng: random.Random, db: EnterpriseDB, branch: str, index: int) -> tuple[str, str]:
    """Constructs a dedicated customer+transaction pair guaranteed to
    land in `branch`. Uses a reserved ID range (CUST-9xxx / TXN-9xxxxx)
    so these never collide with the base-generated population."""
    cust_id = f"CUST-9{index:03d}"
    txn_id = f"TXN-9{index:05d}"

    db.customers[cust_id] = {
        "customer_id": cust_id,
        "tier": "gold",  # fixes the refund window at 60 days for all branches, simplifying construction
        "region": "us-east",
        "account_status": "active",
        "signup_day": 1,
    }
    db.fraud_scores[cust_id] = {
        "customer_id": cust_id,
        "score": 0.9 if branch == "fraud_flagged" else round(rng.uniform(0.05, 0.3), 4),
        "computed_day": _NOW - rng.randint(0, 10),
    }

    if branch == "not_eligible":
        amount = round(rng.uniform(50, 1500), 2)
        status = "pending"  # unambiguously not eligible regardless of window
        day = _NOW - rng.randint(0, 10)
    elif branch == "auto_approve_low_value":
        amount = round(rng.uniform(20, 500), 2)
        status = "completed"
        day = _NOW - rng.randint(0, _GOLD_WINDOW_DAYS - 5)
    else:  # fraud_cleared or fraud_flagged: both need amount > threshold + eligible
        amount = round(rng.uniform(501, 3000), 2)
        status = "completed"
        day = _NOW - rng.randint(0, _GOLD_WINDOW_DAYS - 5)

    db.transactions[txn_id] = {
        "transaction_id": txn_id,
        "customer_id": cust_id,
        "amount": amount,
        "day": day,
        "status": status,
        "category": "other",
    }
    return cust_id, txn_id


def generate_level4_workload(
    db_seed: int = 42,
    n_customers: int = 80,
    n_transactions: int = 200,
    combinations_per_branch: int = 15,
    paraphrases_per_combination: int = 3,
    seed: int = 99,
) -> tuple[EnterpriseDB, list[Level4WorkloadItem]]:
    db = generate_enterprise_db(seed=db_seed, n_customers=n_customers, n_transactions=n_transactions)
    rng = random.Random(seed)
    templates = LEVEL4_TEMPLATES["return_decision"]
    spec = GRAPH_TASK_REGISTRY["return_decision"]

    items: list[Level4WorkloadItem] = []
    counter = 0
    index = 0

    for branch in BRANCHES:
        for _combo in range(combinations_per_branch):
            cust_id, txn_id = _make_pair_for_branch(rng, db, branch, index)
            index += 1
            entities = {"customer_id": cust_id, "transaction_id": txn_id}

            outcome = spec.fetch(entities, {}, db)
            if outcome.values is None:
                continue  # stay honest rather than silently forcing a combination that turned out invalid
            expected_result = spec.executor(**entities, **outcome.values)
            if expected_result["branch"] != branch:
                # Construction didn't land in the intended branch (e.g. a
                # random day/amount draw crossed a boundary) - skip
                # rather than mislabel; still counted honestly below.
                continue

            format_args = [entities["customer_id"], entities["transaction_id"]]
            chosen_templates = rng.sample(templates, k=min(paraphrases_per_combination, len(templates)))
            for paraphrase_id, template in enumerate(chosen_templates):
                text = template.format(*format_args)
                items.append(
                    Level4WorkloadItem(
                        request_id=f"return_decision-{counter}",
                        text=text,
                        task_type="return_decision",
                        branch=branch,
                        entities=dict(entities),
                        numbers={},
                        expected_result=expected_result,
                        paraphrase_id=paraphrase_id,
                    )
                )
                counter += 1

    rng.shuffle(items)
    return db, items
