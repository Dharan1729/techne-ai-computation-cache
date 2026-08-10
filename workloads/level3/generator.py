"""Generates the primary Level 3 comparison workload: a shuffled list of
natural-language requests against a **static** (for the duration of
this workload) enterprise database, one instance of each of the 5
workflow task families, with several distinct entity combinations and
paraphrases each — the same structural-repetition principle Level 1/2
used (`workloads/generators.py`), extended to entities instead of bare
numbers.

Ground truth is computed once, here, independent of any baseline —
directly via each task's own ``fetch`` + ``executor``, exactly what a
computation-cache hit would do, but called by the generator, not read
back from a cache instance under test.

This module deliberately does NOT include database mutations — that's
the separate, dedicated freshness experiment
(``benchmarks/level3/freshness_experiment.py``), which needs a
different (sequential, stateful) shape than this shuffled-and-parallel
one. Keeping them apart matches how the design brief itself scoped the
two experiments as distinct sections.
"""

from __future__ import annotations

import random

from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import EnterpriseDB, generate_enterprise_db

from .templates import LEVEL3_TEMPLATES
from .types import Level3WorkloadItem

ZONES = ["zone-1", "zone-2", "zone-3"]


def _pick_customer_and_transaction(rng: random.Random, db: EnterpriseDB) -> tuple[str, dict]:
    """Picks a transaction first, then uses ITS OWNER as the customer —
    guarantees the pair is always valid (the transaction really does
    belong to that customer), rather than picking both independently
    and risking a mismatch that would make every such request an
    automatic (uninteresting) fallback."""
    txn = rng.choice(list(db.transactions.values()))
    return txn["customer_id"], txn


def generate_level3_workload(
    db_seed: int = 42,
    n_customers: int = 80,
    n_transactions: int = 200,
    combinations_per_task: int = 8,
    paraphrases_per_combination: int = 3,
    seed: int = 99,
) -> tuple[EnterpriseDB, list[Level3WorkloadItem]]:
    db = generate_enterprise_db(seed=db_seed, n_customers=n_customers, n_transactions=n_transactions)
    rng = random.Random(seed)
    customer_ids = list(db.customers.keys())

    items: list[Level3WorkloadItem] = []
    counter = 0

    for task_type, spec in WORKFLOW_TASK_REGISTRY.items():
        templates = LEVEL3_TEMPLATES[task_type]

        for _combo in range(combinations_per_task):
            entities: dict[str, str] = {}
            numbers: dict[str, float] = {}

            needs_txn = any(s.name == "transaction_id" for s in spec.entity_slots)
            if needs_txn:
                customer_id, txn = _pick_customer_and_transaction(rng, db)
                entities["customer_id"] = customer_id
                entities["transaction_id"] = txn["transaction_id"]
            else:
                entities["customer_id"] = rng.choice(customer_ids)

            if any(s.name == "zone" for s in spec.entity_slots):
                entities["zone"] = rng.choice(ZONES)

            for num_name in spec.number_slots:
                if num_name == "order_total":
                    numbers[num_name] = round(rng.uniform(20, 1000), 2)
                elif num_name == "weight":
                    numbers[num_name] = round(rng.uniform(0.5, 30), 1)

            # Oracle: compute ground truth directly, independent of any
            # baseline's own cache instance.
            outcome = spec.fetch(entities, numbers, db)
            if outcome.values is None:
                # Extremely rare given _pick_customer_and_transaction's
                # guarantee, but stay honest rather than silently
                # dropping a combination that turned out invalid for
                # some other reason (e.g. a customer row somehow
                # missing) — skip and move on.
                continue
            expected_result = spec.executor(**entities, **numbers, **outcome.values)

            # Render values in entity_slots order, then number_slots order.
            format_args = [entities[s.name] for s in spec.entity_slots] + [
                numbers[n] for n in spec.number_slots
            ]
            chosen_templates = rng.sample(
                templates, k=min(paraphrases_per_combination, len(templates))
            )
            for paraphrase_id, template in enumerate(chosen_templates):
                text = template.format(*format_args)
                items.append(
                    Level3WorkloadItem(
                        request_id=f"{task_type}-{counter}",
                        text=text,
                        task_type=task_type,
                        entities=dict(entities),
                        numbers=dict(numbers),
                        expected_result=expected_result,
                        paraphrase_id=paraphrase_id,
                    )
                )
                counter += 1

    rng.shuffle(items)
    return db, items
