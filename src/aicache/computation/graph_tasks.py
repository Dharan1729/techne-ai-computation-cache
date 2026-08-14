"""Level 4: a single hand-specified branching workflow, ``return_decision``
— deliberately one task, not a general graph engine, per the same
narrow-scope discipline Level 3 applied (5 hand-specified task families,
not an open-ended registry — see ``docs/level3.md``'s scope notes).

The branch:

    check refund_eligibility (same rule as Level 3's task of that name)
      -> not eligible: DENY
      -> eligible, refund <= $500: APPROVE (auto, no fraud check)
      -> eligible, refund > $500:
           check fraud_score
             -> high risk: DENY (flagged)
             -> else: APPROVE

Reuses ``WorkflowTaskSpec``/``TaskIdentifier``/``WorkflowComputationCache``
completely unchanged (``aicache/computation/workflow_tasks.py``,
``aicache/matching/task_identification.py``,
``aicache/cache/workflow_computation.py``) — Level 4 adds a new *task*
whose ``fetch`` step happens to branch and conditionally call a
TTL-cacheable sub-fetch, not a new *pipeline*. The one thing Level 3's
"always_fresh" freshness model couldn't express: two sub-fetches within
the SAME task with two DIFFERENT freshness requirements (transaction/
customer/policy must always be fresh; fraud_score is legitimately
TTL-cacheable in a real system — see ``aicache/cache/ttl_cache.py``).
That's handled here by building the spec's ``fetch`` closure over a
``TTLCache`` instance (or ``None``), not by changing any shared
infrastructure.
"""

from __future__ import annotations

from typing import Callable

from ..cache.ttl_cache import TTLCache
from ..enterprise.db import EnterpriseDB
from .workflow_tasks import EntitySlot, FetchOutcome, WorkflowTaskSpec

FRAUD_CHECK_AMOUNT_THRESHOLD = 500.0  # refunds above this get a fraud check
FRAUD_SCORE_HIGH_RISK = 0.7  # score scale is [0, 1]
_NOW = 1000  # synthetic clock convention, matches workflow_tasks.py

RETURN_DECISION_ENTITY_SLOTS = [
    EntitySlot("customer_id", r"CUST-\d{4}"),
    EntitySlot("transaction_id", r"TXN-\d{6}"),
]

RETURN_DECISION_PHRASINGS = [
    "Should we approve the return for customer CUST-0042 on transaction TXN-000123?",
    "Process the return request for CUST-0042's transaction TXN-000123.",
    "What is the return decision for customer CUST-0042 regarding transaction TXN-000123?",
]


def _fetch_return_decision(
    entities: dict, numbers: dict, db: EnterpriseDB, fraud_cache: TTLCache | None
) -> FetchOutcome:
    total_ms = 0.0
    customer_id = entities["customer_id"]

    cust = db.fetch_customer(customer_id)
    total_ms += cust.latency_ms
    if not cust.found:
        return FetchOutcome(None, total_ms, "customer_not_found")

    txn = db.fetch_transaction(entities["transaction_id"])
    total_ms += txn.latency_ms
    if not txn.found:
        return FetchOutcome(None, total_ms, "transaction_not_found")
    if txn.value["customer_id"] != customer_id:
        return FetchOutcome(None, total_ms, "transaction_does_not_belong_to_customer")

    policy = db.fetch_policy(cust.value["tier"])
    total_ms += policy.latency_ms
    if not policy.found:
        return FetchOutcome(None, total_ms, "policy_not_found")

    eligible = (
        txn.value["status"] == "completed"
        and (_NOW - txn.value["day"]) <= policy.value["refund_window_days"]
    )

    fraud_score = None
    fraud_checked = False
    if eligible and txn.value["amount"] > FRAUD_CHECK_AMOUNT_THRESHOLD:
        fraud_checked = True
        if fraud_cache is not None:
            fraud_fetch, was_hit = fraud_cache.get_or_compute(
                customer_id, lambda: db.fetch_fraud_score(customer_id)
            )
            fetch_latency = 0.0 if was_hit else fraud_fetch.latency_ms
        else:
            fraud_fetch = db.fetch_fraud_score(customer_id)
            fetch_latency = fraud_fetch.latency_ms
        if not fraud_fetch.found:
            return FetchOutcome(None, total_ms, "fraud_score_not_found")
        total_ms += fetch_latency
        fraud_score = fraud_fetch.value["score"]

    return FetchOutcome(
        {
            "customer": cust.value,
            "transaction": txn.value,
            "policy": policy.value,
            "eligible": eligible,
            "fraud_score": fraud_score,
            "fraud_checked": fraud_checked,
        },
        total_ms,
    )


def _execute_return_decision(
    customer_id: str,
    transaction_id: str,
    customer: dict,
    transaction: dict,
    policy: dict,
    eligible: bool,
    fraud_score: float | None,
    fraud_checked: bool,
) -> dict:
    if not eligible:
        return {"decision": "DENY", "branch": "not_eligible", "refund_amount": 0.0, "fraud_score": None}

    refund_amount = round(transaction["amount"], 2)
    if not fraud_checked:
        return {
            "decision": "APPROVE",
            "branch": "auto_approve_low_value",
            "refund_amount": refund_amount,
            "fraud_score": None,
        }
    if fraud_score >= FRAUD_SCORE_HIGH_RISK:
        return {"decision": "DENY", "branch": "fraud_flagged", "refund_amount": 0.0, "fraud_score": fraud_score}
    return {
        "decision": "APPROVE",
        "branch": "fraud_cleared",
        "refund_amount": refund_amount,
        "fraud_score": fraud_score,
    }


def build_return_decision_spec(fraud_cache: TTLCache | None) -> WorkflowTaskSpec:
    """``fraud_cache=None`` -> baseline F (always fresh, including the
    fraud-score sub-fetch). A real ``TTLCache`` -> baseline G, which
    memoizes only the fraud-score sub-fetch for its TTL window — the
    transaction/customer/policy sub-fetches are ALWAYS fresh regardless,
    since Level 3 already established caching those is never safe; G
    only ever relaxes the one sub-fetch a real system might legitimately
    want to (see module docstring)."""
    return WorkflowTaskSpec(
        task_type="return_decision",
        entity_slots=RETURN_DECISION_ENTITY_SLOTS,
        number_slots=[],
        fetch=lambda entities, numbers, db: _fetch_return_decision(entities, numbers, db, fraud_cache),
        executor=_execute_return_decision,
        output_fields=["decision", "branch", "refund_amount", "fraud_score"],
        freshness_requirements="mixed_ttl" if fraud_cache is not None else "always_fresh",
        example_phrasings=RETURN_DECISION_PHRASINGS,
        level=4,
    )


def build_graph_registry(fraud_cache: TTLCache | None) -> dict[str, WorkflowTaskSpec]:
    """A ``{task_type: WorkflowTaskSpec}`` registry, same shape as
    ``WORKFLOW_TASK_REGISTRY`` — directly usable by ``TaskIdentifier``
    and ``WorkflowComputationCache`` with no changes to either. Built
    fresh per call (not a module-level singleton like Level 3's
    registry) because F and G need two different closures over two
    different ``fraud_cache`` values under the same task_type key."""
    return {"return_decision": build_return_decision_spec(fraud_cache)}


# The ground-truth oracle registry — always fresh, used by the workload
# generator and the simulated LLM's fallback path, exactly the role
# WORKFLOW_TASK_REGISTRY plays at Level 3. Never used by a baseline
# under test.
GRAPH_TASK_REGISTRY: dict[str, WorkflowTaskSpec] = build_graph_registry(fraud_cache=None)
