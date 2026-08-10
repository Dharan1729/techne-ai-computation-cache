"""Level 3 computation representation: workflows over live enterprise
state, as distinct from Level 1/2's pure arithmetic ``TaskSpec``.

Kept as a **separate** registry and dataclass rather than shoehorned
into ``TaskSpec`` — the shapes are genuinely different (numeric
positional params vs. named entities + a live-data fetch + a structured
output), and the brief for this phase was explicit: extend, don't
redesign, the existing Level 1/2 experiment. ``TaskIdentifier``
(``aicache/matching/task_identification.py``) was generalized to accept
either registry, which is the one piece of machinery genuinely shared.

One deliberate improvement over Level 1/2's parameter binding: entity
extraction here is **not positional**. Level 1/2's ``extract_numbers``
pulls plain numbers left-to-right and zips them against a declared
parameter order — a real, documented fragility (two template bugs
exploited exactly this; see docs/experiment-log.md EXP-002/EXP-003).
Enterprise identifiers are typed and prefixed (``CUST-0042``,
``TXN-000123``), so each entity slot is extracted by its own regex,
independent of surrounding word order — see
``aicache/binding/entity_extractor.py``.

Freshness is not a documentation-only field here the way it was at
Level 1/2 (where every task was ``"static"`` and nothing read the
field). Every Level 3 task is ``"always_fresh"`` by construction: the
fetch step re-reads live state on every single invocation, hit or
fallback, cached computation or not. What Level 3's cache reuses is the
*plan* — which entities to extract, which tables to fetch, which
formula to apply — never the fetched values or the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..enterprise.db import EnterpriseDB, FetchResult


@dataclass(frozen=True)
class EntitySlot:
    name: str
    pattern: str  # regex; matched independently of the other slots


@dataclass
class FetchOutcome:
    values: dict | None  # None if a required entity/row wasn't found
    latency_ms: float
    error: str | None = None  # e.g. "customer_not_found"


@dataclass(frozen=True)
class WorkflowTaskSpec:
    task_type: str
    entity_slots: list[EntitySlot]
    number_slots: list[str]  # names of plain-number params, if any (e.g. "order_total")
    # (entities, numbers, db) -> FetchOutcome. Encapsulates the whole
    # "fetch plan" for this task as one function rather than a generic
    # composable DAG — a deliberate scope limit (see docs/level3.md):
    # conditional / data-dependent fetch plans are Level 4 territory.
    fetch: Callable[[dict, dict, EnterpriseDB], FetchOutcome]
    executor: Callable[..., dict]  # (**entities, **numbers, **fetched) -> structured result dict
    output_fields: list[str]
    freshness_requirements: str
    example_phrasings: list[str]
    level: int = 3


WORKFLOW_TASK_REGISTRY: dict[str, WorkflowTaskSpec] = {}


def _register(spec: WorkflowTaskSpec) -> None:
    WORKFLOW_TASK_REGISTRY[spec.task_type] = spec


# --------------------------------------------------------- refund_eligibility --


def _fetch_refund_eligibility(entities: dict, numbers: dict, db: EnterpriseDB) -> FetchOutcome:
    total_ms = 0.0
    cust = db.fetch_customer(entities["customer_id"])
    total_ms += cust.latency_ms
    if not cust.found:
        return FetchOutcome(None, total_ms, "customer_not_found")

    txn = db.fetch_transaction(entities["transaction_id"])
    total_ms += txn.latency_ms
    if not txn.found:
        return FetchOutcome(None, total_ms, "transaction_not_found")
    if txn.value["customer_id"] != entities["customer_id"]:
        return FetchOutcome(None, total_ms, "transaction_does_not_belong_to_customer")

    policy = db.fetch_policy(cust.value["tier"])
    total_ms += policy.latency_ms
    if not policy.found:
        return FetchOutcome(None, total_ms, "policy_not_found")

    return FetchOutcome(
        {"customer": cust.value, "transaction": txn.value, "policy": policy.value}, total_ms
    )


def _execute_refund_eligibility(
    customer_id: str, transaction_id: str, customer: dict, transaction: dict, policy: dict
) -> dict:
    eligible = (
        transaction["status"] == "completed"
        and transaction["day"] >= 0  # placeholder for a real date check
    )
    # "within window" is the only state-dependent condition worth
    # modeling here — day is a synthetic integer clock, not a real date;
    # a transaction is refundable if it's marked completed and the
    # policy's window (in days) hasn't been exceeded, using a synthetic
    # "current day" convention: day 1000 is "now" for this workload.
    now = 1000
    within_window = (now - transaction["day"]) <= policy["refund_window_days"]
    eligible = eligible and within_window
    return {
        "eligible": eligible,
        "refund_amount": round(transaction["amount"], 2) if eligible else 0.0,
        "reason": (
            "within_refund_window"
            if eligible
            else (
                "transaction_not_completed"
                if transaction["status"] != "completed"
                else "outside_refund_window"
            )
        ),
    }


_register(
    WorkflowTaskSpec(
        task_type="refund_eligibility",
        entity_slots=[
            EntitySlot("customer_id", r"CUST-\d{4}"),
            EntitySlot("transaction_id", r"TXN-\d{6}"),
        ],
        number_slots=[],
        fetch=_fetch_refund_eligibility,
        executor=_execute_refund_eligibility,
        output_fields=["eligible", "refund_amount", "reason"],
        freshness_requirements="always_fresh",
        example_phrasings=[
            "Is customer CUST-0042 eligible for a refund on transaction TXN-000123?",
            "Check whether CUST-0042 can get a refund for TXN-000123.",
            "What is the refund eligibility for customer CUST-0042's transaction TXN-000123?",
        ],
    )
)

# --------------------------------------------------------- loyalty_discount --


def _fetch_loyalty_discount(entities: dict, numbers: dict, db: EnterpriseDB) -> FetchOutcome:
    cust = db.fetch_customer(entities["customer_id"])
    if not cust.found:
        return FetchOutcome(None, cust.latency_ms, "customer_not_found")
    policy = db.fetch_policy(cust.value["tier"])
    total_ms = cust.latency_ms + policy.latency_ms
    if not policy.found:
        return FetchOutcome(None, total_ms, "policy_not_found")
    return FetchOutcome({"customer": cust.value, "policy": policy.value}, total_ms)


def _execute_loyalty_discount(
    customer_id: str, order_total: float, customer: dict, policy: dict
) -> dict:
    discount_pct = policy["discount_percent"]
    discount_amount = round(order_total * discount_pct / 100.0, 2)
    final_amount = round(order_total - discount_amount, 2)
    return {
        "original_amount": round(order_total, 2),
        "discount_percent": discount_pct,
        "discount_amount": discount_amount,
        "final_amount": final_amount,
    }


_register(
    WorkflowTaskSpec(
        task_type="loyalty_discount",
        entity_slots=[EntitySlot("customer_id", r"CUST-\d{4}")],
        number_slots=["order_total"],
        fetch=_fetch_loyalty_discount,
        executor=_execute_loyalty_discount,
        output_fields=["original_amount", "discount_percent", "discount_amount", "final_amount"],
        freshness_requirements="always_fresh",
        example_phrasings=[
            "What discount does customer CUST-0042 get on a $250 order?",
            "Calculate the loyalty discount for CUST-0042 on an order total of $250.",
            "Apply CUST-0042's tier discount to a $250 purchase.",
        ],
    )
)

# ----------------------------------------------------------- shipping_cost --


def _fetch_shipping_cost(entities: dict, numbers: dict, db: EnterpriseDB) -> FetchOutcome:
    cust = db.fetch_customer(entities["customer_id"])
    if not cust.found:
        return FetchOutcome(None, cust.latency_ms, "customer_not_found")
    policy = db.fetch_policy(cust.value["tier"])
    rate = db.fetch_shipping_rate(entities["zone"])
    total_ms = cust.latency_ms + policy.latency_ms + rate.latency_ms
    if not policy.found:
        return FetchOutcome(None, total_ms, "policy_not_found")
    if not rate.found:
        return FetchOutcome(None, total_ms, "zone_not_found")
    return FetchOutcome(
        {"customer": cust.value, "policy": policy.value, "zone_rate": rate.value}, total_ms
    )


def _execute_shipping_cost(
    customer_id: str, zone: str, weight: float, customer: dict, policy: dict, zone_rate: float
) -> dict:
    threshold = policy["free_shipping_threshold"]
    free = threshold is not None and weight >= 0 and threshold <= 0
    # free_shipping_threshold is used here as "order value floor" in the
    # policy table but this task only takes weight — treat a threshold
    # of 0 as "always free for this tier", anything else as "not free
    # by weight alone" (a deliberately simple rule; the point is that
    # the rule depends on live policy state, not the rule's own realism).
    cost = 0.0 if free else round(zone_rate * (1 + weight / 50.0), 2)
    return {"cost": cost, "free_shipping": free, "zone": zone}


_register(
    WorkflowTaskSpec(
        task_type="shipping_cost",
        entity_slots=[
            EntitySlot("customer_id", r"CUST-\d{4}"),
            EntitySlot("zone", r"zone-\d"),
        ],
        number_slots=["weight"],
        fetch=_fetch_shipping_cost,
        executor=_execute_shipping_cost,
        output_fields=["cost", "free_shipping", "zone"],
        freshness_requirements="always_fresh",
        example_phrasings=[
            "What is the shipping cost for customer CUST-0042 shipping 5kg to zone-2?",
            "Calculate shipping for CUST-0042, zone-2, 5kg.",
            "How much does it cost for CUST-0042 to ship 5kg of goods to zone-2?",
        ],
    )
)

# ------------------------------------------------------ customer_spend_summary --


def _fetch_customer_spend_summary(entities: dict, numbers: dict, db: EnterpriseDB) -> FetchOutcome:
    cust = db.fetch_customer(entities["customer_id"])
    if not cust.found:
        return FetchOutcome(None, cust.latency_ms, "customer_not_found")
    txns = db.fetch_transactions_for_customer(entities["customer_id"])
    total_ms = cust.latency_ms + txns.latency_ms
    return FetchOutcome({"customer": cust.value, "transactions": txns.value}, total_ms)


def _execute_customer_spend_summary(customer_id: str, customer: dict, transactions: list) -> dict:
    completed = [t for t in transactions if t["status"] == "completed"]
    total_spend = round(sum(t["amount"] for t in completed), 2)
    return {
        "transaction_count": len(completed),
        "total_spend": total_spend,
        "average_spend": round(total_spend / len(completed), 2) if completed else 0.0,
    }


_register(
    WorkflowTaskSpec(
        task_type="customer_spend_summary",
        entity_slots=[EntitySlot("customer_id", r"CUST-\d{4}")],
        number_slots=[],
        fetch=_fetch_customer_spend_summary,
        executor=_execute_customer_spend_summary,
        output_fields=["transaction_count", "total_spend", "average_spend"],
        freshness_requirements="always_fresh",
        example_phrasings=[
            "Summarize the total spend for customer CUST-0042.",
            "What is the completed transaction spend summary for CUST-0042?",
            "How much has CUST-0042 spent across their completed transactions?",
        ],
    )
)

# ----------------------------------------------------------- transaction_risk --


def _fetch_transaction_risk(entities: dict, numbers: dict, db: EnterpriseDB) -> FetchOutcome:
    cust = db.fetch_customer(entities["customer_id"])
    if not cust.found:
        return FetchOutcome(None, cust.latency_ms, "customer_not_found")
    txn = db.fetch_transaction(entities["transaction_id"])
    total_ms = cust.latency_ms + txn.latency_ms
    if not txn.found:
        return FetchOutcome(None, total_ms, "transaction_not_found")
    if txn.value["customer_id"] != entities["customer_id"]:
        return FetchOutcome(None, total_ms, "transaction_does_not_belong_to_customer")
    return FetchOutcome({"customer": cust.value, "transaction": txn.value}, total_ms)


def _execute_transaction_risk(
    customer_id: str, transaction_id: str, customer: dict, transaction: dict
) -> dict:
    risk = "low"
    if transaction["status"] == "disputed":
        risk = "high"
    elif transaction["amount"] > 1000 and customer["tier"] == "bronze":
        risk = "medium"
    elif customer["account_status"] == "suspended":
        risk = "high"
    return {"risk": risk, "amount": transaction["amount"], "status": transaction["status"]}


_register(
    WorkflowTaskSpec(
        task_type="transaction_risk",
        entity_slots=[
            EntitySlot("customer_id", r"CUST-\d{4}"),
            EntitySlot("transaction_id", r"TXN-\d{6}"),
        ],
        number_slots=[],
        fetch=_fetch_transaction_risk,
        executor=_execute_transaction_risk,
        output_fields=["risk", "amount", "status"],
        freshness_requirements="always_fresh",
        example_phrasings=[
            "What is the risk classification for CUST-0042's transaction TXN-000123?",
            "Assess the risk of transaction TXN-000123 for customer CUST-0042.",
            "Is transaction TXN-000123 from CUST-0042 high risk?",
        ],
    )
)
