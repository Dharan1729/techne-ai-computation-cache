"""Paraphrase templates for Level 3 tasks. Placeholders are positional
across entity slots first (in ``WorkflowTaskSpec.entity_slots`` order),
then number slots (in ``number_slots`` order) — the same convention
Level 1/2's templates used. Unlike Level 1/2, positional order here only
matters for *generating* concrete request text; *extraction* is
order-independent (see aicache/binding/entity_extractor.py), so a
template that states its entities out of order is not the correctness
hazard it was at Level 1/2 (see docs/experiment-log.md EXP-002/EXP-003
for that history) — it would still extract correctly. These templates
keep a natural order anyway, matching the wording already used as
``example_phrasings`` in ``aicache/computation/workflow_tasks.py``.
"""

from __future__ import annotations

LEVEL3_TEMPLATES: dict[str, list[str]] = {
    "refund_eligibility": [
        "Is customer {0} eligible for a refund on transaction {1}?",
        "Check whether {0} can get a refund for {1}.",
        "What is the refund eligibility for customer {0}'s transaction {1}?",
        "Can {0} be refunded for {1}?",
    ],
    "loyalty_discount": [
        "What discount does customer {0} get on a ${1} order?",
        "Calculate the loyalty discount for {0} on an order total of ${1}.",
        "Apply {0}'s tier discount to a ${1} purchase.",
    ],
    "shipping_cost": [
        "What is the shipping cost for customer {0} shipping {2}kg to {1}?",
        "Calculate shipping for {0}, {1}, {2}kg.",
        "How much does it cost for {0} to ship {2}kg of goods to {1}?",
    ],
    "customer_spend_summary": [
        "Summarize the total spend for customer {0}.",
        "What is the completed transaction spend summary for {0}?",
        "How much has {0} spent across their completed transactions?",
    ],
    "transaction_risk": [
        "What is the risk classification for {0}'s transaction {1}?",
        "Assess the risk of transaction {1} for customer {0}.",
        "Is transaction {1} from {0} high risk?",
    ],
}
