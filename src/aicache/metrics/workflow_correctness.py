"""Correctness checking for Level 3's structured (multi-field) results —
the workflow-task counterpart to ``correctness.py``'s single-number
``is_correct``. A refund decision, a discount breakdown, a risk
classification aren't reducible to "the last number in the text" the
way arithmetic answers are (``parse_final_number``'s whole reason for
existing, see EXP-007) — some fields are booleans, one is a category
string, most tasks have a "headline" dollar figure plus supporting
detail. Each task type gets its own small, honest parser rather than
one generic regex, mirroring the entity-extraction module's per-slot
design rather than pretending a single mechanism covers every output
shape.

Tested only against this repo's own renderer (``workflow_rendering.py``)
— real-backend validation of *this specific parsing layer* is
documented future work in docs/level3.md, the same "simulated first"
scope boundary Level 1/2 used before EXP-007/008 validated it against
real models.
"""

from __future__ import annotations

import math
import re

_REL_TOL = 1e-3
_ABS_TOL = 1e-3


def _numbers_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_ABS_TOL, _REL_TOL * abs(b))


def _parse_dollar_amount(text: str) -> float | None:
    """Prefer an explicit $-prefixed figure (what this repo's own
    renderer always produces); fall back to the last plain number in
    the text otherwise, same reasoning as parse_final_number."""
    dollar_matches = re.findall(r"\$([\d,]+\.?\d*)", text)
    if dollar_matches:
        return float(dollar_matches[-1].replace(",", ""))
    from .correctness import parse_final_number

    return parse_final_number(text)


def _said_yes(text_lower: str) -> bool:
    return bool(re.search(r"\byes\b", text_lower)) and "not eligible" not in text_lower


def is_workflow_correct(task_type: str, served_text: str, expected: dict) -> bool:
    text_lower = served_text.lower()

    if task_type == "refund_eligibility":
        predicted_eligible = _said_yes(text_lower)
        if predicted_eligible != expected["eligible"]:
            return False
        if not expected["eligible"]:
            return True
        amt = _parse_dollar_amount(served_text)
        return amt is not None and _numbers_close(amt, expected["refund_amount"])

    if task_type == "loyalty_discount":
        amt = _parse_dollar_amount(served_text)
        return amt is not None and _numbers_close(amt, expected["final_amount"])

    if task_type == "shipping_cost":
        if expected["free_shipping"]:
            return "free" in text_lower
        amt = _parse_dollar_amount(served_text)
        return amt is not None and _numbers_close(amt, expected["cost"])

    if task_type == "customer_spend_summary":
        amt = _parse_dollar_amount(served_text)
        return amt is not None and _numbers_close(amt, expected["total_spend"])

    if task_type == "transaction_risk":
        for cat in ("low", "medium", "high"):
            if cat in text_lower:
                return cat == expected["risk"]
        return False

    return False
