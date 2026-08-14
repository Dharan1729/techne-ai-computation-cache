"""Correctness checking for Level 4's ``return_decision`` — the same
role ``workflow_correctness.py`` plays for Level 3's flat tasks. What
actually matters for a return decision is the decision itself
(APPROVE/DENY) and, when approved, the refund amount — not whether the
rendered text happens to mention "fraud review", since a *stale* fraud
sub-fetch (Level 4's baseline G) produces text that reads identically
to a fresh one right up until the underlying score has actually moved
past the risk threshold. Checking decision + amount is what actually
detects that: a stale-fraud false APPROVE shows up as a decision
mismatch against independently-recomputed ground truth, exactly the
same way a stale refund_eligibility answer showed up as a mismatch at
Level 3.
"""

from __future__ import annotations

import re

_REL_TOL = 1e-3
_ABS_TOL = 1e-3


def _numbers_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_ABS_TOL, _REL_TOL * abs(b))


def _parse_dollar_amount(text: str) -> float | None:
    dollar_matches = re.findall(r"\$([\d,]+\.?\d*)", text)
    if dollar_matches:
        return float(dollar_matches[-1].replace(",", ""))
    return None


def _said_approved(text_lower: str) -> bool:
    return "approved" in text_lower and "denied" not in text_lower


def is_graph_correct(task_type: str, served_text: str, expected: dict) -> bool:
    if task_type != "return_decision":
        return False

    text_lower = served_text.lower()
    predicted_approved = _said_approved(text_lower)
    expected_approved = expected["decision"] == "APPROVE"
    if predicted_approved != expected_approved:
        return False
    if not expected_approved:
        return True

    amt = _parse_dollar_amount(served_text)
    return amt is not None and _numbers_close(amt, expected["refund_amount"])
