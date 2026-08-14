"""Render Level 4's ``return_decision`` result (a dict — e.g.
``{"decision": "APPROVE", "branch": "fraud_cleared", "refund_amount":
900.0, "fraud_score": 0.1}``) as natural language, the same role
``workflow_rendering.py`` plays for Level 3. Kept as a separate module
for the same reason Level 3's was: rendering and correctness-checking
are different concerns that happen to need to agree on a shape.
"""

from __future__ import annotations


def render_graph_answer(task_type: str, result: dict) -> str:
    if task_type != "return_decision":
        return str(result)

    branch = result["branch"]
    if branch == "not_eligible":
        return "Your return request is denied: this transaction is not eligible for a refund."
    if branch == "auto_approve_low_value":
        return f"Your return request is approved. Refund amount: ${result['refund_amount']:.2f}."
    if branch == "fraud_cleared":
        return (
            f"Your return request is approved after a fraud review. "
            f"Refund amount: ${result['refund_amount']:.2f}."
        )
    if branch == "fraud_flagged":
        return "Your return request is denied: flagged for fraud review."
    return str(result)
