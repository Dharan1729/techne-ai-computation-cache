"""Render a Level 3 workflow's structured result (a dict — e.g.
``{"eligible": True, "refund_amount": 500.0, "reason": "..."}``) as a
natural-language answer, the way a real model would phrase it.

Used by the simulated backend (every baseline that calls "the LLM" needs
text to store/return, not a raw dict) and, if a real-backend validation
run is attempted later, as the target shape a real model's response
should be prompted to resemble. See
``aicache/metrics/workflow_correctness.py`` for the matching parser —
kept in a separate module because "how do I say the answer" and "how do
I check the answer" are different concerns that happen to need to agree
on a shape.
"""

from __future__ import annotations


def render_workflow_answer(task_type: str, result: dict) -> str:
    if task_type == "refund_eligibility":
        if result["eligible"]:
            return (
                f"Yes, this transaction is eligible for a refund of "
                f"${result['refund_amount']:.2f} ({result['reason']})."
            )
        return f"No, this transaction is not eligible for a refund ({result['reason']})."

    if task_type == "loyalty_discount":
        return (
            f"The order total is ${result['original_amount']:.2f}, with a "
            f"{result['discount_percent']:.0f}% discount of ${result['discount_amount']:.2f}, "
            f"for a final amount of ${result['final_amount']:.2f}."
        )

    if task_type == "shipping_cost":
        if result["free_shipping"]:
            return f"Shipping to {result['zone']} is free for this customer."
        return f"Shipping to {result['zone']} costs ${result['cost']:.2f}."

    if task_type == "customer_spend_summary":
        # total_spend (the headline figure) is placed LAST deliberately —
        # the parser takes the final dollar amount in the text, same
        # convention as every other renderer here (and as
        # parse_final_number at Level 1/2). An earlier version put the
        # average after the total and the parser silently picked up the
        # average instead — caught by round-trip testing before this
        # ever reached a real experiment run (see docs/level3.md).
        return (
            f"This customer has {result['transaction_count']} completed transactions "
            f"averaging ${result['average_spend']:.2f} each, for a total spend of "
            f"${result['total_spend']:.2f}."
        )

    if task_type == "transaction_risk":
        return (
            f"This transaction is classified as {result['risk']} risk "
            f"(amount ${result['amount']:.2f}, status {result['status']})."
        )

    return str(result)
