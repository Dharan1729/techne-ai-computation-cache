from aicache.llm.workflow_rendering import render_workflow_answer
from aicache.metrics.workflow_correctness import is_workflow_correct

_CASES = [
    ("refund_eligibility", {"eligible": True, "refund_amount": 500.0, "reason": "within_refund_window"}),
    ("refund_eligibility", {"eligible": False, "refund_amount": 0.0, "reason": "outside_refund_window"}),
    (
        "loyalty_discount",
        {"original_amount": 400.0, "discount_percent": 10.0, "discount_amount": 40.0, "final_amount": 360.0},
    ),
    ("shipping_cost", {"cost": 0.0, "free_shipping": True, "zone": "zone-2"}),
    ("shipping_cost", {"cost": 9.36, "free_shipping": False, "zone": "zone-1"}),
    ("customer_spend_summary", {"transaction_count": 3, "total_spend": 150.0, "average_spend": 50.0}),
    ("transaction_risk", {"risk": "high", "amount": 200.0, "status": "disputed"}),
]


def test_renderer_and_checker_round_trip():
    for task_type, result in _CASES:
        text = render_workflow_answer(task_type, result)
        assert is_workflow_correct(task_type, text, result), (task_type, text, result)


def test_checker_rejects_wrong_expected_values():
    text = render_workflow_answer(
        "refund_eligibility", {"eligible": True, "refund_amount": 500.0, "reason": "within_refund_window"}
    )
    wrong_expected = {"eligible": True, "refund_amount": 999.0, "reason": "within_refund_window"}
    assert not is_workflow_correct("refund_eligibility", text, wrong_expected)


def test_checker_rejects_wrong_eligibility_direction():
    text = render_workflow_answer(
        "refund_eligibility", {"eligible": False, "refund_amount": 0.0, "reason": "outside_refund_window"}
    )
    wrong_expected = {"eligible": True, "refund_amount": 500.0, "reason": "within_refund_window"}
    assert not is_workflow_correct("refund_eligibility", text, wrong_expected)


def test_checker_rejects_wrong_risk_category():
    text = render_workflow_answer("transaction_risk", {"risk": "low", "amount": 10.0, "status": "completed"})
    wrong_expected = {"risk": "high", "amount": 10.0, "status": "completed"}
    assert not is_workflow_correct("transaction_risk", text, wrong_expected)


def test_customer_spend_summary_headline_field_is_the_total_not_the_average():
    """Regression test for the renderer/parser ordering bug found before
    any experiment run used it — the parser takes the LAST dollar figure
    in the text, so the renderer must put total_spend last."""
    result = {"transaction_count": 4, "total_spend": 999.0, "average_spend": 12.5}
    text = render_workflow_answer("customer_spend_summary", result)
    assert text.rstrip(".").endswith(f"${result['total_spend']:.2f}")
