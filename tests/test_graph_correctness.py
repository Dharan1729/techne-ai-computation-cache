from aicache.llm.graph_rendering import render_graph_answer
from aicache.metrics.graph_correctness import is_graph_correct

_CASES = [
    {"decision": "DENY", "branch": "not_eligible", "refund_amount": 0.0, "fraud_score": None},
    {"decision": "APPROVE", "branch": "auto_approve_low_value", "refund_amount": 245.0, "fraud_score": None},
    {"decision": "APPROVE", "branch": "fraud_cleared", "refund_amount": 900.0, "fraud_score": 0.1},
    {"decision": "DENY", "branch": "fraud_flagged", "refund_amount": 0.0, "fraud_score": 0.9},
]


def test_renderer_and_checker_round_trip():
    for case in _CASES:
        text = render_graph_answer("return_decision", case)
        assert is_graph_correct("return_decision", text, case), (text, case)


def test_checker_rejects_wrong_refund_amount():
    text = render_graph_answer(
        "return_decision", {"decision": "APPROVE", "branch": "fraud_cleared", "refund_amount": 900.0, "fraud_score": 0.1}
    )
    wrong = {"decision": "APPROVE", "branch": "fraud_cleared", "refund_amount": 111.0, "fraud_score": 0.1}
    assert not is_graph_correct("return_decision", text, wrong)


def test_checker_rejects_approve_vs_deny_mismatch():
    approved_text = render_graph_answer(
        "return_decision", {"decision": "APPROVE", "branch": "auto_approve_low_value", "refund_amount": 100.0, "fraud_score": None}
    )
    expected_deny = {"decision": "DENY", "branch": "not_eligible", "refund_amount": 0.0, "fraud_score": None}
    assert not is_graph_correct("return_decision", approved_text, expected_deny)


def test_this_regression_this_bug_was_caught_by():
    """Regression test for the real bug found during Level 4 development:
    WorkflowComputationCache hardcoded Level 3's renderer, so a
    return_decision result rendered as a raw dict repr (e.g.
    "{'decision': 'APPROVE', ...}") instead of natural language — and
    the substring check for "approved" never matched "'APPROVE'"
    (different case, no trailing "d"), silently marking every APPROVE
    case incorrect. See docs/level4.md's "Bugs found and fixed"."""
    raw_repr = str({"decision": "APPROVE", "branch": "auto_approve_low_value", "refund_amount": 100.0, "fraud_score": None})
    expected = {"decision": "APPROVE", "branch": "auto_approve_low_value", "refund_amount": 100.0, "fraud_score": None}
    assert not is_graph_correct("return_decision", raw_repr, expected)

    correct_text = render_graph_answer("return_decision", expected)
    assert is_graph_correct("return_decision", correct_text, expected)
