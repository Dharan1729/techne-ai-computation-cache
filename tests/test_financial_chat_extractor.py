"""Slot extractor for the financial-chat integration: it must either bind
every number to the right parameter or refuse. It must never bind wrongly."""

from benchmarks.real_app.financial_chat_integration import extract_args_v2


def test_extracts_position_size_in_any_order():
    a = extract_args_v2("position_size", "Size my TSLA trade: stop 436.81, entry 476.92, risking 0.5 percent of $50000.")
    assert a.ok
    assert a.args == {
        "symbol": "TSLA", "entry_price": 476.92, "stop_price": 436.81,
        "account_size": 50000.0, "risk_percent": 0.5,
    }


def test_percent_is_not_converted_to_a_fraction():
    # The LLM router in the benchmark turned 0.5% into 0.005. A deterministic
    # extractor keeps the unit the tool expects.
    a = extract_args_v2("position_size", "Position size for AAPL: account $25000, risk 2%, entry 150, stop 140.")
    assert a.ok and a.args["risk_percent"] == 2.0


def test_ratio_notation_is_read_as_one_number():
    a = extract_args_v2("r_multiples", "Give me a 4:1 reward-to-risk profit target on AAPL: in at 215.5, out at 200 if wrong.")
    assert a.ok and a.args["risk_multiple"] == 4


def test_refuses_when_a_number_is_unclaimed():
    r = extract_args_v2("r_multiples", "Target for a 3 to 1 payoff on AAPL, entry 100, stop 95")
    assert not r.ok


def test_refuses_when_a_parameter_is_missing():
    r = extract_args_v2("position_size", "Position size for AAPL with entry 100 and stop 95")
    assert not r.ok
