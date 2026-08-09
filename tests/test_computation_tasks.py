import math

from aicache.computation.tasks import TASK_REGISTRY


def test_add():
    assert TASK_REGISTRY["add"].executor(x=10, y=20) == 30


def test_subtract():
    assert TASK_REGISTRY["subtract"].executor(x=51, y=93) == -42


def test_multiply():
    assert TASK_REGISTRY["multiply"].executor(x=712, y=381) == 271_272


def test_percentage():
    assert TASK_REGISTRY["percentage"].executor(pct=25, base=200) == 50


def test_compound_interest():
    spec = TASK_REGISTRY["compound_interest"]
    # $1000 at 10% for 2 years => 1000 * 1.1^2 = 1210
    result = spec.executor(principal=1000, rate=0.10, years=2)
    assert math.isclose(result, 1210.0)


def test_revenue():
    assert TASK_REGISTRY["revenue"].executor(price=10, quantity=5) == 50


def test_discounted_price():
    spec = TASK_REGISTRY["discounted_price"]
    assert math.isclose(spec.executor(price=100, discount=0.20), 80.0)


def test_tax():
    spec = TASK_REGISTRY["tax"]
    assert math.isclose(spec.executor(income=1000, rate=0.15), 150.0)


def test_margin():
    spec = TASK_REGISTRY["margin"]
    assert math.isclose(spec.executor(revenue=200, cost=150), 0.25)


def test_margin_zero_revenue_does_not_raise():
    spec = TASK_REGISTRY["margin"]
    assert spec.executor(revenue=0, cost=10) == 0.0


def test_bind_and_execute_uses_positional_order():
    spec = TASK_REGISTRY["compound_interest"]
    # raw numbers as extracted from text, in the template's declared order
    result = spec.bind_and_execute([1000.0, 10.0, 2.0])  # principal, rate%, years
    assert math.isclose(result, 1210.0)


def test_bind_and_execute_raises_on_too_few_numbers():
    spec = TASK_REGISTRY["add"]
    try:
        spec.bind_and_execute([1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_every_task_has_example_phrasings():
    for task_type, spec in TASK_REGISTRY.items():
        assert spec.example_phrasings, f"{task_type} has no example phrasings"
