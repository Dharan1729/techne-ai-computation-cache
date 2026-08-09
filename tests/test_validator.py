import math

from aicache.computation.tasks import TASK_REGISTRY
from aicache.validation.validator import validate_binding


def test_valid_binding_ok():
    spec = TASK_REGISTRY["add"]
    result = validate_binding(spec, [10.0, 20.0])
    assert result.ok


def test_missing_parameters_rejected():
    spec = TASK_REGISTRY["compound_interest"]  # needs 3
    result = validate_binding(spec, [1000.0, 10.0])
    assert not result.ok
    assert "missing_parameters" in result.reason


def test_nan_rejected():
    spec = TASK_REGISTRY["add"]
    result = validate_binding(spec, [float("nan"), 1.0])
    assert not result.ok
    assert "unsafe_parameter" in result.reason


def test_inf_rejected():
    spec = TASK_REGISTRY["add"]
    result = validate_binding(spec, [float("inf"), 1.0])
    assert not result.ok


def test_extra_numbers_are_fine_uses_first_n():
    spec = TASK_REGISTRY["add"]  # needs 2
    result = validate_binding(spec, [10.0, 20.0, 30.0])
    assert result.ok
