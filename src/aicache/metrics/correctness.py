"""Numeric correctness checking, shared by every baseline in the harness.

Comparisons are numeric (parse the first number out of whatever text was
served, compare with tolerance) rather than string-exact, because the
simulated LLM, the computation cache, and a cached-and-reused answer can
all format the same number slightly differently (``"42"`` vs ``"42.0"``
vs ``"42.0000"``) without that being a correctness failure.
"""

from __future__ import annotations

from ..binding.extractor import extract_numbers

# Loose enough to tolerate reasonable answer rounding (the simulated
# backend formats non-integer results to 4 decimal places; a real model
# would round similarly), tight enough that reusing a DIFFERENT parameter
# draw's answer — the false-hit failure mode this benchmark exists to
# measure — still reliably fails the check rather than being tolerated
# away.
_REL_TOL = 1e-3
_ABS_TOL = 1e-3


def parse_first_number(text: str) -> float | None:
    numbers = extract_numbers(text)
    return numbers[0] if numbers else None


def is_correct(served_text: str, expected: float) -> bool:
    got = parse_first_number(served_text)
    if got is None:
        return False
    return abs(got - expected) <= max(_ABS_TOL, _REL_TOL * abs(expected))
