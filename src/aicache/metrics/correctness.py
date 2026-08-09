"""Numeric correctness checking, shared by every baseline in the harness.

Comparisons are numeric (parse a number out of whatever text was served,
compare with tolerance) rather than string-exact, because the simulated
LLM, the computation cache, and a cached-and-reused answer can all format
the same number slightly differently (``"42"`` vs ``"42.0"`` vs
``"42.0000"``) without that being a correctness failure.

Parses the LAST number in the text, not the first — found the hard way
during real-backend validation (see docs/experiment-log.md, EXP-007):
the simulated LLM always answers with a single bare number, so
first-vs-last made no difference and this bug sat invisible through
every simulated-backend result in this repo. A real model (tested:
Gemini) naturally shows its work — "512 + 937 = **1449**" — so the
*first* number in the text is almost always an input operand, not the
answer; only the *last* number reliably lands on the model's final
answer across every observed response shape (inline arithmetic, a
worked multi-step solution, a bolded closing sentence). This is a
genuine, not cosmetic, distinction: the wrong choice here doesn't fail
loudly, it silently marks every correct real-model answer as incorrect.
"""

from __future__ import annotations

from ..binding.extractor import extract_numbers

# Loose enough to tolerate reasonable answer rounding (the simulated
# backend formats non-integer results to 4 decimal places; a real model
# rounds similarly — e.g. Gemini returned "1469.33" for a ground truth
# of 1469.328...), tight enough that reusing a DIFFERENT parameter
# draw's answer — the false-hit failure mode this benchmark exists to
# measure — still reliably fails the check rather than being tolerated
# away.
_REL_TOL = 1e-3
_ABS_TOL = 1e-3


def parse_final_number(text: str) -> float | None:
    numbers = extract_numbers(text)
    return numbers[-1] if numbers else None


def is_correct(served_text: str, expected: float) -> bool:
    got = parse_final_number(served_text)
    if got is None:
        return False
    return abs(got - expected) <= max(_ABS_TOL, _REL_TOL * abs(expected))
