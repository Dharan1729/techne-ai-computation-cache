"""Parameter-binding validation.

Per the brief's §12, correctness is first-class: a computation-cache hit
that can't be validated must fall back to the LLM rather than silently
executing on bad parameters. This module implements the validation
dimensions that are actually testable at Level 1/2 (parameter count,
finiteness, type); the others the brief lists — tool compatibility,
policy version, data freshness, authorization context — don't apply to
closed-form arithmetic/formula tasks and become relevant starting at
Level 3 (see docs/architecture.md and docs/eviction-future-work.md-style
scoping note there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..computation.tasks import TaskSpec


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


def validate_binding(task_spec: TaskSpec, raw_numbers: list[float]) -> ValidationResult:
    needed = len(task_spec.params)
    if len(raw_numbers) < needed:
        return ValidationResult(
            False, f"missing_parameters: need {needed}, extracted {len(raw_numbers)}"
        )

    candidates = raw_numbers[:needed]
    for v in candidates:
        if not isinstance(v, (int, float)):
            return ValidationResult(False, "unsafe_parameter: non-numeric")
        if math.isnan(v) or math.isinf(v):
            return ValidationResult(False, "unsafe_parameter: nan_or_inf")

    return ValidationResult(True)
