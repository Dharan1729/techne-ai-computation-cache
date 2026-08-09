"""Baseline D (experimental system): the computation cache.

    request
       -> task/computation identification (embedding nearest-neighbor
          against canonical task example phrasings — no LLM call)
       -> retrieve reusable computation (TASK_REGISTRY lookup by task_type)
       -> extract + bind fresh parameters (positional numeric extraction)
       -> validate (parameter count/type/finiteness)
       -> execute directly (pure Python function — no LLM call on a hit)
       -> fresh, correct answer

The whole point (H3/H4): under this design, a "hit" never risks serving a
stale numeric answer, because nothing is served from a store keyed by a
past request — the parameters are always freshly bound and freshly
computed. The cost is scope: this only works because Level 1/2 tasks are
closed-form and deterministic. It does not generalize to Level 3/4 without
a different representation (see docs/architecture.md) — that's flagged as
scope, not hidden.

Any failure to identify or validate falls back to the LLM (Outcome.FALLBACK)
rather than guessing — this is the "fallback to LLM when binding fails"
requirement from the brief's §11.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..binding.extractor import extract_numbers
from ..computation.tasks import TASK_REGISTRY
from ..matching.task_identification import TaskIdentifier
from ..validation.validator import validate_binding


@dataclass
class ComputationLookupResult:
    hit: bool
    task_type: str | None = None
    similarity: float = -1.0
    answer: float | None = None
    reason: str = ""  # populated when hit=False (why it fell back)


class ComputationCache:
    def __init__(self, identifier: TaskIdentifier | None = None) -> None:
        self.identifier = identifier or TaskIdentifier()

    def lookup(self, text: str) -> ComputationLookupResult:
        ident = self.identifier.identify(text)
        if ident.task_spec is None:
            return ComputationLookupResult(
                hit=False,
                similarity=ident.similarity,
                reason="task_identification_below_threshold",
            )

        raw_numbers = extract_numbers(text)
        validation = validate_binding(ident.task_spec, raw_numbers)
        if not validation.ok:
            return ComputationLookupResult(
                hit=False,
                task_type=ident.task_type,
                similarity=ident.similarity,
                reason=validation.reason,
            )

        try:
            answer = ident.task_spec.bind_and_execute(raw_numbers)
        except Exception as exc:  # pragma: no cover - defensive
            return ComputationLookupResult(
                hit=False,
                task_type=ident.task_type,
                similarity=ident.similarity,
                reason=f"execution_error: {exc}",
            )

        return ComputationLookupResult(
            hit=True,
            task_type=ident.task_type,
            similarity=ident.similarity,
            answer=answer,
        )
