"""Level 3's computation cache: identify -> extract entities -> validate
-> fetch live state -> execute. Two configurations share this one class
via a single constructor flag, deliberately — the point of the
freshness ablation (baseline E) is that it is *identical* to the
primary condition (baseline D) in every way except one:

    D (``use_stale_cache=False``, the primary condition): every lookup
       re-fetches live state from the database. This is the condition
       whose correctness this whole Level 3 phase is trying to verify.

    E (``use_stale_cache=True``, an ablation, not a competitor —
       see docs/level3.md): the first fetch for a given (task, entity)
       combination is memoized and reused on every subsequent lookup,
       even after the underlying database has been mutated. This
       exists specifically to demonstrate *why* D always re-fetches —
       it is expected to serve stale, wrong answers after a mutation,
       and that is the point, not a bug in E.

No LLM call happens anywhere in this class, on a hit or a fallback-free
path — identification is an embedding lookup, extraction is regex,
validation is a syntactic check, and execution is a pure function. A
fallback (identification/extraction/validation failure) is signaled to
the caller, which is expected to route it to an LLM itself — mirroring
``aicache/cache/computation.py``'s Level 1/2 design exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..binding.entity_extractor import extract_entities_and_numbers
from ..enterprise.db import EnterpriseDB
from ..matching.task_identification import TaskIdentifier


@dataclass
class LatencyBreakdown:
    identify_ms: float = 0.0
    extract_ms: float = 0.0
    validate_ms: float = 0.0
    fetch_ms: float = 0.0
    execute_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.identify_ms
            + self.extract_ms
            + self.validate_ms
            + self.fetch_ms
            + self.execute_ms
        )


@dataclass
class WorkflowLookupResult:
    hit: bool
    task_type: str | None = None
    similarity: float = -1.0
    result: dict | None = None
    served_text: str | None = None
    reason: str = ""
    used_stale_fetch: bool = False
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)


class WorkflowComputationCache:
    def __init__(
        self,
        db: EnterpriseDB,
        identifier: TaskIdentifier,
        use_stale_cache: bool = False,
    ) -> None:
        self.db = db
        self.identifier = identifier
        self.use_stale_cache = use_stale_cache
        # (task_type, sorted entity items) -> fetched values. Only ever
        # written to / read from when use_stale_cache=True — see the
        # class docstring; this dict staying empty is what makes D and E
        # behave identically until a mutation exposes the difference.
        self._fetch_memo: dict[tuple, dict] = {}

    def lookup(self, text: str) -> WorkflowLookupResult:
        lat = LatencyBreakdown()

        t0 = time.perf_counter()
        ident = self.identifier.identify(text)
        lat.identify_ms = (time.perf_counter() - t0) * 1000
        if ident.task_spec is None:
            return WorkflowLookupResult(
                hit=False,
                similarity=ident.similarity,
                reason="task_identification_below_threshold",
                latency=lat,
            )
        spec = ident.task_spec
        task_type = ident.task_type

        t0 = time.perf_counter()
        extraction = extract_entities_and_numbers(text, spec.entity_slots, spec.number_slots)
        lat.extract_ms = (time.perf_counter() - t0) * 1000
        if extraction.missing_slots:
            return WorkflowLookupResult(
                hit=False,
                task_type=task_type,
                similarity=ident.similarity,
                reason=f"missing_slots:{extraction.missing_slots}",
                latency=lat,
            )

        t0 = time.perf_counter()
        valid = all(
            isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
            for v in extraction.numbers.values()
        )
        lat.validate_ms = (time.perf_counter() - t0) * 1000
        if not valid:
            return WorkflowLookupResult(
                hit=False,
                task_type=task_type,
                similarity=ident.similarity,
                reason="unsafe_parameter",
                latency=lat,
            )

        cache_key = (task_type, tuple(sorted(extraction.entities.items())))
        used_stale = False
        if self.use_stale_cache and cache_key in self._fetch_memo:
            fetched = self._fetch_memo[cache_key]
            lat.fetch_ms = 0.0  # no fetch happened at all — the point of the ablation
            used_stale = True
        else:
            t0 = time.perf_counter()
            outcome = spec.fetch(extraction.entities, extraction.numbers, self.db)
            lat.fetch_ms = (time.perf_counter() - t0) * 1000
            if outcome.values is None:
                return WorkflowLookupResult(
                    hit=False,
                    task_type=task_type,
                    similarity=ident.similarity,
                    reason=outcome.error or "fetch_failed",
                    latency=lat,
                )
            fetched = outcome.values
            if self.use_stale_cache:
                self._fetch_memo[cache_key] = fetched

        t0 = time.perf_counter()
        try:
            result = spec.executor(**extraction.entities, **extraction.numbers, **fetched)
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            lat.execute_ms = (time.perf_counter() - t0) * 1000
            return WorkflowLookupResult(
                hit=False,
                task_type=task_type,
                similarity=ident.similarity,
                reason=f"execution_error:{exc}",
                latency=lat,
            )
        lat.execute_ms = (time.perf_counter() - t0) * 1000

        from ..llm.workflow_rendering import render_workflow_answer

        served_text = render_workflow_answer(task_type, result)

        return WorkflowLookupResult(
            hit=True,
            task_type=task_type,
            similarity=ident.similarity,
            result=result,
            served_text=served_text,
            used_stale_fetch=used_stale,
            latency=lat,
        )
