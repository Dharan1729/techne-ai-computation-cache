"""Task/computation identification: map a natural-language request to a
canonical computation signature in some registry of "identifiable"
specs (Level 1/2's ``TASK_REGISTRY`` of pure ``TaskSpec``s, Level 3's
``WORKFLOW_TASK_REGISTRY`` of ``WorkflowTaskSpec``s, or any future
registry — see ``aicache/computation/workflow_tasks.py``).

This is deliberately the simplest thing that could work, per the brief's
§14 instruction not to reach for LLM-based extraction until the
deterministic version is validated: embed the request, compare against
embeddings of each canonical task's example phrasings, and take the
task whose closest example exceeds a similarity threshold. No LLM call
is made for identification — this step is (and must be, to keep the
computation-cache hit path free of the very LLM call it's trying to
avoid) itself a lightweight embedding lookup, not a generative one.

Deliberately reused as-is for Level 3 rather than forked: the mechanism
(embed request, nearest-neighbor against example phrasings) doesn't
change between "this is an arithmetic op" and "this is an enterprise
workflow" — only *what's in the registry* changes, so this class takes
the registry as a parameter instead of importing ``TASK_REGISTRY``
directly. A single call site can even search a merged registry (Level
1/2 arithmetic ∪ Level 3 workflows at once) by passing a combined dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..computation.tasks import TASK_REGISTRY
from .embedder import Embedder


@dataclass
class IdentificationResult:
    task_type: str | None
    task_spec: Any | None  # TaskSpec (Level 1/2) or WorkflowTaskSpec (Level 3)
    similarity: float
    matched_example: str | None


class TaskIdentifier:
    # 0.52, calibrated against langcache-embed-v1 on this project's own
    # Level 1/2 workload (see docs/experiment-log.md): zero misidentified
    # matches observed at this threshold across 150 generated requests,
    # while still accepting ~82% of genuinely-correct matches. Not a
    # theoretical optimum — an empirically-checked value for this
    # embedding model and this workload, documented as such rather than
    # asserted as a universal default. Level 3 re-validates this
    # threshold on its own workload rather than assuming it transfers —
    # see docs/level3.md.
    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.52,
        registry: Mapping[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self.registry: Mapping[str, Any] = TASK_REGISTRY if registry is None else registry
        self._example_texts: list[str] = []
        self._example_task_types: list[str] = []
        self._example_vecs: np.ndarray | None = None

    def _build_index(self) -> None:
        if self._example_vecs is not None:
            return
        for task_type, spec in self.registry.items():
            for phrasing in spec.example_phrasings:
                self._example_texts.append(phrasing)
                self._example_task_types.append(task_type)
        self._example_vecs = self.embedder.encode(self._example_texts)

    def identify(self, request_text: str) -> IdentificationResult:
        self._build_index()
        assert self._example_vecs is not None
        query = self.embedder.encode_one(request_text)
        sims = self._example_vecs @ query
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.threshold:
            return IdentificationResult(None, None, best_sim, None)

        task_type = self._example_task_types[best_idx]
        return IdentificationResult(
            task_type=task_type,
            task_spec=self.registry[task_type],
            similarity=best_sim,
            matched_example=self._example_texts[best_idx],
        )
