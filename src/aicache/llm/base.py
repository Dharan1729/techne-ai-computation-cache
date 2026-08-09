"""LLM client abstraction.

Every baseline (no-cache, exact cache, semantic cache, computation cache)
calls through this interface for the "ask the model" step. Swapping the
backend (simulated vs. a real provider) changes nothing about the cache
logic being tested — that is the point of the abstraction: it isolates
the *caching* experiment from *which model answers on a miss*.

Two backends ship here:

- ``SimulatedLLM`` (see ``simulated.py``) — deterministic, offline,
  zero-cost. Computes real answers for the arithmetic/formula workloads
  (so correctness checking is meaningful) and reports token counts and
  latency drawn from a documented, labeled model — not measurements of
  a real API. Every artifact this backend produces is tagged
  ``backend="simulated"`` so results can never be mistaken for a real
  benchmark.
- ``GeminiLLM`` (see ``gemini.py``) — real inference via the Gemini API.
  Inactive unless ``GEMINI_API_KEY`` is set; the harness must be told
  explicitly to use it (see ``benchmarks/run_experiment.py --backend``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Result of one LLM call, uniform across backends."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str
    backend: str  # "simulated" | "gemini" | ...
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    """Minimal interface every backend implements."""

    backend_name: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str, *, task_hint: dict[str, Any] | None = None) -> LLMResponse:
        """Answer ``prompt``.

        ``task_hint`` is optional structured info (e.g. a canonical
        computation signature + bound parameters) that a real backend
        would ignore (it only sees the prompt text, like a real API
        call) but that the simulated backend uses to compute a correct
        answer deterministically instead of hallucinating one — the
        simulated backend is standing in for "a model that is good at
        this task," not testing model quality itself. That is a
        deliberate scope boundary: this project studies *caching*, not
        *model correctness*.
        """
        raise NotImplementedError
