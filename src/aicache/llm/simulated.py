"""Simulated LLM backend.

Every number this backend produces is derived, not measured — no network
call, no real model. It exists so the caching experiment (the actual
subject of this repo) can be built, debugged, and run to completion without
depending on API access or spending money, per the brief's own instruction
to isolate "the representation problem" from "the LLM-extraction problem"
before wiring up a real model.

What's real: the token-count heuristic (chars // 4, the same rough
approximation used elsewhere in caching literature and in this author's
other projects) and the per-token cost table (`pricing.py`, sourced from
published Claude API pricing). What's synthetic: the latency distribution
(a lognormal shaped to resemble typical small-prompt API latency, not a
measurement) and — for anything outside the deterministic workload
families this repo studies — the "answer" itself.

Every artifact carries backend="simulated" so it can never be mistaken for
a real benchmark result in a plot, CSV, or README table.
"""

from __future__ import annotations

import random
import time
from typing import Any

from .base import LLMClient, LLMResponse
from .pricing import DEFAULT_MODEL

# Latency model: lognormal, shaped (not measured) to resemble typical
# small-prompt API latency for a fast model — roughly 200ms median with a
# long right tail for occasional slow requests. This is a documented
# assumption, not a measurement of any real API.
_LATENCY_MU = 5.3  # ln(ms), median ~= exp(5.3) ~= 200ms
_LATENCY_SIGMA = 0.45


def _heuristic_token_count(text: str) -> int:
    """chars // 4, the same rough heuristic Kairos's budgeting.py uses.

    Not a real tokenizer. Good enough for relative comparisons across
    baselines in this controlled benchmark; explicitly not a claim of
    real token accuracy.
    """
    return max(1, len(text) // 4)


def _simulated_latency_ms(rng: random.Random) -> float:
    return rng.lognormvariate(_LATENCY_MU, _LATENCY_SIGMA)


class SimulatedLLM(LLMClient):
    """Deterministic, offline "model that is good at this task."

    For workload families this repo studies (arithmetic, business
    formulas), ``task_hint`` carries a canonical operation and bound
    parameters; the simulated backend evaluates it directly so
    correctness checking is meaningful. Outside those families, or when
    ``task_hint`` is absent, it echoes a template answer built from the
    prompt — sufficient for testing the *caching* plumbing, but not
    treated as a correctness ground truth for that path.
    """

    backend_name = "simulated"

    def __init__(self, model_name: str = DEFAULT_MODEL, seed: int | None = 0) -> None:
        self.model_name = model_name
        self._rng = random.Random(seed)

    def complete(self, prompt: str, *, task_hint: dict[str, Any] | None = None) -> LLMResponse:
        start = time.monotonic()
        text = self._answer(prompt, task_hint)
        latency_ms = _simulated_latency_ms(self._rng)

        input_tokens = _heuristic_token_count(prompt)
        output_tokens = _heuristic_token_count(text)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_name=self.model_name,
            backend=self.backend_name,
            raw={
                "elapsed_wall_s": time.monotonic() - start,
                "simulated": True,
            },
        )

    def _answer(self, prompt: str, task_hint: dict[str, Any] | None) -> str:
        if task_hint is None:
            return f"[simulated answer to: {prompt[:60]}...]"

        op = task_hint.get("operation")
        params = task_hint.get("params", {})

        # Single source of truth: the same TASK_REGISTRY executor the
        # computation cache uses to compute its own answers. Duplicating
        # the formulas here would risk the simulated "ground truth" and
        # the computation cache's actual computation silently drifting
        # apart — exactly the kind of bug this project is trying to
        # measure, not introduce.
        from ..computation.tasks import TASK_REGISTRY

        spec = TASK_REGISTRY.get(op) if op else None
        if spec is None:
            return f"[simulated answer, op={op}, params={params}]"

        try:
            result = spec.executor(**params)
        except (KeyError, ZeroDivisionError, TypeError) as exc:
            return f"[simulated answer error: {exc}]"

        return f"{result:.4f}" if isinstance(result, float) else str(result)
