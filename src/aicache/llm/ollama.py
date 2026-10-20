"""Real LLM backend: Ollama, running fully locally.

Inactive unless a local Ollama daemon is reachable at
``http://localhost:11434`` (or ``OLLAMA_HOST`` if set) — the harness
still has to be told to use this backend explicitly (``--backend
ollama``). Added alongside the Gemini backend (``gemini.py``) after
hitting a hard 20-requests/day free-tier quota wall there (see
docs/experiment-log.md, EXP-007): a local model has no such ceiling,
which matters for this project specifically because the whole point of
a real-backend validation run is to exercise the full workload, not a
quota-limited sliver of it.

No API key, no rate limiting needed (nothing external is being called),
no cost — but also no published per-token pricing, since nothing is
billed. ``estimate_cost_usd`` (aicache/llm/pricing.py) doesn't have an
entry for Ollama's models and isn't meant to; cost accounting for this
backend's results should read as "$0, real local compute time traded for
it" — the latency/token numbers are real, the dollar figure is not
meaningful here and callers should not read $0.00 as "free at the model
layer" the way it would for, say, a cached hit.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import LLMClient, LLMResponse

DEFAULT_OLLAMA_MODEL = "llama3"
_API_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaNotConfigured(RuntimeError):
    """Raised when OllamaLLM is used but no local daemon is reachable."""


class OllamaLLM(LLMClient):
    backend_name = "ollama"

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        timeout_s: float = 120.0,
        options: dict[str, Any] | None = None,
        response_format: str | None = None,
    ) -> None:
        self.model_name = model_name
        # Sampling options passed straight to Ollama (e.g.
        # {"temperature": 0, "seed": 0}) so a benchmark run can pin
        # determinism instead of sampling at the model's default.
        self.options = options
        self.response_format = response_format
        self._client = httpx.Client(base_url=_API_BASE, timeout=timeout_s)
        try:
            resp = self._client.get("/api/version")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaNotConfigured(
                f"Ollama not reachable at {_API_BASE} — is `ollama serve` running? "
                f"({exc})"
            ) from exc

    def complete(self, prompt: str, *, task_hint: dict[str, Any] | None = None) -> LLMResponse:
        # task_hint is intentionally ignored — see gemini.py's docstring
        # for why: a real backend only ever sees what a real call would.
        start = time.monotonic()
        payload: dict[str, Any] = {"model": self.model_name, "prompt": prompt, "stream": False}
        if self.options:
            payload["options"] = self.options
        if self.response_format:
            payload["format"] = self.response_format
        resp = self._client.post("/api/generate", json=payload)
        elapsed_ms = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response", "")
        # Ollama reports real token counts for the actual local
        # inference — prompt_eval_count / eval_count are the input /
        # output token counts respectively (not a heuristic, unlike the
        # simulated backend's chars//4).
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            model_name=self.model_name,
            backend=self.backend_name,
            raw=data,
        )

    def close(self) -> None:
        self._client.close()
