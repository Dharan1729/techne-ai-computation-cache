"""Real LLM backend: Google Gemini, via plain REST (no SDK dependency).

Inactive unless ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set — the
harness has to be told explicitly to use this backend (``--backend gemini``
on ``benchmarks/run_experiment.py``). Chosen over Anthropic for the first
real-backend pass because Gemini's free tier makes small-scale validation
runs of this benchmark possible without a paid key; nothing about the
cache/harness code is provider-specific, so swapping in an Anthropic
backend later is a new ~60-line file, not a redesign.

Uses raw HTTP (httpx) rather than the `google-generativeai` /
`google-genai` SDK to avoid adding a dependency for what is, here, a
single endpoint. Real Anthropic API integration (were this an
Anthropic-model project) would go through the official `anthropic` SDK
per this environment's house convention — REST is the exception here
specifically because no first-party Python SDK dependency was already
present and the call surface is trivial.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import LLMClient, LLMResponse

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiNotConfigured(RuntimeError):
    """Raised when GeminiLLM is used without an API key configured."""


class GeminiLLM(LLMClient):
    backend_name = "gemini"

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, timeout_s: float = 30.0) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise GeminiNotConfigured(
                "GeminiLLM requires GEMINI_API_KEY or GOOGLE_API_KEY to be set. "
                "This backend makes real API calls and may incur cost/quota "
                "usage against your Google account — export the key only "
                "when you intend to run the real-backend experiment."
            )
        self._api_key = api_key
        self.model_name = model_name
        self._client = httpx.Client(timeout=timeout_s)

    def complete(self, prompt: str, *, task_hint: dict[str, Any] | None = None) -> LLMResponse:
        # task_hint is intentionally ignored — a real backend only ever
        # sees what a real API call would see: the prompt text.
        url = f"{_API_BASE}/models/{self.model_name}:generateContent"
        start = time.monotonic()
        resp = self._client.post(
            url,
            params={"key": self._api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

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
