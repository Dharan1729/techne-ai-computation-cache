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

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        timeout_s: float = 30.0,
        min_interval_s: float = 4.0,
    ) -> None:
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
        # Firing requests back-to-back exhausts a free-tier key's
        # requests-per-minute quota in seconds — the retry-with-backoff
        # in _request_with_backoff is a safety net, not a substitute for
        # not hammering the API in the first place. 4s => max 15 RPM;
        # tune down if a higher-tier key's limits are known to allow it.
        self._min_interval_s = min_interval_s
        self._last_request_at: float | None = None

    def complete(self, prompt: str, *, task_hint: dict[str, Any] | None = None) -> LLMResponse:
        # task_hint is intentionally ignored — a real backend only ever
        # sees what a real API call would see: the prompt text.
        url = f"{_API_BASE}/models/{self.model_name}:generateContent"
        start = time.monotonic()
        resp = self._request_with_backoff(url, prompt)
        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        # Gemini 2.5 Flash bills "thinking" tokens as output even though
        # they're not part of the visible response — folded in here so
        # cost accounting reflects what a real bill would show, not just
        # the visible answer text. (Disabled via thinkingBudget=0 below
        # for this workload anyway; kept defensive in case a future
        # model/config re-enables it.)
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            model_name=self.model_name,
            backend=self.backend_name,
            raw=data,
        )

    def _request_with_backoff(
        self, url: str, prompt: str, max_retries: int = 5
    ) -> httpx.Response:
        """Free-tier keys hit 429s under any real burst of traffic —
        expected and normal, not a sign anything's wrong. Retries 429
        (rate limit) and 503 (transient overload) with exponential
        backoff; anything else (401, 400, etc.) is a real error and
        raises immediately rather than being retried into a longer
        failure."""
        delay_s = 5.0
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(max_retries):
            self._throttle()
            # This workload is closed-form arithmetic/formula questions —
            # no reasoning benefit from extended thinking, and it
            # triples-plus real latency/token cost for no accuracy gain
            # here (0 disables it on 2.5 Flash).
            resp = self._client.post(
                url,
                params={"key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
                },
            )
            if resp.status_code not in (429, 503):
                resp.raise_for_status()
                return resp
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code} from Gemini API", request=resp.request, response=resp
            )
            if attempt < max_retries - 1:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2, 30.0)
        assert last_exc is not None
        raise last_exc

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def close(self) -> None:
        self._client.close()
