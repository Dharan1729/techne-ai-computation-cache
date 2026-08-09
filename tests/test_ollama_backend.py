"""Tests for the optional real Ollama LLM backend.

Skipped (not failed) when no local Ollama daemon is reachable — this is
a real external system this repo doesn't own, unlike the simulated
backend. See aicache/llm/ollama.py's docstring for why it exists
alongside the Gemini backend (no rate limit, so it's what EXP-008 in
docs/experiment-log.md actually ran at full scale)."""

from __future__ import annotations

import pytest

from aicache.llm.ollama import OllamaLLM, OllamaNotConfigured
from aicache.metrics.correctness import is_correct

try:
    OllamaLLM()
    _OLLAMA_OK = True
    _OLLAMA_REASON = ""
except OllamaNotConfigured as exc:
    _OLLAMA_OK = False
    _OLLAMA_REASON = str(exc)


@pytest.mark.skipif(not _OLLAMA_OK, reason=_OLLAMA_REASON)
def test_ollama_answers_and_reports_real_token_counts():
    llm = OllamaLLM()
    resp = llm.complete("What is 10 plus 20?")
    assert resp.backend == "ollama"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.text.strip() != ""


@pytest.mark.skipif(not _OLLAMA_OK, reason=_OLLAMA_REASON)
def test_ollama_final_number_parsing_handles_worked_solutions():
    """The real reason parse_final_number exists (see
    aicache/metrics/correctness.py and docs/experiment-log.md EXP-007):
    a real model shows its work, so the answer has to be found, not
    assumed to be the first number in the text."""
    llm = OllamaLLM()
    resp = llm.complete("What is 512 plus 937?")
    assert is_correct(resp.text, 1449.0), f"expected 1449 in {resp.text!r}"
