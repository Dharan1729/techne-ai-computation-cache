"""Published Claude API pricing, used only for the SIMULATED backend's cost math.

Snapshot date: 2026-06-24 (per Anthropic's Claude API documentation, as
surfaced by the claude-api reference at the time this repo was built,
2026-08-10). This is real, publicly published pricing — not fabricated —
but it is a point-in-time snapshot a reader should re-verify before citing
in anything long-lived. It exists so the simulated backend's "inference
cost avoided" numbers are grounded in a real rate card instead of an
invented one, while being clear that no actual API calls produced them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    model_name: str
    input_per_mtok_usd: float
    output_per_mtok_usd: float


# USD per million tokens, snapshot 2026-06-24.
PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing("claude-haiku-4-5", 1.00, 5.00),
    "claude-sonnet-5": ModelPricing("claude-sonnet-5", 3.00, 15.00),
    "claude-opus-5": ModelPricing("claude-opus-5", 5.00, 25.00),
    "claude-fable-5": ModelPricing("claude-fable-5", 10.00, 50.00),
}

DEFAULT_MODEL = "claude-haiku-4-5"


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model_name, PRICING[DEFAULT_MODEL])
    return (input_tokens / 1_000_000) * p.input_per_mtok_usd + (
        output_tokens / 1_000_000
    ) * p.output_per_mtok_usd
