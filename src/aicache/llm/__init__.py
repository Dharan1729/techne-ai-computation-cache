from .base import LLMClient, LLMResponse
from .pricing import DEFAULT_MODEL, PRICING, estimate_cost_usd
from .simulated import SimulatedLLM

__all__ = [
    "LLMClient",
    "LLMResponse",
    "SimulatedLLM",
    "PRICING",
    "DEFAULT_MODEL",
    "estimate_cost_usd",
]


def make_backend(name: str = "simulated", **kwargs):
    """Factory so the harness can select a backend by string name.

    Kept separate from a top-level GeminiLLM import so importing this
    package never requires `httpx` to be configured with a live key —
    Gemini is imported lazily, only if actually requested.
    """
    if name == "simulated":
        return SimulatedLLM(**kwargs)
    if name == "gemini":
        from .gemini import GeminiLLM

        return GeminiLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {name!r}")
