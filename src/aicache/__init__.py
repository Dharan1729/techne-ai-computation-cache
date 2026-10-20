"""ai-computation-cache: an investigation into caching computation structure
(vs. final responses) for AI workloads whose parameters change per request.

See docs/problem.md for the research question and docs/architecture.md for
how the pieces here fit together.
"""

__version__ = "0.1.0"

# Public integration API (experimental). See README, "Use it in your own workflow".
from .router import ComputationRouter, Result, Slot, TextSlot, bind_slots  # noqa: E402

__all__ = ["ComputationRouter", "Result", "Slot", "TextSlot", "bind_slots"]
