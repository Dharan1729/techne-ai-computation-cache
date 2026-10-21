"""aicache: learns the operation behind calculation-shaped LLM requests from
your model's own answers, verifies it, and answers them itself.

Start with ``aicache.engine.ActionCache`` (learning), ``ComputationRouter`` (tools
you write yourself), the ``aicache`` command line, or ``aicache serve`` for an
OpenAI-compatible gateway. The package also contains the research benchmarks
behind it; see docs/experiment-log.md.
"""

__version__ = "0.3.0"

from .router import ComputationRouter, Result, Slot, TextSlot, bind_slots  # noqa: E402
from .telemetry import Telemetry  # noqa: E402

__all__ = ["ComputationRouter", "Result", "Slot", "TextSlot", "Telemetry", "bind_slots", "__version__"]


def __getattr__(name: str):  # ActionCache imports config (pydantic, yaml); load it only when asked for
    if name == "ActionCache":
        from .engine import ActionCache

        return ActionCache
    raise AttributeError(name)
