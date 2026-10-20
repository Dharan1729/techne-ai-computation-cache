"""aicache: answer calculation-shaped LLM requests with your own code, and send
everything else to the model.

Start with ``ComputationRouter`` (or ``aicache.config.build_router`` for a YAML
config), the ``aicache`` command line, or ``aicache serve`` for an
OpenAI-compatible gateway. The package also contains the research benchmarks
behind it; see docs/experiment-log.md.
"""

__version__ = "0.2.0"

from .router import ComputationRouter, Result, Slot, TextSlot, bind_slots  # noqa: E402
from .telemetry import Telemetry  # noqa: E402

__all__ = ["ComputationRouter", "Result", "Slot", "TextSlot", "Telemetry", "bind_slots", "__version__"]
