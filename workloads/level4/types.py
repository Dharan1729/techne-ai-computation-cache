from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Level4WorkloadItem:
    request_id: str
    text: str
    task_type: str
    branch: str  # which of return_decision's 4 branches this item is constructed to hit
    entities: dict[str, str]
    numbers: dict[str, float]
    expected_result: dict = field(default_factory=dict)
    paraphrase_id: int = 0
