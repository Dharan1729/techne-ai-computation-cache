"""Shared workload item type used by every workload-level generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkloadItem:
    request_id: str
    text: str
    task_type: str
    level: int
    raw_numbers: list[float]  # the numbers as they appear, left-to-right, in `text`
    bound_params: dict[str, float]  # same numbers after each param's unit transform
    expected_answer: float
    paraphrase_id: int  # which paraphrase template produced `text`
    param_set_id: int  # which parameter draw this is, within its task_type
    metadata: dict = field(default_factory=dict)
