from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Level3WorkloadItem:
    request_id: str
    text: str
    task_type: str
    entities: dict[str, str]
    numbers: dict[str, float]
    # Ground truth, computed once at generation time via the SAME
    # fetch+execute oracle path every baseline's own logic uses — but
    # called independently, here, not read back from any baseline under
    # test. Valid because the database is static for the duration of the
    # primary comparison run (see workloads/level3/generator.py); the
    # dedicated freshness experiment (benchmarks/level3/freshness_experiment.py)
    # recomputes ground truth at each step instead, since that run's
    # whole point is a database that changes mid-run.
    expected_result: dict = field(default_factory=dict)
    paraphrase_id: int = 0
