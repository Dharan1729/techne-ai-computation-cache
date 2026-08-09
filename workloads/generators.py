"""Level 1 (arithmetic) + Level 2 (business formula) workload generator.

For each task type: draw N independent parameter sets (the "live database"
in miniature — different numbers every time, exactly the RQ3 scenario),
then render each parameter set through several paraphrase templates. The
result is a workload where computational *structure* repeats constantly
(every request is one of nine task types) while *parameters* essentially
never repeat exactly — the shape the brief's central hypothesis is about.

Reproducible via a fixed seed (brief §28) — the same seed always produces
the same workload, so results are comparable run to run.
"""

from __future__ import annotations

import random

from aicache.computation.tasks import TASK_REGISTRY

from .templates import TEMPLATES
from .types import WorkloadItem

# (low, high) inclusive integer ranges for each task's raw (pre-transform,
# as-written-in-text) parameters, in the same order as TaskSpec.params.
PARAM_RANGES: dict[str, list[tuple[int, int]]] = {
    "add": [(1, 999), (1, 999)],
    "subtract": [(1, 999), (1, 999)],
    "multiply": [(1, 999), (1, 999)],
    "percentage": [(1, 99), (10, 10_000)],
    "compound_interest": [(100, 100_000), (1, 20), (1, 30)],
    "revenue": [(1, 1_000), (1, 10_000)],
    "discounted_price": [(1, 10_000), (1, 90)],
    "tax": [(1, 500_000), (1, 50)],
    "margin": [(1, 1_000_000), (1, 1_000_000)],
}


def generate_workload(
    task_types: list[str] | None = None,
    param_sets_per_task: int = 6,
    paraphrases_per_param_set: int = 3,
    seed: int = 42,
) -> list[WorkloadItem]:
    rng = random.Random(seed)
    task_types = task_types or list(TASK_REGISTRY.keys())

    items: list[WorkloadItem] = []
    request_counter = 0

    for task_type in task_types:
        spec = TASK_REGISTRY[task_type]
        ranges = PARAM_RANGES[task_type]
        templates = TEMPLATES[task_type]

        for param_set_id in range(param_sets_per_task):
            raw_numbers = [float(rng.randint(lo, hi)) for lo, hi in ranges]
            bound_params = {
                p.name: p.transform(v) for p, v in zip(spec.params, raw_numbers)
            }
            expected_answer = spec.executor(**bound_params)

            chosen_templates = rng.sample(
                templates, k=min(paraphrases_per_param_set, len(templates))
            )
            for paraphrase_id, template in enumerate(chosen_templates):
                text = template.format(*[_fmt(n) for n in raw_numbers])
                items.append(
                    WorkloadItem(
                        request_id=f"{task_type}-{request_counter}",
                        text=text,
                        task_type=task_type,
                        level=spec.level,
                        raw_numbers=raw_numbers,
                        bound_params=bound_params,
                        expected_answer=float(expected_answer),
                        paraphrase_id=paraphrase_id,
                        param_set_id=param_set_id,
                    )
                )
                request_counter += 1

    rng.shuffle(items)
    return items


def _fmt(n: float) -> str:
    # Integers render without a trailing ".0" so templates read naturally
    # ("What is 51 plus 93?" not "What is 51.0 plus 93.0?").
    return str(int(n)) if n == int(n) else f"{n:.2f}"
