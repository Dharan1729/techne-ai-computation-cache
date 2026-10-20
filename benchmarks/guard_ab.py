"""Reproduce EXP-016's guard comparison: wrong cache hits with and without the
lexical-cue guard, over seeds 1-10. Offline, no model download beyond the
embedding model, under a minute.

    python -m benchmarks.guard_ab

Expected (from docs/experiment-log.md, EXP-016): without the guard, 20 wrong
hits in 1500 requests (per seed 3,3,4,0,2,1,0,3,0,4); with it, 0 in 1500.
Seeds 1-5 are the ones where the failures were first seen; seeds 6-10 were
not looked at while the guard was written.
"""

from __future__ import annotations

import os

from aicache.llm.simulated import SimulatedLLM
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.types import Outcome

from benchmarks.baselines import run_computation_cache
from workloads.generators import generate_workload


def run(seed: int, guard: bool, identifier: TaskIdentifier) -> tuple[int, int, int]:
    if guard:
        os.environ.pop("AICACHE_NO_CUES", None)
    else:
        os.environ["AICACHE_NO_CUES"] = "1"
    results = run_computation_cache(generate_workload(seed=seed), SimulatedLLM(seed=0), identifier)
    invalid = sum(r.outcome == Outcome.HIT_INVALID for r in results)
    hits = sum(r.outcome in (Outcome.HIT_VALID, Outcome.HIT_INVALID) for r in results)
    return invalid, hits, len(results)


def main() -> None:
    identifier = TaskIdentifier()  # one embedder load, shared by every run
    totals = {False: [0, 0, 0], True: [0, 0, 0]}
    per_seed: dict[bool, list[int]] = {False: [], True: []}
    for seed in range(1, 11):
        for guard in (False, True):
            inv, hits, n = run(seed, guard, identifier)
            per_seed[guard].append(inv)
            for i, v in enumerate((inv, hits, n)):
                totals[guard][i] += v
    for guard, label in ((False, "without guard"), (True, "with guard   ")):
        inv, hits, n = totals[guard]
        print(f"{label}: {inv:2d} wrong hits in {n} requests "
              f"({inv / n:.2%}); hit rate {hits / n:.1%}; per seed {per_seed[guard]}")


if __name__ == "__main__":
    main()
