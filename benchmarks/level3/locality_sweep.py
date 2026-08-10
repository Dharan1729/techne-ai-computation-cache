"""Level 3b, sweep 1: structural locality.

The primary Level 3 comparison (EXP-009) deliberately built a
high-repetition workload — 5 task families x a fixed set of entity
combinations x 3 paraphrases each — and reported a 99.1% computation-cache
hit rate on it. That number is a property of *this workload's* structural
locality, not a general constant (`docs/level3.md` RQ9 flags this
explicitly). This sweep measures the thing RQ9 could previously only
assert: what happens to each baseline's hit rate as structural locality
drops.

**What "structural locality" means operationally here**: this system's
task registry is intentionally small (5 hand-specified enterprise
workflows — see `docs/level3.md`'s scope notes) rather than hundreds of
synthetic task types, so the lever that actually varies request-to-request
similarity is the number of DISTINCT ENTITY/PARAMETER COMBINATIONS drawn
per task family, not the number of task types. This sweep holds the 5
task families and 3-paraphrases-per-combination fixed and varies
``combinations_per_task`` — i.e. how many distinct (customer, transaction,
...) tuples the workload draws before it starts repeating them. Low
values mean many paraphrases of a SMALL set of combinations (high
locality, lots of near-duplicate text); high values mean a LARGE set of
combinations, most seen only once or twice (low locality, few
near-duplicates for a response cache to match against).

This deliberately does NOT vary embedding threshold or task-identification
logic — only workload structure — so any effect measured is attributable
to locality itself, not a confound.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.metrics.aggregate import summarize_all
from aicache.metrics.types import RequestResult

from benchmarks.level3.baselines import (
    run_exact_cache,
    run_no_cache,
    run_semantic_cache,
    run_workflow_computation_cache,
)
from benchmarks.level3.run_experiment import LEVEL3_IDENTIFICATION_THRESHOLD
from workloads.level3.generator import generate_level3_workload

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level3"

# Fixed at the maximum available (see workloads/level3/templates.py — each
# task has exactly 3 phrasings), so every sweep point gets the SAME
# per-combination repetition; only combinations_per_task varies.
PARAPHRASES_PER_COMBINATION = 3

# Distinct entity combinations drawn per task family. n_transactions
# scales with the top of this range so large sweep points aren't starved
# of distinct transactions to draw from (see generator.py's
# _pick_customer_and_transaction, which samples with replacement).
COMBINATIONS_SWEEP = [2, 5, 10, 20, 40, 80]


def _run_point(combinations_per_task: int, seed: int, db_seed: int) -> list[RequestResult]:
    n_transactions = max(200, combinations_per_task * 6)
    n_customers = max(80, combinations_per_task * 2)
    db, workload = generate_level3_workload(
        db_seed=db_seed,
        n_customers=n_customers,
        n_transactions=n_transactions,
        combinations_per_task=combinations_per_task,
        paraphrases_per_combination=PARAPHRASES_PER_COMBINATION,
        seed=seed,
    )

    embedder = Embedder()
    identifier = TaskIdentifier(
        embedder=embedder, threshold=LEVEL3_IDENTIFICATION_THRESHOLD, registry=WORKFLOW_TASK_REGISTRY
    )

    results: list[RequestResult] = []
    results += run_no_cache(workload, _fresh_llm(), db.clone())
    results += run_exact_cache(workload, _fresh_llm(), db.clone())
    results += run_semantic_cache(workload, _fresh_llm(), db.clone(), threshold=0.65, embedder=embedder)

    d_results = run_workflow_computation_cache(workload, _fresh_llm(), db.clone(), identifier=identifier, use_stale_cache=False)
    for r in d_results:
        r.baseline = "computation_cache_fresh"
    results += d_results

    e_results = run_workflow_computation_cache(workload, _fresh_llm(), db.clone(), identifier=identifier, use_stale_cache=True)
    for r in e_results:
        r.baseline = "computation_cache_stale_ablation"
    results += e_results

    for r in results:
        r.extra = {**r.extra, "combinations_per_task": combinations_per_task, "n_requests": len(workload)}
    return results


def _fresh_llm():
    from aicache.llm.simulated import SimulatedLLM

    return SimulatedLLM(seed=0)


def main() -> None:
    all_results: list[RequestResult] = []
    rows_out: list[dict] = []

    for combos in COMBINATIONS_SWEEP:
        t0 = time.perf_counter()
        results = _run_point(combos, seed=99, db_seed=42)
        elapsed = time.perf_counter() - t0
        n_requests = results[0].extra["n_requests"] if results else 0
        print(f"[locality] combinations_per_task={combos:>4} -> {n_requests} requests, {elapsed:.1f}s wall")

        summaries = summarize_all(results)
        for name, s in summaries.items():
            rows_out.append(
                {
                    "combinations_per_task": combos,
                    "n_requests": n_requests,
                    "baseline": name,
                    **{k: v for k, v in asdict(s).items() if k != "outcome_counts"},
                }
            )
        all_results.extend(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "locality_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    json_path = RESULTS_DIR / "locality_sweep.json"
    with open(json_path, "w") as f:
        json.dump(rows_out, f, indent=2, default=str)

    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
