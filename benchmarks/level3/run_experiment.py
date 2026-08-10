"""Level 3 primary comparison: baselines A-E over a static (for this
run — see the freshness_experiment.py for the mutating version)
enterprise workload. Writes results/level3/results.csv + summary.json.

Usage:
    python -m benchmarks.level3.run_experiment
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from aicache.llm import make_backend
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
from workloads.level3.generator import generate_level3_workload

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level3"

# Calibrated on this workload (see docs/level3.md): the 5 enterprise
# task families embed far more distinctly than Level 1/2's 9 arithmetic
# ops did — zero misidentified matches across the ENTIRE similarity
# range in calibration, not just above some threshold. 0.45 is chosen
# for full recall at that same zero-risk level, not because a tighter
# threshold was needed the way Level 1/2's 0.52 was.
LEVEL3_IDENTIFICATION_THRESHOLD = 0.45


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="simulated", choices=["simulated", "gemini", "ollama"])
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--db-seed", type=int, default=42)
    ap.add_argument("--n-customers", type=int, default=80)
    ap.add_argument("--n-transactions", type=int, default=200)
    ap.add_argument("--combinations-per-task", type=int, default=15)
    ap.add_argument("--paraphrases-per-combination", type=int, default=3)
    ap.add_argument("--semantic-threshold", type=float, default=0.65)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    print(f"[config] backend={args.backend} seed={args.seed} db_seed={args.db_seed}")
    db, workload = generate_level3_workload(
        db_seed=args.db_seed,
        n_customers=args.n_customers,
        n_transactions=args.n_transactions,
        combinations_per_task=args.combinations_per_task,
        paraphrases_per_combination=args.paraphrases_per_combination,
        seed=args.seed,
    )
    n_task_types = len({w.task_type for w in workload})
    print(f"[workload] {len(workload)} requests across {n_task_types} task types, "
          f"db: {len(db.customers)} customers / {len(db.transactions)} transactions")

    embedder = Embedder()
    identifier = TaskIdentifier(
        embedder=embedder, threshold=LEVEL3_IDENTIFICATION_THRESHOLD, registry=WORKFLOW_TASK_REGISTRY
    )

    all_results: list[RequestResult] = []

    def run(name: str, fn) -> None:
        t0 = time.perf_counter()
        llm = make_backend(args.backend)
        # Each baseline replays against its own fresh DB clone — the
        # workload's ground truth was computed against the ORIGINAL
        # snapshot, so every baseline must see that same snapshot, not
        # whatever an earlier baseline may have mutated (none of A-E
        # mutate the DB in this run, but cloning keeps that invariant
        # true by construction rather than by accident).
        results = fn(workload, llm, db.clone())
        elapsed = time.perf_counter() - t0
        all_results.extend(results)
        n_hit = sum(1 for r in results if r.outcome.value.startswith("hit"))
        print(f"[{name}] {len(results)} requests, {n_hit} hits, {elapsed:.2f}s wall")

    run("no_cache", run_no_cache)
    run("exact_cache", run_exact_cache)
    run("semantic_cache", lambda wl, llm, db_: run_semantic_cache(wl, llm, db_, threshold=args.semantic_threshold, embedder=embedder))

    def d_fn(wl, llm, db_):
        results = run_workflow_computation_cache(wl, llm, db_, identifier=identifier, use_stale_cache=False)
        for r in results:
            r.baseline = "computation_cache_fresh"
        return results

    def e_fn(wl, llm, db_):
        results = run_workflow_computation_cache(wl, llm, db_, identifier=identifier, use_stale_cache=True)
        for r in results:
            r.baseline = "computation_cache_stale_ablation"
        return results

    run("computation_cache_fresh", d_fn)
    run("computation_cache_stale_ablation", e_fn)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(all_results, output_dir / "results.csv")
    summaries = summarize_all(all_results)
    _write_summary_json(summaries, output_dir / "summary.json")

    print("\n=== Summary ===")
    header = f"{'baseline':<32}{'n':>6}{'hit%':>8}{'valid%':>8}{'invalid%':>9}{'fallback%':>10}{'correct%':>9}{'tokens':>10}{'p50ms':>8}"
    print(header)
    for name, s in sorted(summaries.items()):
        print(
            f"{name:<32}{s.request_count:>6}{s.hit_rate*100:>7.1f}%{s.valid_hit_rate*100:>7.1f}%"
            f"{s.invalid_hit_rate*100:>8.1f}%{s.fallback_rate*100:>9.1f}%{s.correctness_rate*100:>8.1f}%"
            f"{s.total_tokens:>10}{s.p50_latency_ms:>8.1f}"
        )
    print(f"\nWrote {output_dir / 'results.csv'} and {output_dir / 'summary.json'}")


def _write_csv(results: list[RequestResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id", "baseline", "outcome", "correct", "latency_ms",
                "input_tokens", "output_tokens", "cost_usd", "llm_called",
                "task_type", "similarity", "extra",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.request_id, r.baseline, r.outcome.value, r.correct, r.latency_ms,
                    r.input_tokens, r.output_tokens, r.cost_usd, r.llm_called,
                    r.task_type, r.similarity, json.dumps(r.extra),
                ]
            )


def _write_summary_json(summaries: dict, path: Path) -> None:
    serializable = {name: asdict(s) for name, s in summaries.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()
