"""Level 4 primary comparison: baselines A/C (carried forward) and F/G
(the branching, graduated-freshness pair) over a static (for this run —
see ttl_sweep.py for the mutating version) `return_decision` workload.
Writes results/level4/results.csv + summary.json.

Usage:
    python -m benchmarks.level4.run_experiment
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from aicache.cache.ttl_cache import TTLCache
from aicache.llm import make_backend
from aicache.matching.embedder import Embedder
from aicache.metrics.aggregate import summarize_all
from aicache.metrics.types import RequestResult

from benchmarks.level4.baselines import run_graph_computation_cache, run_no_cache, run_semantic_cache
from workloads.level4.generator import generate_level4_workload

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level4"

# Same calibration logic as Level 3 (docs/level3.md): one task family
# registered per baseline here, so identification only ever has to
# distinguish "is this a return_decision request at all" against
# whatever else might be in a shared embedding space — kept identical to
# Level 3's constant rather than re-derived, since nothing about adding
# a branch inside the task changes what identification has to do.
LEVEL4_IDENTIFICATION_THRESHOLD = 0.45


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="simulated", choices=["simulated", "gemini", "ollama"])
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--db-seed", type=int, default=42)
    ap.add_argument("--n-customers", type=int, default=80)
    ap.add_argument("--n-transactions", type=int, default=200)
    ap.add_argument("--combinations-per-branch", type=int, default=15)
    ap.add_argument("--paraphrases-per-combination", type=int, default=3)
    ap.add_argument("--semantic-threshold", type=float, default=0.65)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    print(f"[config] backend={args.backend} seed={args.seed} db_seed={args.db_seed}")
    db, workload = generate_level4_workload(
        db_seed=args.db_seed,
        n_customers=args.n_customers,
        n_transactions=args.n_transactions,
        combinations_per_branch=args.combinations_per_branch,
        paraphrases_per_combination=args.paraphrases_per_combination,
        seed=args.seed,
    )
    print(f"[workload] {len(workload)} requests, db: {len(db.customers)} customers / "
          f"{len(db.transactions)} transactions / {len(db.fraud_scores)} fraud scores")

    embedder = Embedder()
    all_results: list[RequestResult] = []

    def run(name: str, fn) -> None:
        t0 = time.perf_counter()
        llm = make_backend(args.backend)
        results = fn(workload, llm, db.clone())
        elapsed = time.perf_counter() - t0
        all_results.extend(results)
        n_hit = sum(1 for r in results if r.outcome.value.startswith("hit"))
        print(f"[{name}] {len(results)} requests, {n_hit} hits, {elapsed:.2f}s wall")

    run("no_cache", run_no_cache)
    run("semantic_cache", lambda wl, llm, db_: run_semantic_cache(wl, llm, db_, threshold=args.semantic_threshold, embedder=embedder))

    def f_fn(wl, llm, db_):
        results = run_graph_computation_cache(wl, llm, db_, embedder, LEVEL4_IDENTIFICATION_THRESHOLD, fraud_cache=None)
        for r in results:
            r.baseline = "graph_cache_fresh"
        return results

    def g_fn(wl, llm, db_):
        # ttl_seconds=None: memoize the fraud sub-fetch forever within
        # this run — the maximal-risk ablation endpoint, same role
        # Level 3's E played relative to D. ttl_sweep.py explores finite
        # TTLs against actual fraud-score volatility.
        results = run_graph_computation_cache(
            wl, llm, db_, embedder, LEVEL4_IDENTIFICATION_THRESHOLD, fraud_cache=TTLCache(ttl_seconds=None)
        )
        for r in results:
            r.baseline = "graph_cache_ttl_ablation"
        return results

    run("graph_cache_fresh", f_fn)
    run("graph_cache_ttl_ablation", g_fn)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(all_results, output_dir / "results.csv")
    summaries = summarize_all(all_results)
    _write_summary_json(summaries, output_dir / "summary.json")

    print("\n=== Summary ===")
    header = f"{'baseline':<28}{'n':>6}{'hit%':>8}{'valid%':>8}{'invalid%':>9}{'fallback%':>10}{'correct%':>9}{'tokens':>10}{'p50ms':>8}"
    print(header)
    for name, s in sorted(summaries.items()):
        print(
            f"{name:<28}{s.request_count:>6}{s.hit_rate*100:>7.1f}%{s.valid_hit_rate*100:>7.1f}%"
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
