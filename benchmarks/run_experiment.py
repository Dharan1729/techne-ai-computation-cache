"""Run the immediate experiment (brief §5): baselines A-D over the Level 1/2
workload, plus a semantic-cache threshold sweep (brief §13), and write
results to results/results.csv + results/summary.json (or --output-dir).

Usage:
    python -m benchmarks.run_experiment
    python -m benchmarks.run_experiment --backend simulated --seed 42
    python -m benchmarks.run_experiment --backend gemini   # needs GEMINI_API_KEY
    python -m benchmarks.run_experiment --backend ollama   # needs `ollama serve` running locally
    python -m benchmarks.run_experiment --backend ollama --output-dir results/real_backend_ollama \\
        --skip-threshold-sweep --skip-external   # don't clobber the headline simulated-backend results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from aicache.llm import make_backend
from aicache.matching.embedder import Embedder
from aicache.metrics.aggregate import summarize_all
from aicache.metrics.types import RequestResult

from benchmarks.baselines import (
    run_computation_cache,
    run_exact_cache,
    run_no_cache,
    run_semantic_cache,
)
from benchmarks.external import gptcache_adapter, redisvl_adapter
from workloads.generators import generate_workload

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Calibrated against redis/langcache-embed-v1 (see docs/experiment-log.md):
# this model separates genuine-paraphrase similarity from
# different-parameter similarity much more sharply than a generic
# sentence-similarity model, so the "dangerous" region of this sweep
# sits lower (0.55-0.65) than it did under this project's original
# MiniLM-based embedder (0.80-0.85) — the sweep range moved with it,
# not arbitrarily widened.
SEMANTIC_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="simulated", choices=["simulated", "gemini", "ollama"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--param-sets-per-task", type=int, default=6)
    ap.add_argument("--paraphrases-per-param-set", type=int, default=3)
    ap.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="Threshold used for the headline semantic_cache condition (the sweep runs separately at fixed thresholds above)",
    )
    ap.add_argument("--skip-threshold-sweep", action="store_true")
    ap.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip the GPTCache/RedisVL comparison baselines even if available",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to write results.csv/summary.json. Defaults to results/ — "
            "pass a different directory (e.g. results/real_backend_ollama) for "
            "any run that shouldn't overwrite the headline simulated-backend "
            "results (see docs/experiment-log.md EXP-008 for why this flag "
            "exists: an earlier real-backend run clobbered EXP-006's results "
            "before this existed, and had to be recovered from git history)."
        ),
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    print(f"[config] backend={args.backend} seed={args.seed}")
    workload = generate_workload(
        param_sets_per_task=args.param_sets_per_task,
        paraphrases_per_param_set=args.paraphrases_per_param_set,
        seed=args.seed,
    )
    print(f"[workload] {len(workload)} requests across {len({w.task_type for w in workload})} task types")

    # Share one embedder across every baseline/condition that needs one —
    # it's an offline model load, not a per-baseline cost, and reusing it
    # keeps the run fast.
    embedder = Embedder()

    all_results: list[RequestResult] = []

    def run(name: str, fn, optional: bool = False) -> None:
        t0 = time.perf_counter()
        llm = make_backend(args.backend)
        try:
            results = fn(workload, llm)
        except (gptcache_adapter.GPTCacheUnavailable, redisvl_adapter.RedisVLUnavailable) as exc:
            if optional:
                print(f"[{name}] SKIPPED — {exc}")
                return
            raise
        elapsed = time.perf_counter() - t0
        all_results.extend(results)
        n_hit = sum(1 for r in results if r.outcome.value.startswith("hit"))
        print(f"[{name}] {len(results)} requests, {n_hit} hits, {elapsed:.2f}s wall")

    run("no_cache", run_no_cache)
    run("exact_cache", run_exact_cache)
    run("semantic_cache", lambda wl, llm: run_semantic_cache(wl, llm, threshold=args.semantic_threshold, embedder=embedder))
    run("computation_cache", run_computation_cache)

    if not args.skip_external:
        gc_ok, gc_reason = gptcache_adapter.is_available()
        rv_ok, rv_reason = redisvl_adapter.is_available()
        print(f"[external] gptcache available={gc_ok} ({gc_reason or 'ok'}); redisvl available={rv_ok} ({rv_reason or 'ok'})")

        run(
            "gptcache",
            lambda wl, llm: gptcache_adapter.run_gptcache(wl, llm, threshold=args.semantic_threshold),
            optional=True,
        )
        run(
            "redisvl",
            lambda wl, llm: redisvl_adapter.run_redisvl(wl, llm, similarity_threshold=args.semantic_threshold),
            optional=True,
        )

    if not args.skip_threshold_sweep:
        for thresh in SEMANTIC_THRESHOLDS:
            label = f"semantic_cache@{thresh:.2f}"

            def fn(wl, llm, thresh=thresh):
                results = run_semantic_cache(wl, llm, threshold=thresh, embedder=embedder)
                for r in results:
                    r.baseline = label
                return results

            run(label, fn)

        if not args.skip_external:
            for thresh in SEMANTIC_THRESHOLDS:
                gc_label = f"gptcache@{thresh:.2f}"
                rv_label = f"redisvl@{thresh:.2f}"

                def gc_fn(wl, llm, thresh=thresh, label=gc_label):
                    results = gptcache_adapter.run_gptcache(wl, llm, threshold=thresh)
                    for r in results:
                        r.baseline = label
                    return results

                def rv_fn(wl, llm, thresh=thresh, label=rv_label):
                    results = redisvl_adapter.run_redisvl(wl, llm, similarity_threshold=thresh)
                    for r in results:
                        r.baseline = label
                    return results

                run(gc_label, gc_fn, optional=True)
                run(rv_label, rv_fn, optional=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(all_results, output_dir / "results.csv")
    summaries = summarize_all(all_results)
    _write_summary_json(summaries, output_dir / "summary.json")

    print("\n=== Summary ===")
    header = f"{'baseline':<24}{'n':>6}{'hit%':>8}{'valid%':>8}{'invalid%':>9}{'fallback%':>10}{'correct%':>9}{'tokens':>10}{'cost$':>10}{'p50ms':>8}"
    print(header)
    for name, s in sorted(summaries.items()):
        print(
            f"{name:<24}{s.request_count:>6}{s.hit_rate*100:>7.1f}%{s.valid_hit_rate*100:>7.1f}%"
            f"{s.invalid_hit_rate*100:>8.1f}%{s.fallback_rate*100:>9.1f}%{s.correctness_rate*100:>8.1f}%"
            f"{s.total_tokens:>10}{s.total_cost_usd:>10.5f}{s.p50_latency_ms:>8.1f}"
        )
    print(f"\nWrote {output_dir / 'results.csv'} and {output_dir / 'summary.json'}")


def _write_csv(results: list[RequestResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "baseline",
                "outcome",
                "correct",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "cost_usd",
                "llm_called",
                "task_type",
                "similarity",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.request_id,
                    r.baseline,
                    r.outcome.value,
                    r.correct,
                    r.latency_ms,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_usd,
                    r.llm_called,
                    r.task_type,
                    r.similarity,
                ]
            )


def _write_summary_json(summaries: dict, path: Path) -> None:
    serializable = {name: asdict(s) for name, s in summaries.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()
