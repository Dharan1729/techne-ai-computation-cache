"""EXP-020: the learning cache against the other caches, on the same two streams.

Every system sees the stream used in EXP-019: 150 requests (seed 1), then 150 new
requests with new numbers (seed 2). The main LLM is the always-correct stand-in,
so any wrong answer is the cache's own.

Baselines (from ``benchmarks/baselines.py``, the same ones used since EXP-004):
  no_cache          every request goes to the LLM
  exact_cache       reuse only a byte-identical request
  semantic_cache    reuse the nearest earlier request's ANSWER above a similarity
                    threshold (this repo's implementation, matched to RedisVL in EXP-006)
  hand-written registry (the previous design): operations written by a person
                    in advance. It is a reference, not a rival: it needs the
                    human effort that learning is meant to remove.

aicache rows are read from results/learning/ (EXP-019), so they are the real-model
runs, not a re-simulation.

    python -m benchmarks.learning_vs_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

from aicache.llm.simulated import SimulatedLLM
from aicache.matching.embedder import Embedder
from aicache.metrics.types import Outcome

from benchmarks.baselines import run_computation_cache, run_exact_cache, run_no_cache, run_semantic_cache
from workloads.generators import generate_workload

N = 150


def summarize(results) -> dict:
    hits = [r for r in results if r.outcome in (Outcome.HIT_VALID, Outcome.HIT_INVALID)]
    return {
        "answered_by_cache": len(hits),
        "wrong_answers": sum(r.outcome == Outcome.HIT_INVALID for r in hits),
        "llm_calls": sum(bool(r.llm_called) for r in results),
        "correct_overall": sum(bool(r.correct) for r in results),
    }


def both(results) -> dict:
    return {"learning_stream": summarize(results[:N]), "new_stream": summarize(results[N:])}


def scenario(name: str, new_stream, embedder, agree_files: dict[int, str] | None) -> dict:
    workload = generate_workload(seed=1) + new_stream
    llm = lambda: SimulatedLLM(seed=0)  # noqa: E731  a fresh always-correct model per system
    rows = {
        "no_cache": both(run_no_cache(workload, llm())),
        "exact_cache": both(run_exact_cache(workload, llm())),
    }
    for t in (0.75, 0.90, 0.97):
        rows[f"semantic_cache@{t}"] = both(run_semantic_cache(workload, llm(), threshold=t, embedder=embedder))
    rows["hand_written_registry"] = both(run_computation_cache(workload, llm()))
    for k, path in (agree_files or {}).items():
        d = json.loads(Path(path).read_text())
        L, T = d["learn_stream"], d["held_out_stream"]
        rows[f"aicache_learning@{k}_agreements"] = {
            "learning_stream": {"answered_by_cache": L["computed"], "wrong_answers": L["wrong_executions"],
                                "llm_calls": L["llm_calls"] + d["normalizer_calls"]},
            "new_stream": {"answered_by_cache": T["computed"], "wrong_answers": T["wrong_executions"],
                           "llm_calls": T["llm_calls"], "requests": T["n"]},
            "note": f"learning stream llm_calls = {L['llm_calls']} main-model + {d['normalizer_calls']} small-model",
        }
    print(f"\n== {name}")
    print(f"{'system':36s} {'--- learning stream ---':>26s}   {'--- new requests ---':>26s}")
    print(f"{'':36s} {'cache':>7s} {'wrong':>6s} {'LLM':>6s}   {'cache':>7s} {'wrong':>6s} {'LLM':>6s}")
    for n, r in rows.items():
        a, b = r["learning_stream"], r["new_stream"]
        print(f"{n:36s} {a['answered_by_cache']:7d} {a['wrong_answers']:6d} {a['llm_calls']:6d}   "
              f"{b['answered_by_cache']:7d} {b['wrong_answers']:6d} {b['llm_calls']:6d}")
    return rows


def main(out: str = "results/learning/baselines.json") -> None:
    embedder = Embedder()
    result = {"numbers_repeat": scenario(
        "new requests REPEAT earlier numbers about two thirds of the time (the EXP-019 stream)",
        generate_workload(seed=2), embedder,
        {k: f"results/learning/llama3_teacher-correct_agree{k}.json" for k in (5, 3, 1)})}
    norepeat = {k: f"results/learning/llama3_norepeat_agree{k}.json" for k in (5, 3)
                if Path(f"results/learning/llama3_norepeat_agree{k}.json").exists()}
    result["numbers_never_repeat"] = scenario(
        "NO new request repeats earlier numbers",
        generate_workload(seed=2, param_sets_per_task=18, paraphrases_per_param_set=1), embedder, norepeat)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
