"""Does the learning loop work with a real small LLM? (EXP-019)

Phase 1 (learn): stream a workload through ``ActionCache``. The main LLM answers
every request the cache cannot; a real small model (Ollama) proposes actions
from those (request, answer) pairs; actions go through verification, probation
and promotion.

Phase 2 (held-out): freeze the store, stream a different workload (new seed, new
parameter values and phrasings) and measure what the learned actions answer, and
whether any answer was wrong. Wrong executions are the number that matters.

    python -m benchmarks.learning_eval --normalizer llama3 --main simulated \
        --output results/learning/llama3_simulated.json

``--main simulated`` is a main LLM that is always right, which isolates the
learning mechanism. ``--main ollama:llama3`` uses a real, fallible model, which
is the realistic case: the teacher can be wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from aicache.actions import ActionStore
from aicache.embedders import SentenceTransformerEmbedder
from aicache.engine import ActionCache
from aicache.learner import LearnPolicy
from aicache.llm.ollama import OllamaLLM
from aicache.llm.simulated import SimulatedLLM
from aicache.metrics.correctness import is_correct
from aicache.normalizer import Normalizer
from aicache.router import ComputationRouter

from workloads.generators import generate_workload

OLLAMA = "http://localhost:11434/v1"


def make_main(spec: str):
    """Returns (callable(item, text) -> answer text, name)."""
    if spec == "simulated":
        sim = SimulatedLLM(seed=0)
        return (lambda item, text: sim.complete(text, task_hint={"operation": item.task_type, "params": item.bound_params}).text), "simulated"
    kind, _, model = spec.partition(":")
    assert kind == "ollama" and model, "use --main simulated or --main ollama:<model>"
    llm = OllamaLLM(model_name=model, options={"temperature": 0, "seed": 0})
    return (lambda item, text: llm.complete(text).text), spec


def run_stream(cache: ActionCache, workload, main, *, learn: bool) -> dict:
    rows = []
    disagreements: list[dict] = []
    real_observe = cache.learner.observe if cache.learner else None
    if real_observe:
        def traced(name, template, args, agrees, _o=real_observe):
            if not agrees:
                disagreements.append({"action": name, "template": template, "args": args, "request": traced.current})
            return _o(name, template, args, agrees)
        traced.current = ""
        cache.learner.observe = traced  # type: ignore[method-assign]
    llm_calls = 0
    for item in workload:
        calls = {"n": 0}

        def llm(text, item=item, calls=calls):
            calls["n"] += 1
            return main(item, text)

        if real_observe:
            traced.current = item.text  # type: ignore[name-defined]
        if not learn:
            saved, cache.learner = cache.learner, None   # freeze: no learning, no probation bookkeeping
        t0 = time.perf_counter()
        out = cache.handle(item.text, llm)
        ms = (time.perf_counter() - t0) * 1000
        if not learn:
            cache.learner = saved
        llm_calls += calls["n"]
        computed = out.source.startswith("computation")
        rows.append({
            "task": item.task_type, "source": out.source, "action": out.action,
            "correct": is_correct(str(out.value) if computed else out.text, item.expected_answer),
            "learning": out.learning, "ms": ms,
        })
    n = len(rows)
    computed = [r for r in rows if r["source"].startswith("computation")]
    return {
        "n": n, "computed": len(computed), "llm_calls": llm_calls,
        "wrong_executions": sum(not r["correct"] for r in computed),
        "overall_correct": sum(r["correct"] for r in rows),
        "computed_series": [int(r["source"].startswith("computation")) for r in rows],
        "computed_by_task": dict(Counter(r["task"] for r in computed)),
        "requests_by_task": dict(Counter(r["task"] for r in rows)),
        "wrong_by_task": dict(Counter(r["task"] for r in computed if not r["correct"])),
        "learning_outcomes": dict(Counter(r["learning"] for r in rows if r["learning"])),
        "probation_disagreements": disagreements[:10],
        "median_ms_computed": statistics.median(r["ms"] for r in computed) if computed else None,
        "median_ms_llm": statistics.median(r["ms"] for r in rows if r not in computed) if len(computed) < n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalizer", default="llama3", help="Ollama model that proposes actions")
    ap.add_argument("--main", default="simulated", help="simulated, or ollama:<model>")
    ap.add_argument("--learn-seed", type=int, default=1)
    ap.add_argument("--test-seed", type=int, default=2)
    ap.add_argument("--min-similarity", type=float, default=0.6)
    ap.add_argument("--min-agreements", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N learning requests")
    ap.add_argument("--output", default="results/learning/run.json")
    a = ap.parse_args()

    embedder = SentenceTransformerEmbedder()
    router = ComputationRouter(embedder, min_similarity=a.min_similarity)
    store = ActionStore(":memory:", embedder_id=embedder.model_name)
    normalizer = Normalizer(OLLAMA, a.normalizer, timeout_s=120)
    counted = {"n": 0}
    _propose = normalizer.propose

    def propose(*args, **kw):
        counted["n"] += 1
        return _propose(*args, **kw)

    normalizer.propose = propose  # type: ignore[method-assign]
    cache = ActionCache(router, store, normalizer,
                        LearnPolicy(min_agreements=a.min_agreements, reject_after=2), synchronous=True)
    main_llm, main_name = make_main(a.main)

    learn_set = generate_workload(seed=a.learn_seed)
    test_set = generate_workload(seed=a.test_seed)
    if a.limit:
        learn_set = learn_set[: a.limit]
    t0 = time.time()
    learn = run_stream(cache, learn_set, main_llm, learn=True)
    learn["seconds"] = round(time.time() - t0)
    actions = [{"name": x.name, "status": x.status, "expression": x.compute,
                "phrasings": [{"text": t.text, "status": t.status, "agree": t.agree, "disagree": t.disagree}
                              for t in x.templates]} for x in store.list()]
    test = run_stream(cache, test_set, main_llm, learn=False)
    result = {"normalizer": a.normalizer, "main": main_name, "min_similarity": a.min_similarity,
              "learn_seed": a.learn_seed, "test_seed": a.test_seed, "normalizer_calls": counted["n"],
              "min_agreements": a.min_agreements, "learn_stream": learn, "held_out_stream": test, "actions": actions,
              "telemetry": cache.router.telemetry.snapshot()["learning"]}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "actions"}, indent=2))
    print("\nACTIONS")
    for x in actions:
        by = Counter(p["status"] for p in x["phrasings"])
        print(f"  {x['name']:20s} {x['expression']:45s} phrasings: {dict(by)}")


if __name__ == "__main__":
    main()
