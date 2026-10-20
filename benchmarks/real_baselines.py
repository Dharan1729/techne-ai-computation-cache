"""Stronger real-model baselines (EXP-015).

EXP-008 compared the computation cache against a *naive* real-model
baseline: the raw question sent to a local 8B model, sampled at the
model's default temperature, with no reasoning scaffolding and no tool.
That is the weakest honest baseline, and it inflated the gap. This script
adds the baselines a careful reviewer would ask for, all on the same
workload, same model, temperature pinned to 0:

  llm_direct   raw question -> model (EXP-008's baseline, now temp 0)
  llm_cot      question + "think step by step" instruction
  llm_router   model picks a task + extracts parameters as JSON, then the
               *same deterministic executor* the computation cache uses
               computes the answer. This is how a tool-using agent
               actually works today, so it is the baseline that matters.
  comp_cache   computation cache, falling back to llm_direct
  hybrid       computation cache, falling back to llm_router

Usage:
    python -m benchmarks.real_baselines --model llama3 --seed 42 \
        --output-dir results/real_baselines
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from aicache.cache.computation import ComputationCache
from aicache.computation.tasks import TASK_REGISTRY
from aicache.llm.ollama import OllamaLLM
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.correctness import is_correct

from workloads.generators import generate_workload

COT_SUFFIX = (
    "\n\nThink step by step, showing each calculation. "
    "End your reply with a line of the form 'Final answer: <number>'."
)


def router_prompt(question: str) -> str:
    lines = []
    for name, spec in TASK_REGISTRY.items():
        params = ", ".join(spec.required_param_names)
        lines.append(f'- "{name}": parameters in order = [{params}]')
    catalogue = "\n".join(lines)
    return (
        "You are a tool router. Choose exactly one tool for the user's question "
        "and extract its parameters.\n"
        f"Tools:\n{catalogue}\n\n"
        "Return only JSON of the form "
        '{"task": "<tool name>", "params": [<numbers in the tool\'s parameter order>]}. '
        "Give each number exactly as written in the question (for example write 20 for "
        "'20%', not 0.2).\n\n"
        f"Question: {question}"
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pctile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def parse_cot(text: str) -> str:
    m = re.findall(r"final answer\s*[:\-]?\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    return m[-1].replace(",", "") if m else text


def run_router(llm_json: OllamaLLM, question: str) -> tuple[float | None, object, str]:
    resp = llm_json.complete(router_prompt(question))
    try:
        data = json.loads(resp.text)
        spec = TASK_REGISTRY[data["task"]]
        nums = [float(str(v).replace(",", "").replace("$", "")) for v in data["params"]]
        answer = spec.bind_and_execute(nums)
        return answer, resp, ""
    except Exception as exc:  # malformed JSON, unknown task, wrong arity
        return None, resp, f"{type(exc).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N requests")
    ap.add_argument("--output-dir", default="results/real_baselines")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    workload = generate_workload(seed=args.seed)
    if args.limit:
        workload = workload[: args.limit]

    opts = {"temperature": 0, "seed": 0}
    llm = OllamaLLM(model_name=args.model, options=opts)
    llm_json = OllamaLLM(model_name=args.model, options=opts, response_format="json")

    # Warm the model and the embedder so one-off load time is not billed to any condition.
    llm.complete("What is 1 plus 1?")
    llm_json.complete(router_prompt("What is 1 plus 1?"))
    cache = ComputationCache(identifier=TaskIdentifier())
    cache.lookup("What is 2 plus 2?")

    rows: list[dict] = []

    def record(cond, item, correct, latency_ms, in_tok, out_tok, llm_called, note=""):
        rows.append(
            dict(
                condition=cond, request_id=item.request_id, task_type=item.task_type,
                correct=bool(correct), latency_ms=latency_ms, input_tokens=in_tok,
                output_tokens=out_tok, llm_called=llm_called, note=note,
            )
        )

    t_start = time.time()
    for n, item in enumerate(workload):
        # llm_direct
        r = llm.complete(item.text)
        record("llm_direct", item, is_correct(r.text, item.expected_answer),
               r.latency_ms, r.input_tokens, r.output_tokens, True)
        # llm_cot
        r = llm.complete(item.text + COT_SUFFIX)
        record("llm_cot", item, is_correct(parse_cot(r.text), item.expected_answer),
               r.latency_ms, r.input_tokens, r.output_tokens, True)
        # llm_router
        ans, r, err = run_router(llm_json, item.text)
        ok = ans is not None and is_correct(str(ans), item.expected_answer)
        record("llm_router", item, ok, r.latency_ms, r.input_tokens, r.output_tokens, True, err)
        # computation cache lookup (shared by comp_cache and hybrid)
        t0 = time.perf_counter()
        lk = cache.lookup(item.text)
        lk_ms = (time.perf_counter() - t0) * 1000
        if lk.hit:
            ok = is_correct(str(lk.answer), item.expected_answer)
            for cond in ("comp_cache", "hybrid"):
                record(cond, item, ok, lk_ms, 0, 0, False, "hit")
        else:
            r = llm.complete(item.text)
            record("comp_cache", item, is_correct(r.text, item.expected_answer),
                   lk_ms + r.latency_ms, r.input_tokens, r.output_tokens, True, "fallback")
            ans, r, err = run_router(llm_json, item.text)
            ok = ans is not None and is_correct(str(ans), item.expected_answer)
            record("hybrid", item, ok, lk_ms + r.latency_ms, r.input_tokens, r.output_tokens,
                   True, "fallback" + (f":{err}" if err else ""))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(workload)} done, {time.time() - t_start:.0f}s", flush=True)

    # ---- summarise
    summary: dict = {"model": args.model, "seed": args.seed, "temperature": 0,
                     "n_requests": len(workload), "conditions": {}}
    conds = ["llm_direct", "llm_cot", "llm_router", "comp_cache", "hybrid"]
    for c in conds:
        cr = [r for r in rows if r["condition"] == c]
        n = len(cr)
        k = sum(r["correct"] for r in cr)
        lo, hi = wilson(k, n)
        lat = [r["latency_ms"] for r in cr]
        by_task: dict[str, list[bool]] = defaultdict(list)
        for r in cr:
            by_task[r["task_type"]].append(r["correct"])
        summary["conditions"][c] = {
            "n": n, "correct": k, "accuracy": k / n, "wilson95": [lo, hi],
            "p50_ms": statistics.median(lat), "p95_ms": pctile(lat, 0.95),
            "mean_ms": statistics.fmean(lat),
            "llm_call_rate": sum(r["llm_called"] for r in cr) / n,
            "mean_input_tokens": statistics.fmean(r["input_tokens"] for r in cr),
            "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in cr),
            "accuracy_by_task": {t: sum(v) / len(v) for t, v in sorted(by_task.items())},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "results.csv", "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(json.dumps({c: {k: v for k, v in s.items() if k != "accuracy_by_task"}
                      for c, s in summary["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
