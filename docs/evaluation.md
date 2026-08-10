# Evaluation Methodology

## Metrics tracked

Per request (`aicache/metrics/types.py::RequestResult`): outcome
(`NO_CACHE` / `HIT_VALID` / `HIT_INVALID` / `MISS` / `FALLBACK`),
correctness (checked independently of outcome — see below), latency,
input/output tokens, simulated cost, whether the LLM was called at all,
and the identified task type + similarity score where applicable.

Aggregated per baseline (`aicache/metrics/aggregate.py::summarize`):
request count, hit/valid-hit/invalid-hit/miss/fallback rates, LLM-call
rate, overall correctness rate, total and average tokens/cost, and
p50/p95/p99 latency. Deliberately more than hit rate alone — see H7 in
`docs/hypotheses.md` for why hit rate by itself would have been
misleading on this project's own results.

## Correctness checking

`aicache/metrics/correctness.py` parses the first number out of whatever
text was ultimately served and compares it against the workload item's
own ground truth (computed directly from `TASK_REGISTRY`, independent of
whatever cache/backend served the request) with a ±1e-3 relative/absolute
tolerance. The tolerance width and its rationale (loose enough for
reasonable rounding, tight enough to still catch genuine parameter
mismatches) are documented inline in that module and were tuned once,
during EXP-001 in `docs/experiment-log.md`, in response to an observed
false-negative — not chosen a priori to make later results look better.

## What this evaluation does *not* do

Per the brief's own statistical-discipline requirement (§33) and this
project's honesty standard: **the headline numbers in
`docs/experiment-log.md` come from a single run at a single seed.** No
repeated-runs variance, no confidence intervals, are reported anywhere
in this repo. This is a real limitation, not an oversight — treat every
specific percentage in this repo's docs as illustrative of a pattern
(computation caching being 100% correct while semantic caching is not,
across a real threshold sweep) rather than as a precise, statistically
validated estimate. The workload generator is fully seeded and
reproducible (`workloads/generators.py`), so running with several seeds
and reporting mean/variance is mechanically straightforward — it simply
wasn't done in this pass, and is listed as follow-up work rather than
silently implied to have been done.

## External baselines (GPTCache, RedisVL)

Same correctness methodology as this repo's own baselines
(`is_correct` against independent ground truth), applied to whatever
text the external system actually served — see
`docs/architecture.md`'s "External comparison baselines" section for
the subprocess/protocol mechanics. One methodological note specific to
these two: **both are optional and environment-dependent** (a specific
local venv layout, and for RedisVL, a reachable Redis instance) — a
reader who clones this repo without that exact local setup will see
`benchmarks/run_experiment.py` skip them with a printed reason rather
than reproduce their rows in `results/results.csv`. This repo's own
four baselines never depend on either being present, by design.

## Level 3 additions

Level 3 (`docs/level3.md`) reuses this same methodology unchanged —
`aicache/metrics/aggregate.py::summarize` and the
outcome/correctness-vs-hit-rate discipline above apply identically —
with two additions specific to workflow tasks over live state:

- **Per-task correctness checking**
  (`aicache/metrics/workflow_correctness.py`) replaces the single
  generic `parse_final_number` comparison with a small, honest parser
  per task type (booleans compared exactly, dollar amounts via a
  `$`-prefixed regex with the same ±1e-3 tolerance, risk category via
  keyword search) — a structured multi-field result isn't reducible to
  "the last number in the text" the way a plain arithmetic answer is.
  Tested only against this repo's own renderer
  (`aicache/llm/workflow_rendering.py`); real-backend validation of this
  parsing layer specifically (the way EXP-007/008 validated Level 1/2's
  parser against Gemini and Ollama) is listed as follow-up in
  `docs/level3.md`'s Limitations, not attempted in this phase.
- **Per-stage latency decomposition** (`identify_ms` / `extract_ms` /
  `validate_ms` / `fetch_ms` / `execute_ms`, via
  `aicache/cache/workflow_computation.py::LatencyBreakdown`) alongside
  the aggregate p50/p95/p99 figures above — see `docs/level3.md`'s L5
  and RQ5 for what this decomposition revealed that an aggregate number
  would have hidden.

The single-run, no-repeated-trials limitation stated above applies to
Level 3's results identically — see `docs/level3.md`'s own Limitations
section rather than restating it here.

## Reproducibility

- Python 3.11.8, macOS (Darwin), CPU only — see
  `docs/experiment-log.md` for the full environment note.
- Workload generation is seeded (`--seed`, default 42) and independently
  verified reproducible by `tests/test_workload_generator.py`.
- The simulated LLM backend (`aicache/llm/simulated.py`) is itself
  seeded for its latency distribution; its *answers* are fully
  deterministic (computed directly from `TASK_REGISTRY`), so re-running
  the same workload against it always produces the same correctness
  outcomes — only latency figures have any randomness, and that
  randomness is seeded too.
- Dependency versions are declared in `pyproject.toml` but not pinned to
  exact patch releases as of this writing — a reproducibility gap
  worth closing before treating any specific number here as
  citable outside this repo.
