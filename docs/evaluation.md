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
