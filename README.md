# ai-computation-cache

This repository investigates whether some AI workloads benefit more from
caching **reusable computation structure** than from caching **final
responses** — and, where they do, what that costs in terms of coverage
(how often reuse is even possible) versus what response caching gains in
correctness risk.

It is a controlled engineering investigation, not a product and not a
paper claiming a new technique. See `docs/prior-art.md` for what's
established prior art (exact caching, semantic caching, plan caching,
cost-aware eviction) versus what this repo actually tests (one specific
combination of representation, identification, binding, and validation,
on a narrow, deterministic workload family).

## Problem

Exact-match caching only reuses byte-identical requests. Semantic
caching relaxes that to "similar enough" requests — which is unsafe when
a request's *answer* depends on parameters that vary independently of
its *wording*:

```
"What is X + Y?"        X=10,  Y=20   -> 30
"Calculate X plus Y."   X=51,  Y=93   -> 144
"Can you sum X and Y?"  X=712, Y=381  -> 1093
```

These three requests are near-identical in wording and completely
different in answer. What's stable across them isn't the response —
it's the computational structure, `ADD(x, y)`. Full write-up:
`docs/problem.md`.

## Hypothesis

> Some AI workloads may benefit more from caching reusable computation
> structure than from caching final responses, when parameters vary but
> structure doesn't.

Tested via eight falsifiable sub-hypotheses (H1–H8) — see
`docs/hypotheses.md` for each one's actual outcome against this repo's
own experiment.

## Architecture

Four conditions run over the same workload with the same LLM backend
(full detail: `docs/architecture.md`):

```
A. No cache          Request -> LLM -> Answer
B. Exact cache        Request -> exact match? -> cached answer : (LLM -> store)
C. Semantic cache      Request -> embed -> nearest neighbor above threshold?
                              -> cached answer : (LLM -> store)
D. Computation cache   Request -> identify task -> retrieve computation
                              -> extract + bind parameters -> validate
                              -> valid? execute directly : fall back to LLM
```

Scope: **Level 1 (deterministic arithmetic) and Level 2 (parameterized
business formulas) only** — closed-form, deterministic tasks, which is
what makes "reuse the procedure, rebind fresh parameters, execute
directly" a coherent design. Level 3 (database-backed workflows) and
Level 4 (agent workflows) need a different computation representation
and are explicitly out of scope for this prototype — see
`docs/architecture.md`'s scope note.

## Experimental setup

- Workload: 9 canonical tasks (`add`, `subtract`, `multiply`,
  `percentage`, `compound_interest`, `revenue`, `discounted_price`,
  `tax`, `margin`), each with several paraphrase templates and randomized
  parameters. 150 requests by default, fully seeded and reproducible
  (`workloads/generators.py`).
- LLM backend: a **simulated** backend by default (deterministic,
  offline, zero cost — see `aicache/llm/simulated.py` for exactly what's
  real (token-count heuristic, published pricing) vs. synthetic (latency
  distribution) about it), a pluggable real **Gemini** backend
  (`aicache/llm/gemini.py`, inactive unless `GEMINI_API_KEY` is set;
  free-tier-capped at 20 requests/day), and a pluggable real **Ollama**
  backend (`aicache/llm/ollama.py`, fully local, no quota — used for the
  full-scale real-model validation in EXP-008).
- Embedding model: `redis/langcache-embed-v1` — chosen after directly
  comparing it against a generic sentence-similarity model and finding
  it separates genuine paraphrases from different-parameter requests far
  more sharply (see `docs/experiment-log.md`, EXP-005, and
  `aicache/matching/embedder.py`'s docstring).
- **External comparison baselines**: real GPTCache and real RedisVL,
  each run through the identical protocol as this repo's own semantic
  cache, in their own dependency-isolated subprocesses — see
  `docs/architecture.md`'s "External comparison baselines" section.
  Optional: skipped gracefully if the required venv/Redis instance isn't
  present.
- Metrics: hit/valid-hit/invalid-hit/miss/fallback rates, correctness
  (checked independently against ground truth, not assumed from
  outcome), tokens, cost, latency — see `docs/evaluation.md`.

Run it:

```
pip install -e .
python -m pytest tests/ -q                    # 51 tests (48 always run + 3 external, auto-skipped if unavailable)
python -m benchmarks.run_experiment           # writes results/results.csv, results/summary.json
python -m benchmarks.plots                    # writes results/*.png

# Real-model validation (needs `ollama serve` running + a model pulled, e.g. `ollama pull llama3`):
python -m benchmarks.run_experiment --backend ollama --skip-threshold-sweep --skip-external \
    --output-dir results/real_backend_ollama
python -m benchmarks.plot_real_backend
```

## Results

From the run documented in full in `docs/experiment-log.md` (EXP-006;
150 requests, seed 42, simulated LLM backend, `redis/langcache-embed-v1`
embeddings):

| Baseline | Hit rate | Valid | Invalid | Correctness | Tokens | Cost (USD) |
|---|---|---|---|---|---|---|
| A — No cache | 0.0% | — | — | 100.0% | 2193 | 0.00338 |
| B — Exact cache | 0.0% | 0.0% | 0.0% | 100.0% | 2193 | 0.00338 |
| C — This repo's semantic cache (0.75) | 64.0% | 63.3% | 0.7% | 99.3% | 778 | 0.00120 |
| **RedisVL** (real system, same embedding model) | 64.0% | 63.3% | 0.7% | 99.3% | 778 | 0.00120 |
| **GPTCache** (real system, generic embedding model) | 88.7% | 18.0% | **70.7%** | **29.3%** | 194 | 0.00033 |
| D — Computation cache | 77.3% | 77.3% | **0.0%** | **100.0%** | 346 | 0.00061 |

![Hit rate split by correctness](results/hit_rate_by_baseline.png)

**Two findings, not one.** First — the credibility check: this repo's
own from-scratch semantic-cache reimplementation matches RedisVL's real,
production implementation *within noise, at every threshold tested*.
It isn't a strawman.

![Correctness vs threshold across three real systems](results/cross_system_correctness.png)

Second — embedding model choice turns out to matter as much as, or more
than, caching algorithm choice: GPTCache's own out-of-the-box embedding
default trails RedisVL/this-repo's-cache by up to 70 percentage points
of correctness at the *identical* nominal threshold, on the *identical*
workload, run through the *identical* protocol. The full sweep
(`results/semantic_threshold_sweep.png`) shows this isn't fixable by
retuning GPTCache's threshold alone within any reasonable range.

**Headline finding, computation vs. response caching**: the computation
cache reaches a *higher* hit rate (77.3%) than any of the three
semantic-response-cache configurations at their safe operating point,
at *zero* observed invalid hits and the lowest token/cost footprint of
any caching condition tested (346 tokens / $0.00061, an 84% reduction
vs. no-cache). Its 22.7% fallback rate is a deliberate, correctness-preserving
refusal to guess, not a failure. Full numbers and interpretation:
`docs/experiment-log.md` (EXP-004 through EXP-006). All charts
(`tokens_and_cost_by_baseline.png`, `latency_by_baseline.png`,
`semantic_threshold_sweep.png`, `cross_system_correctness.png`) are in
`results/`.

### A third finding: what a real model adds (EXP-008, real local LLM, no simulation)

Every result above uses a simulated LLM that is correct by construction
— useful for isolating caching-algorithm behavior, but blind to one
thing: whether the underlying model can actually do the computation.
Re-running all four core baselines at full scale (150 requests) against
a real local model (`llama3:8b` via Ollama, chosen specifically because
it runs unlimited and free on this project's own hardware, after a
Gemini API free-tier key turned out to cap at 20 requests/day) answered
that question directly:

| Baseline (real LLM, same model throughout) | Hit rate | Correctness |
|---|---|---|
| A — No cache | 0.0% | **48.7%** |
| B — Exact cache | 0.0% | 46.7% |
| C — This repo's semantic cache | 64.0% | 54.0% |
| D — Computation cache | 77.3% | **86.7%** |

![Correctness by baseline, real backend](results/real_backend_ollama/correctness_real_backend.png)

Calling the model directly for every request got these questions right
less than half the time — and 0% of the time on `compound_interest` and
`margin` specifically (wrong on all 18 and all 12 attempts):

![Real LLM accuracy by task type](results/real_backend_ollama/llm_accuracy_by_task_type.png)

Because the computation cache routes 77.3% of its traffic around the
model entirely, its correctness came out to 86.7% — using the *exact
same* underlying LLM as the 48.7%-correct `no_cache` baseline. This is a
different, and arguably larger, argument for computation caching than
the stale-reuse story above: it's not just safer than semantic caching,
it's a hedge against the model's own unreliability at deterministic
sub-tasks — something no response-caching approach can offer, since it
still asks the model to compute the answer on every miss. Full
breakdown, methodology, and the (real, reported-not-hidden) limitations
of a single un-repeated local run: `docs/experiment-log.md` EXP-008,
and the addendum in `docs/hypotheses.md`.

## Limitations

- **Single run, single seed** — no repeated-runs variance or confidence
  intervals are reported. See `docs/evaluation.md`.
- **Deterministic tasks only** — the computation cache's zero-invalid-hit
  result depends entirely on Level 1/2 tasks being closed-form and
  side-effect-free. It does not generalize to Level 3/4 without a
  different representation.
- **Positional parameter extraction** — a real, documented limitation
  (`aicache/binding/extractor.py`); two template bugs exploiting exactly
  this limitation were found and fixed during development (see
  `docs/experiment-log.md`, EXP-002/EXP-003) — not swept under the rug.
- **Task-identification threshold is a real, narrowed-but-not-eliminated
  trade-off** — see H6 in `docs/hypotheses.md`: even with a strong
  embedding model, correct- and incorrect-match similarity ranges
  overlap slightly, and the calibrated threshold (0.52) sits inside
  that overlap by design.
- **External baselines are environment-dependent** — GPTCache/RedisVL
  results require a specific local venv layout and, for RedisVL, a
  reachable Redis instance; both are skipped gracefully (not faked) when
  absent. See `docs/evaluation.md`.
- **Simulated LLM backend by default** — see `aicache/llm/simulated.py`
  for exactly what's real vs. synthetic about its numbers. Two real
  backends are wired and working: Gemini (`aicache/llm/gemini.py`,
  hit a hard 20 requests/day free-tier quota — see EXP-007) and Ollama
  (`aicache/llm/ollama.py`, ran the full 150-request core comparison
  with no quota limit — EXP-008). The real-backend runs found two real
  bugs in this repo's own harness before they could distort results (a
  correctness-parser bug, EXP-007; a cost-fallback bug that would have
  mislabeled real-model costs as Claude Haiku pricing, EXP-008) — both
  fixed and covered by tests. EXP-008's threshold sweep and external
  (GPTCache/RedisVL) comparison weren't re-run against a real backend
  (deliberately out of scope — see that entry); those remain validated
  only against the simulated backend.
- **No eviction/admission policy of any kind** — see
  `docs/eviction-future-work.md`; explicitly deferred, not solved.

## Level 3 and 3b: database-backed enterprise workflows

Extends the same "cache the plan, not the data" idea from closed-form
arithmetic to workflows that must read live, mutable state (customer/
transaction/policy records) before answering — full write-up:
`docs/level3.md`. Headline: 99.1% hit rate, **0% invalid hits**, on a
workload where semantic caching hits 35.6% invalid — and a dedicated
freshness experiment demonstrating the mechanism directly: a database
mutation flips the always-fresh condition's answer correctly while
semantic caching and a deliberate stale-fetch ablation keep silently
serving the pre-mutation answer.

`docs/level3b.md` sweeps the two variables the primary comparison held
fixed: structural locality (computation-cache hit rate/correctness stay
flat across a 40x range of entity diversity; semantic caching's
correctness falls *while its hit rate rises* over the same range) and
state volatility (correctness under mutation exposure follows
`1 − mutation_rate` precisely for both semantic caching and the stale
ablation).

## Differentiation

`docs/differentiation.md` — is this actually a different idea, and does
it matter? Evidence at four levels of rigor: this repo's own controlled
experiments (0% invalid-hit rate at every phase and every workload
tested), a real bank's production semantic cache (documented false
positives up to 99% zero-shot, 3.8% best-case after months of
hardening — see `docs/prior-art.md`'s "Real-world evidence" section), a
live, reproducible demo against the real `langgraph` library showing
its production node-caching feature (`CachePolicy`) is architecturally
equivalent to this repo's stale-fetch ablation, and the closest
published academic work (Agentic Plan Caching, arXiv:2506.14852; W5H2,
arXiv:2602.18922) — which validates the core architecture but doesn't
rigorously measure the freshness question this repo's Level 3/3b phases
were built to answer. Includes the honest counter-case: what's already
been published, and where this repo's claims stop.

## Prior art

`docs/prior-art.md` — Redis LangCache, **RedisVL and GPTCache (both
directly benchmarked, not just described)**, LangGraph `CachePolicy`
(directly exercised against the real library), CrewAI, AutoGen, vLLM
prefix caching, LMCache, **Agentic Plan Caching and W5H2 (real,
specific papers, not a research-direction placeholder)**, Google
PARROT, TinyLFU/W-TinyLFU, CacheLib, and a real banking semantic-cache
production case study, all mapped against this repo's own design
choices. Nothing in this repo claims to invent computation caching,
semantic caching, or cost-aware eviction.

## Future work

Level 3 and 3b (above) are done, not future work. What's left: Level 4
(parameterized workflow *graphs* — branching execution, tool-result
caching with graduated TTL freshness rather than Level 3's single
always-fresh class — see `docs/level3.md`'s "Deferred: Level 4"),
running the real `langgraph` comparison through this repo's full
benchmark harness (currently a standalone reproducible demo, not yet
integrated — see `docs/differentiation.md` §3 and §7), and
`docs/eviction-future-work.md` (admission/eviction at real scale). See
also the "Next experiments" notes throughout `docs/experiment-log.md`.

## Repository layout

```
ai-computation-cache/
├── docs/                problem, prior-art, architecture, hypotheses,
│                        evaluation, eviction-future-work, experiment-log,
│                        level3, level3b, differentiation
├── src/aicache/         the library: cache/, matching/, computation/,
│                        binding/, validation/, metrics/, llm/, enterprise/
├── workloads/           Level 1/2 workload generator + templates;
│                        workloads/level3/ (enterprise workflow workload)
├── benchmarks/          baselines.py, run_experiment.py, plots.py,
│                        plot_real_backend.py, external/ (real GPTCache +
│                        RedisVL adapters), level3/ (baselines,
│                        freshness/locality/volatility experiments)
├── tests/                94 unit + integration tests
├── results/             results.csv, summary.json, generated charts
│                        (simulated-backend, EXP-006 headline numbers)
│   ├── real_backend_ollama/  real-model validation, EXP-008
│   └── level3/           Level 3/3b results, sweeps, and charts
└── configs/             (reserved for future experiment configs)
```
