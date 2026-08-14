# Differentiation Analysis: Is Computation Caching a Genuinely Different, Worthwhile Idea?

Purpose: answer one question directly, with evidence at every level of
rigor available — this repo's own controlled experiments, a real
production deployment's documented failure, a real production
framework's actual code, and the closest published academic work —
rather than asserting novelty. Structured basic → advanced: each section
assumes the previous one, and the honest counter-case (§7) gets equal
weight to the case for the idea, per this project's standing practice
of reporting what doesn't hold up alongside what does
(`docs/experiment-log.md`, `docs/evaluation.md`).

## 1. The differentiator, stated as simply as possible

Every caching mechanism found during this research — this repo's own
exact/semantic baselines, GPTCache, RedisVL, LangGraph's `CachePolicy`,
CrewAI's tool cache, AutoGen's LLM cache, and a real bank's production
semantic cache — caches **what the answer was**. This repo's computation
cache caches **what operation to run**, and never caches what the
answer was: parameters are re-extracted from the current request and
external state is re-fetched from the current database on every single
reuse, with a validation gate as a hard backstop before anything
executes. See `docs/architecture.md` and `docs/level3.md`'s "cache the
plan, not the data" section for the full mechanism.

That's the whole idea in one sentence. Everything below is evidence for
whether it actually matters.

## 2. Basic evidence: this repo's own controlled experiments

The most basic tier — internally controlled, this repo's own workload,
but rigorous within that scope:

| Phase | Semantic/response cache invalid-hit rate | Computation cache invalid-hit rate |
|---|---|---|
| Level 1/2, tuned embedding model (`docs/hypotheses.md` H2) | 0.7% | 0.0% |
| Level 1/2, generic embedding model (EXP-004) | 36.0% | 0.0% |
| Level 3, enterprise workflows (`docs/level3.md` Experiment A) | 35.6% | 0.0% |
| Level 3b, locality sweep, worst point tested (`docs/level3b.md`) | 52.8% | 0.0% |
| Level 3, real GPTCache, generic embedding (EXP-006) | 70.7% | N/A (not GPTCache's design) |

Zero, at every point tested, across two workload families and a 40x
range of structural locality — not because the workload was built to
flatter the result (`docs/evaluation.md`'s standing constraint: never
optimize the benchmark for a desired outcome), but because the
mechanism structurally cannot produce a false hit the way response
caching can: it either identifies a known operation and executes it
fresh, or it doesn't and falls back to the LLM. There is no code path
where it can execute *the wrong* operation on *someone else's* data
undetected — a misidentification either gets caught by extraction/
validation (two real, observed instances in `docs/level3.md`'s
Experiment A) or the request never reaches an unsafe execution.

## 3. Real-world evidence: what happens when this problem meets production

Basic-tier evidence shows the mechanism works on this repo's own
workload. This tier asks: is the problem it solves even real outside
this repo?

**Yes — a real bank hit it.** `docs/prior-art.md`'s new "Real-world
evidence" section covers this in full; the headline: a financial
services org's production semantic cache matched *"I don't want this
card anymore"* to **investment account closure** at 88.7% confidence.
Their measured false-positive rate: up to 99% zero-shot, and **3.8%**
after months of production hardening (curated FAQ/distractor sets,
seven embedding models tried, quality filtering) — their *best*
achieved number, still non-zero, on a task (banking actions) where a
non-zero false-positive rate is a real liability, not a rounding error.
This repo's 0.0% on a structurally similar failure mode (entity/intent
collision) required no threshold tuning, no curated distractor set, and
no embedding model search.

**Yes — the most popular production agent framework's real caching
feature reproduces the mechanism this repo's ablation (E) exists to
warn about.** Not a synthetic worst case — a live, reproducible demo
against the actual `langgraph` library (not a reimplementation):

```python
db = {'balance': 100}
# node fetches db['balance'], cache_policy=CachePolicy(ttl=2)

call 1: {'balance': 100}                                        # real fetch
db['balance'] = 999                                              # mutation
call 2 (same input, immediately after mutation): {'balance': 100}  # STALE
call 3 (after the 2s TTL expires): {'balance': 999}                # correct
```

`LangGraph.CachePolicy` memoizes a node's output by hashing its input,
with an optional TTL — architecturally identical to this repo's baseline
E, not D. LangChain's own GitHub has an open issue asking to document
"when and why to use the cache parameter" (`langchain-ai/docs#3736`),
and none of the tutorials found during this research mention staleness
risk. This repo's Level 3b volatility sweep (`docs/level3b.md`
EXP-012) is, as far as this search found, the only place this specific
risk has been *quantified* as a function of mutation rate rather than
demonstrated as a single anecdote.

## 4. Academic evidence: has this already been published?

Partially — and this is the tier where intellectual honesty matters
most. Two papers found during this research are close enough that they
change what this repo can claim:

- **[Agentic Plan Caching (arXiv:2506.14852)](https://arxiv.org/pdf/2506.14852)** —
  caches the plan, re-fetches data on reuse, validates before reuse.
  This is this repo's core mechanism, independently published. Their
  reported hit rate (40–60%) is *lower* than this repo's 99.1%
  (`docs/level3.md` EXP-009) — not a contradiction; consistent with
  this repo's own locality sweep (`docs/level3b.md` EXP-011) showing
  hit rate should fall as structural repetition falls, and their
  benchmark almost certainly has less structural repetition than this
  repo's deliberately-repetitive one.
- **[W5H2 (arXiv:2602.18922)](https://arxiv.org/html/2602.18922)** —
  independently arrived at a near-identical split to this repo's
  `TaskIdentifier` (coarse, embedding-based) + `entity_extractor.py`
  (exact, regex-based): canonicalize (What, Where) as the cache key,
  inject parameters separately at execution. Their own numbers (91.1%
  vs. GPTCache's 37.9% accuracy on one benchmark) independently confirm
  H2/H6 and Level 3 Experiment A's finding that structured extraction
  beats semantic similarity for this exact failure class.

**What neither paper does, that this repo does**: a *controlled
ablation* isolating staleness as the sole variable (baseline E shares
one class with D via a single constructor flag — `use_stale_cache`,
`aicache/cache/workflow_computation.py` — so nothing else differs
between them), and a *quantified* mutation-rate response curve
(`docs/level3b.md` EXP-012, matching the predicted `1 − rate` line
within sampling noise at every point). Agentic Plan Caching mentions
re-fetching qualitatively; W5H2 doesn't address freshness at all. This
is the actual, defensible, narrow claim: **not** "cache the plan" (already
published), but "here is exactly how much reuse-without-refetch costs
you, as a function of how fast the world changes underneath it" — measured,
not asserted.

## 5. Quantitative synthesis — every number found, side by side

| Source | What was measured | Result |
|---|---|---|
| This repo, Level 1/2 (tuned embedding) | Semantic cache invalid-hit rate | 0.7% |
| This repo, Level 3 (entity-bearing) | Semantic cache invalid-hit rate | 35.6% |
| This repo, Level 3b (worst locality point) | Semantic cache invalid-hit rate | 52.8% |
| This repo, real GPTCache (EXP-006) | Semantic cache invalid-hit rate | 70.7% |
| Real bank, zero-shot (InfoQ case study) | Semantic cache false-positive rate | up to 99% |
| Real bank, best achieved after hardening | Semantic cache false-positive rate | 3.8% |
| W5H2 paper, MASSIVE benchmark | GPTCache accuracy vs. structured-key accuracy | 37.9% vs. 91.1% |
| W5H2 paper, NyayaBench v2 | GPTCache accuracy vs. structured-key accuracy | 49.1% vs. 55.3% |
| This repo, all phases, computation cache | Invalid-hit rate | **0.0%, every point tested** |
| Agentic Plan Caching paper | Hit rate / cost reduction | 40–60% hit / up to 60% cost reduction |
| This repo, Level 3 primary (EXP-009) | Hit rate / token reduction | 99.1% hit / 99.0% token reduction |
| This repo, Level 3b volatility sweep (EXP-012) | Stale-ablation correctness vs. mutation rate | matches `1 − rate` within noise, every point |

No source found — this repo's own experiments, the real bank, or either
academic paper — reports a non-zero false-positive/invalid-hit floor
below what a purely structured, validate-before-execute approach
achieves. That is the strongest single line in this document, and it's
a synthesis of independent measurements, not one experiment.

## 6. Advanced analysis: why the mechanism produces this specific pattern

Three things worth stating precisely, since "it just works better" isn't
an explanation:

**6.1 — Why hit rate and correctness diverge for response caching, not
just degrade together.** `docs/level3b.md`'s locality sweep found
semantic caching's hit rate *rising* (76.7% → 91.5%) while correctness
*fell* (70.0% → 47.2%) as entity diversity grew — the opposite of naive
intuition. Mechanism: an embedding model's "these are similar" judgment
is dominated by sentence *shape*, and shape is exactly what stays
constant while entity identity varies. As more distinct entities
accumulate in a semantic cache, the population of same-shape,
wrong-entity neighbors grows faster than the population of genuine
repeats, because there are combinatorially more ways to be "shape-similar,
identity-different" than "identical." A structured cache doesn't have
this failure mode because it never asks an embedding model to make an
identity judgment at all — identity comes from exact string extraction,
which has no proximity metric to be fooled by.

**6.2 — Why the stale-fetch degradation curve is exactly `1 − mutation
rate`, not some other shape.** For a mechanism with zero
staleness-detection signal (memoized fetch, or a byte-identical repeat
request against a response cache), correctness after a repeat request
is exactly the probability that nothing changed between the two
requests — there is no partial credit, no graceful degradation
mechanism, because the mechanism has no way to know a change happened at
all. `docs/level3b.md` EXP-012 confirms this is not just a plausible
argument but the measured behavior, predicted before the run and
matched within sampling noise at all 8 points swept.

**6.3 — Why the validation gate is a categorically different safety
property than a similarity threshold.** A semantic cache's threshold is
a *continuous* dial trading recall against precision — the bank's own
case study shows raising it only ever traded one error type for
another (13.4–27.2% FP even after tuning, before the architecture
itself changed). This repo's validation gate is *binary*: either every
required entity/parameter was extracted and is finite, or the request
falls back to the LLM. There's no dial to mistune. The two Level 3
misidentification events (`docs/level3.md` Experiment A) demonstrate
this isn't just a theoretical distinction — a real embedding-similarity
error (a `loyalty_discount` request scored closer to `refund_eligibility`)
was caught by extraction, not by a better-tuned threshold.

## 7. Honest counter-case — what this repo does *not* get to claim

Equal billing with the case above, per this project's standing
practice:

- **The core mechanism is not unprecedented.** Agentic Plan Caching
  (§4) published essentially the same architecture. Two independent
  efforts converging on the same design is evidence the idea is sound —
  it is not evidence this repo invented it, and this repo doesn't claim
  to.
- **"Structured beats semantic" is not a novel empirical claim either.**
  W5H2 measured this independently, on different benchmarks, before
  this repo's Level 3 phase existed (this repo found the paper *after*
  running its own experiments, not before — the results are
  independent, not derived from reading the paper first).
- **This repo's 0.0% invalid-hit rate is not evidence the *mechanism* is
  perfect — it's evidence it's perfect *for the workload class it's
  built for*.** Every result in this document is scoped to tasks that
  are deterministic once their inputs are known (arithmetic, business
  formulas, eligibility/discount/risk rules over fetched records) — see
  `docs/level3.md` RQ9's boundary. Nothing here tests, or claims to
  test, open-ended reasoning, creative generation, or tasks whose
  correct answer isn't a pure function of extractable inputs.
- **The real-world evidence (§3) is illustrative, not a controlled
  comparison.** The bank's numbers come from a different workload,
  different embedding models, and different traffic than this repo's
  own — offered as corroboration that the failure mode is real outside
  a synthetic benchmark, not as a head-to-head benchmark result.
  The LangGraph demo itself was a minimal reproducible script, not run
  through this repo's full benchmark harness the way GPTCache and
  RedisVL were (`docs/prior-art.md`'s ⭐ rows) — that gap is now closed
  differently than originally planned: rather than vendoring the
  `langgraph` library itself into the harness, its confirmed mechanism
  (memoize-by-input-hash with a TTL) was reimplemented as
  `aicache/cache/ttl_cache.py::TTLCache` and driven through a real
  TTL x volatility sweep (Level 4, `docs/level4.md` EXP-014) — a
  reimplementation of a verified mechanism, not the library itself, a
  distinction worth stating plainly rather than letting "we tested
  against LangGraph" imply more than was actually done.
- **Single-run, single-seed results throughout** — the same caveat
  `docs/evaluation.md` states for every phase applies here too; nothing
  in this document should be read as a statistically validated
  estimate rather than an illustration of a reproducible pattern.

## 8. Is it worth the work? A cost/benefit reading of the evidence

The honest cost, measured: computation caching pays a real, non-zero
latency premium to stay safe — `docs/level3.md`'s latency breakdown
shows D's mean fetch cost is 23.17ms per hit, the price of never
trusting stale data. That's not free. But it's small next to what it's
being compared against: a no-cache LLM call costs ~230ms p50 in this
repo's own measurements (`docs/level3.md` EXP-009), and the real bank's
alternative — a non-zero false-positive rate on money-moving actions —
has no latency cost at all, which is exactly why it's dangerous: a wrong
answer returned in 11ms is not obviously wrong to the system serving
it.

The case for "worth it," stated narrowly: for the subset of real agent
traffic that is genuinely deterministic-once-fetched — and the banking,
insurance, and SaaS case studies found during this research (§3,
`docs/prior-art.md`) suggest this subset is large and economically
significant (refund eligibility, discount calculation, risk scoring,
claims triage) — this repo's approach is the only one found, in
production systems or academic literature, that achieves a **zero**
measured false-positive floor rather than a **tuned-down** one. The
bank spent months getting from 99% to 3.8%. This repo's mechanism starts
at 0% for the same class of problem, at the cost of a fetch that would
have to happen at some point regardless (the "no-cache" baseline pays
the same fetch cost plus a full LLM call — see `docs/level3.md`'s
latency breakdown for the actual accounting).

## 9. Bottom line

Every real caching mechanism found in production frameworks or a real
deployment (GPTCache, RedisVL, LangGraph, CrewAI, AutoGen, a real bank's
own semantic cache) caches an answer and accepts a non-zero,
hard-to-eliminate risk of serving the wrong one. The closest published
academic work that avoids this (Agentic Plan Caching, W5H2) validates
the core architecture but doesn't rigorously measure the freshness
question this repo's Level 3/3b phases were built specifically to
answer. On the narrow, honestly-scoped class of tasks this repo
targets, the evidence — this repo's own controlled experiments, a real
production failure, a real production framework's actual code, and the
closest academic literature — points the same direction: separating
"which operation" from "what the answer was," and refusing to cache the
latter, is a real, measurable differentiator, not a restatement of
existing practice with different branding.
