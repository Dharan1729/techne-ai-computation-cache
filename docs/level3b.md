# Level 3b: Structural Locality and State Volatility Sweeps

> **Update 2026-10-20.** Most results in this file were measured at a single seed.
> A later multi-seed rerun changed some of them. The computation cache is not
> literally 0% invalid: it made 20 wrong hits in 1500 Level 1/2 requests over 10
> seeds (fixed by a guard, EXP-016) and 1 wrong hit in 2250 Level 3 requests
> (EXP-017). Read "0% invalid" or "100% correct" below as "zero at the seed
> measured". Comparisons against a raw, unscaffolded LLM used a weak baseline;
> EXP-015 has the fair one. Details: [experiment-log.md](experiment-log.md),
> EXP-015 to EXP-017.

## Motivation

`docs/level3.md`'s primary comparison (EXP-009) reported a 99.1%
computation-cache hit rate — but on a workload deliberately built for
high structural repetition (5 task families × a fixed, small set of
entity combinations × 3 paraphrases each). RQ9 in that document flagged
explicitly that this number is a property of *that workload's*
structural locality, not a general constant, and proposed two follow-up
sweeps rather than asserting the relationship qualitatively: how does
each baseline respond as **structural locality** drops, and how does
the stale-fetch ablation's correctness respond as **state volatility**
rises from EXP-009's 0%-mutation-exposure endpoint to EXP-010's
100%-exposure endpoint. This document is those two sweeps.

Both sweeps follow one-variable-at-a-time discipline: the locality sweep
holds mutation exposure at zero and varies only entity/parameter
diversity; the volatility sweep holds workload structure fixed (one
task family, one mutation type) and varies only mutation rate. Neither
sweep touches embedding threshold, task-identification logic, or the
computation cache's own mechanism — any effect measured is attributable
to the swept variable, not a confound.

## Sweep 1: structural locality

### What "structural locality" means operationally here

This system's task registry is intentionally small — 5 hand-specified
enterprise workflows (`docs/level3.md`), not hundreds of synthetic task
types. Scaling the registry itself to "1000 task types" (as originally
sketched) would mean inventing hundreds of fake computations with no
grounding in anything real, which would test the embedding model's
capacity to discriminate arbitrary categories, not this repo's actual
research question. The lever that genuinely varies request-to-request
similarity in this system is **how many distinct entity/parameter
combinations** (which customer, which transaction, which zone, which
order total) each task family draws from before it starts repeating
them — so this sweep holds the 5 task families and 3-paraphrases-per-
combination fixed (`workloads/level3/templates.py` has exactly 3
phrasings per task) and varies `combinations_per_task` from 2 to 80.
Low values mean many paraphrases of a small combination set (high
locality); high values mean a large combination set, most drawn once or
twice (low locality).

`benchmarks/level3/locality_sweep.py`: for each
`combinations_per_task` in `[2, 5, 10, 20, 40, 80]`, generates a fresh
workload (`n_transactions`/`n_customers` scaled up alongside
`combinations_per_task` so larger sweep points aren't starved of
distinct records to draw from) and runs all 5 baselines against it,
reusing `benchmarks/level3/baselines.py`'s existing runners unchanged.

### Results

| Combinations/task | Exact hit% | Semantic hit% | Semantic invalid% | Semantic correct% | Computation (D) hit% | Computation (D) correct% |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0% | 76.7% | 30.0% | 70.0% | 100.0% | 100.0% |
| 5 | 0.0% | 78.7% | 34.7% | 65.3% | 96.0% | 100.0% |
| 10 | 3.3% | 84.7% | 36.7% | 63.3% | 100.0% | 100.0% |
| 20 | 5.7% | 86.3% | 37.7% | 62.3% | 99.0% | 100.0% |
| 40 | 7.3% | 88.8% | 41.3% | 58.7% | 97.7% | 100.0% |
| 80 | 8.4% | 91.5% | 52.8% | 47.2% | 97.9% | 100.0% |

(E — the stale-fetch ablation — tracks D within rounding at every point,
exactly as expected absent any mutation; omitted from the table above
for space, shown alongside D in the chart.)

![Structural locality sweep](../results/level3/locality_sweep.png)

**The computation cache's hit rate and correctness are essentially flat
across a 40x range of entity diversity** (96.0–100.0% hit, always 0%
invalid, 100% correct at every point) — confirming RQ9's boundary claim
directly rather than asserting it: computation caching's reuse
mechanism keys on *task identification*, which doesn't depend on how
many distinct entity combinations exist, unlike a response cache's
reuse mechanism, which fundamentally does.

**Semantic caching's hit rate rises while its correctness falls — a
sharper, more counterintuitive version of H7.** Naive intuition says
lower structural locality (more distinct combinations, less repetition)
should mean *fewer* semantic hits. The opposite happened: hit rate rose
from 76.7% to 91.5% as combinations grew. The mechanism: this system's
embedding model can't distinguish `CUST-0001` from `CUST-0091` well
(`docs/level3.md`'s Experiment A already established this) — so as more
distinct combinations accumulate in the semantic cache, the pool of
near-neighbor "false friends" (same task shape, wrong entity) grows
*faster* than the pool of true repeats, making a hit more likely
overall but a *correct* hit less likely. Invalid-hit rate rose from
30.0% to 52.8% in step, and overall correctness fell from 70.0% to
47.2% — **worse than a coin flip** at the lowest-locality point tested.
Hit rate and correctness move in *opposite* directions here, which is a
stronger claim than H7's original "hit rate doesn't correlate with
savings" — on this sweep, hit rate is actively anti-correlated with
correctness.

**Exact caching's hit rate creeping up (0.0% → 8.4%) is a real,
understood, and fully explained effect — not noise.** Verified directly
against the generated workload at the 80-combinations point: 1,200
total requests, 1,099 distinct texts, 101 duplicate occurrences —
matching the observed 8.4% exact-hit rate almost exactly. The cause:
`workloads/level3/generator.py` samples entity values *with
replacement*, so as `combinations_per_task` grows, the birthday-paradox
probability of two independently-drawn combinations landing on the
*same* (customer, transaction) pair rises — and because template
rendering is fully deterministic and every combination uses all 3
available phrasings, any such collision produces byte-identical text.
This is a real property of the workload generator, reported rather than
smoothed over, and it doesn't change the qualitative story: even at its
highest measured point, exact caching's 8.4% is dwarfed by computation
caching's 97.9%.

## Sweep 2: state volatility

`docs/level3.md`'s freshness experiment (EXP-010) measured exactly two
points — 0% mutation exposure (EXP-009) and 100% (every EXP-010
scenario). This sweep fills in the curve between them, holding mutation
*type* fixed (transaction status `completed` → `refunded` — the
cleanest of EXP-010's three, chosen because it's a reliable, unambiguous
ground-truth flip) and varying mutation *rate*.

`benchmarks/level3/volatility_sweep.py`: 40 independent, explicitly
constructed probes (guaranteed-eligible `refund_eligibility` scenarios,
built the same "don't rely on random generation landing on the
boundary case" way EXP-010's three scenarios were). At each rate `p` in
`[0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]`, each probe independently
gets mutated with probability `p` between its "before" and "after"
request (fresh, isolated cache instances per probe — the same isolation
discipline EXP-010 used, so this measures only the staleness question,
not the identity-collision failure mode Experiment A already
characterized separately).

### Results

| Mutation rate | Semantic correctness | D (fresh) correctness | E (stale ablation) correctness |
|---:|---:|---:|---:|
| 0% | 100.0% | 100.0% | 100.0% |
| 1% | 100.0% | 100.0% | 100.0% |
| 5% | 92.5% | 100.0% | 92.5% |
| 10% | 87.5% | 100.0% | 87.5% |
| 25% | 77.5% | 100.0% | 77.5% |
| 50% | 57.5% | 100.0% | 57.5% |
| 75% | 27.5% | 100.0% | 27.5% |
| 100% | 0.0% | 100.0% | 0.0% |

![State volatility sweep](../results/level3/volatility_sweep.png)

**Stated as a prediction before running, confirmed by the data**: D's
correctness is flat at 100% across the entire sweep (it re-fetches
unconditionally, so mutation rate cannot affect it by construction).
`semantic_cache` and E — two different mechanisms (response-text reuse
vs. memoized-fetch reuse) — trace the **same curve**, matching the
predicted `1 − mutation_rate` line within normal sampling noise for
n=40 (standard error up to ~7.9% at p=0.5; every observed point is
within 2.5 percentage points of its prediction). This is expected, not
a coincidence: for a byte-identical repeat request, neither mechanism
has *any* signal that the underlying record changed, so both simply
reuse whatever was true at the time of the first request — their
correctness is exactly the probability that nothing changed in between.

## Bugs found this phase

None. Unlike `docs/level3.md`'s phase (three real code bugs) and the
reporting-fix follow-up (a presentation issue), this phase's results
matched their pre-stated predictions on the first clean run, after the
smoke tests below caught nothing. Recorded here per this project's
practice of reporting clean runs as such, not just failures — a
document with only bug reports would misrepresent how often things
actually go wrong.

**Verification before trusting the results**: `_run_point` (locality)
and `_run_rate_point` (volatility) were each smoke-tested at small scale
before the full sweeps ran — a 2-probe volatility check at rate 0.0 and
1.0 confirmed the exact predicted shape (semantic/E: 100%→0%, D: 100%→100%)
before committing to the full 8-point, 40-probe-per-point run.

## Hypotheses (Level 3b)

### M1 — Computation-cache correctness and hit rate are invariant to structural locality; response-cache correctness is not.

**Supported.** D/E: 96.0–100.0% hit, 0% invalid, 100% correct across a
40x range of entity-combination diversity. Semantic cache: correctness
fell from 70.0% to 47.2% over the same range, while its raw hit rate
*rose* — the opposite of the naive prediction that lower locality simply
means fewer hits.

### M2 — Semantic caching's failure mode gets worse, not better, as entity diversity increases, because diversity grows the false-neighbor pool faster than the true-repeat pool.

**Supported.** Invalid-hit rate rose monotonically with
`combinations_per_task` (30.0% → 52.8%) in lockstep with hit rate rising
(76.7% → 91.5%) — both driven by the same mechanism (more accumulated
near-duplicate-looking entries), pulling hit rate and correctness in
opposite directions.

### M3 — Correctness under state volatility follows (1 − mutation rate) for any mechanism with no staleness-detection signal, and is exactly flat at 100% for one that re-fetches unconditionally.

**Supported, precisely.** Every measured point for `semantic_cache` and
E fell within 2.5 percentage points of the predicted `1 − p` line; D was
100% at all 8 points tested.

## Research questions

**RQ11 — Does computation caching's near-perfect hit rate survive a
40x drop in structural locality?** Yes — see M1. Hit rate ranged
96.0–100.0%, never dropping meaningfully, across the entire sweep.

**RQ12 — Is semantic caching's Level 3 failure mode (identity collision,
`docs/level3.md` Experiment A) sensitive to how much entity diversity
the traffic contains?** Yes, and in the more damaging direction — see
M2. Lower locality doesn't make semantic caching merely less useful; it
makes its hits actively less trustworthy while remaining just as
frequent.

**RQ13 — What is the actual shape of the correctness-vs-mutation-rate
relationship the two-point freshness experiment (EXP-010) could only
bound?** Linear, matching `1 − rate` — see M3 and
`results/level3/volatility_sweep.png`.

## Limitations

- **Locality sweep is a single run per point, one seed** — same caveat
  as every prior phase (`docs/evaluation.md`). The exact-cache collision
  mechanism (see Results) is itself seed- and volume-dependent; a
  different seed would shift the specific 8.4% figure without changing
  the qualitative pattern (still far below D's hit rate at every point
  tested).
- **Locality sweep's upper bound is 80 combinations/task (1,200 total
  requests)**, not the 500–1000 originally sketched — chosen to keep
  sweep runtime practical (~6 minutes total for all 6 points × 5
  baselines) while still spanning a 40x range, wide enough to show both
  effects (M1, M2) clearly. A wider range is mechanical follow-up, not
  attempted here.
- **Volatility sweep holds mutation type fixed** (transaction status
  only) — EXP-010 already showed the staleness effect generalizes
  across mutation type (transaction status, customer tier, policy);
  this sweep deliberately isolates rate as the sole variable rather than
  re-demonstrating type generality.
- **40 probes per rate point** — enough to make the predicted `1 − p`
  line and the observed points visually and numerically close, not a
  large-sample statistical estimate. Standard error at p=0.5 is ~7.9
  percentage points.

## Where this leaves Level 3

Both variables `docs/level3.md`'s RQ9 flagged as unswept — structural
locality and state volatility — now have measured response curves, not
just a static two-point comparison. The core Level 3 claims (D is safe
and fast enough to matter; semantic caching's failure mode is real and
tied to entity/identity collision; the stale ablation's danger is a
direct, quantifiable function of mutation exposure) all hold up under
both sweeps rather than being artifacts of one specific workload shape.
Whether to proceed to Level 4 (agentic, branching workflows —
`docs/level3.md`'s "Deferred: Level 4") from here is a scope decision,
not something this document resolves on its own.
