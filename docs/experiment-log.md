# Experiment Log

All runs use the deterministic `simulated` LLM backend (see
`aicache/llm/simulated.py`) — no real API calls, no real money spent.
Every number below is reproducible: `python -m benchmarks.run_experiment
--seed <seed>` regenerates the same workload and (given the same code)
the same outcome distribution. Failed/superseded experiments are kept
here, not deleted, per this project's own standard for what "honest"
means.

**Environment**: Python 3.11.8, macOS (Darwin 25.4.0), CPU only.
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. No GPU used
anywhere in this repo. Dependencies pinned via `pyproject.toml` (not
version-locked to exact patch versions as of this writing — a
reproducibility gap flagged in `docs/architecture.md`-adjacent notes,
not hidden).

---

## EXP-001 — Smoke test surfaces a correctness-tolerance bug

**Date**: 2026-08-10
**Hypothesis under test**: none yet — this was a plumbing smoke test
(`--param-sets-per-task 1 --paraphrases-per-param-set 1`, 9 requests) to
confirm all four baselines run end to end before trusting any real
result.

**Observed**: `no_cache`, `exact_cache`, and `semantic_cache` all showed
88.9% correctness (8/9), not the expected 100% — `no_cache` calls the
simulated LLM every time, and the simulated LLM computes ground truth
directly from the same task registry the workload generator uses, so a
mismatch here could only be a comparison bug, not a real computation
error.

**Root cause**: `aicache/metrics/correctness.py`'s `is_correct` used a
1e-6 tolerance, but the simulated LLM formats non-integer answers to 4
decimal places (`f"{result:.4f}"`). For the `margin` task specifically
(`(revenue-cost)/revenue`, ranges up to 1,000,000), the rounding error
from 4-decimal formatting (~2.35e-5 in the observed case) exceeded the
1e-6 tolerance.

**Fix**: widened tolerance to 1e-3 (relative and absolute) — loose
enough to tolerate reasonable answer rounding by any backend (simulated
or real), tight enough that reusing a genuinely different parameter
draw's answer (the actual failure mode this benchmark exists to detect)
still reliably fails the check. See the tolerance rationale documented
inline in `aicache/metrics/correctness.py`.

**Interpretation**: a real, if narrow, bug — the kind that would have
quietly deflated every baseline's reported correctness by a fixed,
misleading amount. Caught by the smoke test precisely because it ran
100% of the way through the pipeline on a tiny workload before scaling
up, rather than debugging a partial run at full scale.

**Next**: re-run the smoke test; if clean, scale up.

---

## EXP-002 — Full-scale run surfaces a template-ordering bug in `computation_cache`

**Date**: 2026-08-10
**Hypothesis under test**: H3/H4 (`docs/hypotheses.md`) — computation
cache hits should be 100% correct by construction, since each hit
freshly executes on freshly bound parameters rather than reusing a
stored response.

**Configuration**: default workload (`--param-sets-per-task 6
--paraphrases-per-param-set 3`, seed 42, 150 requests, 9 task types).

**Observed**: `computation_cache` showed a 55.6% hit rate but only 50.0
percentage points valid — 5.3 percentage points **invalid** hits (a
"hit," by this project's own design, should never be invalid, since
nothing is reused from a stored response). 7 of 8 invalid hits were
`compound_interest`; the rest `discounted_price`.

**Root cause**: two paraphrase templates in `workloads/templates.py`
stated their placeholder numbers in a different left-to-right order than
the task's declared parameter order — e.g. `"What will ${0} grow to
after {2} years at {1} percent compounded annually?"` presents
*principal, years, rate* in the text, but `compound_interest`'s
parameters are declared as *principal, rate, years*. Since
`aicache/binding/extractor.py` extracts numbers **positionally** (a
documented, explicit limitation — see that module's docstring), this
template silently bound the rate value into the years slot and vice
versa, producing a wrong (but successfully "validated" — both are
finite numbers) computation. This is the exact "wrong task
classification / under-generalization" failure category the brief calls
out in its failure-analysis requirements, just at the binding stage
rather than the identification stage. A parallel bug in
`discounted_price`'s second template did the same thing.

**Fix**: rewrote both templates to preserve the declared parameter
order. `revenue`'s second template had the same *shape* of bug but
never produced a wrong answer in practice, because `price × quantity` is
commutative — fixed anyway, on the principle that being correct by
construction is better than being correct by accident of the specific
formula's algebra.

**Interpretation**: confirms this repo's own documented limitation
(positional extraction) is a real, not theoretical, risk — and confirms
it's the kind of bug that specifically evades a "does the answer look
plausible" check, since a swapped rate/years pair still produces a
plausible-looking number. Re-affirms why validation (§ correctness
checking against independent ground truth, not just "did it parse") is
load-bearing, not decorative, in this benchmark's own harness — the same
principle the brief asks the computation cache itself to follow.

**Next**: add a regression test asserting every workload item's raw
numbers extract in the template's declared order, so this class of bug
can't reappear silently; re-run.

---

## EXP-003 — Regression test catches a second, un-triaged instance of EXP-002's bug class

**Date**: 2026-08-10
**Hypothesis under test**: none — writing the regression test suite
itself (`tests/test_workload_generator.py::test_text_numbers_extract_in_declared_param_order`,
added directly in response to EXP-002).

**Observed**: the new test failed immediately, on `subtract`, which
EXP-002's manual invalid-hit inspection hadn't flagged (subtract's
template bug happened not to produce any invalid hits in that specific
150-request run — plausibly because those particular `subtract`
paraphrases fell below the identification threshold and fell back to
the LLM before binding was ever attempted, masking the bug rather than
avoiding it).

**Root cause**: same class as EXP-002 — two `subtract` templates
("Subtract {1} from {0}." and a "take {1} away from {0}" variant) stated
the operands in reverse order. Unlike `compound_interest`, subtraction
is not commutative (`x - y ≠ y - x`), so this was a live, not merely
latent, correctness bug — it simply hadn't surfaced yet in observed
output.

**Fix**: rewrote both templates.

**Interpretation**: this is the actual argument for the regression test
existing at all — EXP-002's bug was caught by *inspecting a run's
invalid hits*, which only works if the buggy template actually produces
an invalid hit in that specific run (a matter of luck: whether task
identification happens to succeed for that phrasing, whether the swap
happens to change the answer). The regression test checks the *property*
directly (extracted order matches declared order) regardless of whether
any particular run happens to exercise it, and would have caught both
bugs in one pass had it existed first.

**Next**: full test suite, then a clean final run.

---

## EXP-004 — Final headline run (all 48 tests passing, both template bugs fixed)

**Date**: 2026-08-10
**Hypothesis under test**: H1–H4, H6, H7 (see `docs/hypotheses.md` for
full writeups; summarized here).

**Configuration**: `python -m benchmarks.run_experiment` — default
workload (150 requests, 9 task types, 6 parameter draws × up to 3
paraphrases each, seed 42), simulated backend
(`claude-haiku-4-5` pricing snapshot for cost math, see
`aicache/llm/pricing.py`), semantic-cache threshold sweep at
{0.80, 0.85, 0.90, 0.92, 0.95, 0.97} plus a headline run at 0.75.

**Observed** (full table in `results/summary.json` /
`results/results.csv`; charts in `results/*.png`):

| Baseline | Hit rate | Valid hit | Invalid hit | Correctness | Tokens | Cost (USD) | p50 latency |
|---|---|---|---|---|---|---|---|
| A — No cache | 0.0% | — | — | 100.0% | 2193 | 0.00338 | 206.9 ms |
| B — Exact cache | 0.0% | 0.0% | 0.0% | 100.0% | 2193 | 0.00338 | 206.9 ms |
| C — Semantic cache (0.75) | 72.7% | 36.7% | 36.0% | 64.0% | 527 | 0.00085 | 5.4 ms |
| D — Computation cache | 50.0% | 50.0% | 0.0% | 100.0% | 749 | 0.00134 | 78.6 ms |

Semantic-cache threshold sweep (hit rate / invalid-hit rate /
correctness): 0.80 → 59.3% / 25.3% / 74.7%; 0.85 → 44.7% / 19.3% / 80.7%;
0.90 → 26.0% / 14.0% / 86.0%; 0.92 → 12.0% / 8.0% / 92.0%; 0.95 → 4.7% /
0.7% / 99.3%; 0.97 → 3.3% / 0.0% / 100.0%.

**Interpretation**:

- Exact caching does essentially nothing on this workload (RQ1) — as
  expected, since requests are paraphrased and parameterized, almost
  never byte-identical.
- Semantic caching adds real reuse (RQ2: 72.7% hit rate) but roughly
  half of that reuse is *wrong* (RQ3/H2) — this is the central
  motivating failure mode, measured directly rather than argued for.
- No single similarity threshold buys both good hit rate and good
  correctness simultaneously on this workload: correctness only
  approaches 100% once the hit rate has collapsed to ~3–5% (threshold
  0.95+), at which point semantic caching is barely doing anything.
  There is no threshold in the swept range that Pareto-dominates the
  computation cache's (50% hit, 100% correct) point — every semantic
  threshold either has a lower hit rate *and* lower correctness, or a
  higher hit rate *and* materially lower correctness.
- The computation cache reaches 50% hit rate at 100% correctness and a
  ~66% token reduction vs. no-cache — the central positive result
  (H3/H4).
- Hit rate alone would rank semantic caching (72.7%) above computation
  caching (50.0%) — reading it together with the valid/invalid split
  reverses that conclusion for any use case where a wrong answer has
  real cost (H7).
- The computation cache's 50% *fallback* rate (not a failure — a
  deliberate, correctness-preserving refusal to guess) traces to a
  specific, honestly-reported limitation: task-identification
  similarity for terse arithmetic phrasings runs low enough
  (~0.3–0.5) against this embedding model that a threshold high enough
  to reject one observed wrong-task confusion (0.457) also rejects
  several genuinely-correct low-similarity matches (see H6 in
  `docs/hypotheses.md` for the specific numbers).

**Next experiments** (not run in this pass — see also
`docs/eviction-future-work.md` for the eviction-specific deferral):

1. **Workload-repetition sweep** (H5) — vary the number of distinct task
   types at a fixed total request count, to test whether computation
   reuse's advantage scales with structural repetition as predicted.
2. **Task-identification threshold sweep** — mirror the semantic-cache
   threshold sweep for the computation cache's identification step, to
   characterize the fallback-rate/false-identification trade-off
   quantitatively instead of the single fixed threshold (0.60) used
   here.
3. **Real-backend validation** — re-run a reduced-size version of this
   experiment against a real Gemini backend (`--backend gemini`, see
   `aicache/llm/gemini.py`) to confirm the simulated backend's
   qualitative findings hold with a real model in the loop. Not run in
   this pass because it requires the user to provision and fund a
   `GEMINI_API_KEY`, a decision explicitly left to them (see repo
   history / README).
4. **Level 3 (DB-backed) prototype** — the natural next scope expansion,
   requiring a different computation representation (see
   `docs/architecture.md`'s scope note) rather than a parameter change
   to this one.
5. **Hierarchical / combined baseline** — a computation-cache-fallback
   that checks a semantic response cache before calling the LLM (brief's
   condition G), to test whether the two mechanisms compose or conflict.
