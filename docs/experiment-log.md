# Experiment Log

All runs use the deterministic `simulated` LLM backend (see
`aicache/llm/simulated.py`) — no real API calls, no real money spent.
Every number below is reproducible: `python -m benchmarks.run_experiment
--seed <seed>` regenerates the same workload and (given the same code)
the same outcome distribution. Failed/superseded experiments are kept
here, not deleted, per this project's own standard for what "honest"
means.

**Environment**: Python 3.11.8, macOS (Darwin 25.4.0), CPU only. No GPU
used anywhere in this repo. Dependencies pinned via `pyproject.toml` (not
version-locked to exact patch versions as of this writing — a
reproducibility gap flagged in `docs/architecture.md`-adjacent notes,
not hidden).

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` through
EXP-004; `redis/langcache-embed-v1` from EXP-005 onward (see that
entry for why, and for the earlier-model results this project's own
docs referenced before the switch — kept in EXP-001–EXP-004 rather than
retroactively edited).

**External comparison baselines** (EXP-006 onward): real GPTCache
(local FAISS+SQLite) and real RedisVL (`redis/redis-stack` via Docker,
`localhost:6379`), each run in its own dedicated virtualenv as a
subprocess — see `benchmarks/external/` and its module docstrings for
why they're kept out-of-process (dependency isolation) and how a run
degrades (skips with a printed reason, doesn't fail) when either isn't
available.

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

---

## EXP-005 — Embedding model switch: `all-MiniLM-L6-v2` → `redis/langcache-embed-v1`

**Date**: 2026-08-10
**Motivation**: user request, prompted by two things surfacing together —
(a) discovering a working local copy of `redis/langcache-embed-v1` was
already validated in this environment (the sibling `semantic
caching/docker-tests` project's own `RESULTS.md` flagged it as "a model
Redis trained specifically for semantic caching," and had already solved
the setup friction for it), and (b) a direct side-by-side check before
committing to the switch.

**Calibration check**: cosine similarity for a genuine paraphrase pair
("What is 10 plus 20?" / "Calculate the sum of 10 and 20.") —
`all-MiniLM-L6-v2`: in the 0.3–0.5 range (see EXP-002/EXP-004's H6
discussion); `redis/langcache-embed-v1`: **0.768** on the same pair.

**Re-calibration required in two places**, both done empirically rather
than by guessing new numbers:

1. **Task-identification threshold** (`TaskIdentifier`, used by the
   computation cache). Across 150 generated requests: correct-match
   similarities ranged 0.405–0.676; incorrect-match similarities (9
   observed misidentifications, e.g. `percentage` confused with
   `compound_interest`/`multiply`/`tax`/`margin` — "percent" apparently
   a confusable token across several formula phrasings) ranged
   0.434–0.514. A threshold sweep found **0.52** gives 0 wrong matches
   accepted at 82.3% recall of correct matches, the best zero-error
   point found. Old default (0.60, tuned for the previous model) would
   have been far too strict for this model's similarity scale.
2. **Semantic-cache threshold sweep range**. Re-ran the same
   hit-rate/invalid-rate calibration used to pick EXP-004's sweep range:
   at threshold 0.55, invalid-hit rate is 52.7% (the dangerous region
   still exists!); by 0.70–0.75 it's down to 0.7–2.0%; by 0.80+ it's
   0%. The whole "dangerous" region shifted from ~0.80–0.85 (original
   MiniLM tuning) down to ~0.55–0.65 — the sweep range in
   `benchmarks/run_experiment.py` was moved to match
   (`[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]`), not arbitrarily
   widened.

**Regression fallout** (expected, and useful): 3 of 48 existing tests
failed after the switch — two in `tests/test_computation_cache.py`
because a specific test phrase ("What is 512 plus 937?") happened to
score 0.517, just under the new 0.52 threshold (swapped for an
unambiguous phrase scoring 0.563); one in
`tests/test_end_to_end.py::test_semantic_cache_hits_are_not_all_correct`
because threshold 0.75 — dangerous under the old model — is now safe
under the new one, so the test's own assumption needed updating to a
threshold (0.60) that's still in the new dangerous region. All three
fixes are updates to *what threshold demonstrates the phenomenon*, not
changes to what phenomenon is being tested — see each test's updated
docstring for the specific reasoning.

**Interpretation**: this is the single highest-leverage change made in
this project's history — see H2/H6 in `docs/hypotheses.md` for how much
the qualitative story shifted, not just the numbers. It's also a
reminder that this project's own earlier "semantic caching is unsafe
under parameter variation" framing (EXP-004) was more specific than it
first appeared: it was true for one embedding model's default behavior,
and the room for a better model to substantially close that gap was
real, not hypothetical.

**Next**: re-run the full benchmark under the new model; separately,
add real external-system comparison baselines (GPTCache, RedisVL) to
check this repo's own semantic-cache reimplementation against production
systems rather than only against itself.

---

## EXP-006 — Real GPTCache and RedisVL comparison baselines

**Date**: 2026-08-10
**Hypothesis under test**: none of H1–H8 directly — this tests whether
this repo's own semantic-cache reimplementation is a fair stand-in for
real semantic-caching systems, and separately, how much embedding-model
choice (as opposed to caching-algorithm implementation) explains the
semantic cache's correctness behavior.

**Setup**: reused the sibling `semantic caching/docker-tests` project's
already-solved environments — a dedicated `.venv-gptcache` (GPTCache
0.1.44, local FAISS+SQLite, generic `all-MiniLM-L6-v2` embedding — its
own out-of-the-box default, deliberately left unchanged, not forced to
match this repo's embedding choice) and `.venv-redisvl` (RedisVL 0.25.1
against a `redis/redis-stack` Docker container already running on
`localhost:6379`, defaulting — confirmed by direct inspection, not
assumed — to `redis/langcache-embed-v1`, the same embedding model this
repo's own semantic cache now uses). Both run as subprocesses
(`benchmarks/external/runner_gptcache.py`,
`runner_redisvl.py`) communicating via a shared JSON protocol
(`benchmarks/external/protocol.py`) — kept out-of-process specifically
so neither system's dependency tree (torch/onnxruntime/faiss-cpu for
GPTCache; the `redis` client for RedisVL) risks conflicting with this
repo's own environment. On a cache miss, both runners store the exact
same precomputed (simulated-LLM, ground-truth-correct) answer text this
repo's own semantic-cache baseline would have stored, so all three
systems are replaying the identical protocol over the identical
workload.

**Configuration**: same 150-request workload as EXP-004/005 (seed 42),
same threshold sweep range as the recalibrated semantic-cache sweep
(`[0.55, ..., 0.90]`), with RedisVL's distance-threshold convention
converted via `distance = 1 - similarity` so all three systems' sweeps
are aligned on the same nominal scale.

**Observed** (headline row = threshold 0.75; full sweep in
`results/results.csv` and charted in
`results/cross_system_correctness.png`):

| System | Embedding model | Hit rate | Valid | Invalid | Correctness | Tokens | Cost (USD) |
|---|---|---|---|---|---|---|---|
| This repo's semantic cache | `langcache-embed-v1` | 64.0% | 63.3% | 0.7% | 99.3% | 778 | 0.00120 |
| RedisVL | `langcache-embed-v1` (its own default) | 64.0% | 63.3% | 0.7% | 99.3% | 778 | 0.00120 |
| GPTCache | `all-MiniLM-L6-v2` (its own default) | 88.7% | 18.0% | **70.7%** | **29.3%** | 194 | 0.00033 |

**This repo's semantic cache and RedisVL match to the number of
requests** at every point in the threshold sweep, not just at 0.75 —
see the sweep table in `results/results.csv` (`semantic_cache@*` vs.
`redisvl@*` rows). GPTCache's own sweep (`gptcache@*` rows) shows
correctness climbing from 2.0% (threshold 0.55) to 74.7% (threshold
0.90) — still worse than semantic_cache/RedisVL's *worst* tested point
(47.3% at their most dangerous threshold, 0.55) until GPTCache's
threshold is pushed well past 0.90.

**Interpretation**:

- **This repo's own semantic-cache reimplementation is not a strawman.**
  Matching a real, independently-implemented, widely-used system
  (RedisVL) within noise at every threshold is strong evidence the
  earlier baselines (and by extension the computation-cache comparison
  against them) aren't artifacts of a weak or unusually bad reference
  implementation.
- **Embedding model choice is at least as load-bearing as caching
  algorithm choice, for this workload.** Two systems built on similar
  architecture (embed the request, nearest-neighbor search, threshold
  cutoff) diverge by up to ~70 percentage points of correctness at the
  identical nominal threshold, purely because of which embedding model
  each defaults to. This corroborates — with a controlled sweep, on a
  different workload — the qualitative finding the sibling
  `docker-tests` project's `RESULTS.md` first flagged.
- **This sharpens, rather than undercuts, the case for computation
  caching.** A good embedding model can make semantic response caching
  *safe in practice* at the right threshold — but "safe in practice, at
  a threshold someone has to find and defend" is a materially weaker
  correctness guarantee than "correct by construction, because nothing
  is ever reused from a prior response" (the computation cache's actual
  property, unaffected by embedding-model choice since its identification
  failures are caught by parameter validation, not left to hope). Both
  can be true at once: semantic caching-with-a-good-embedding-model is a
  credible engineering choice for many workloads, and computation
  caching offers a stronger guarantee where the task structure permits
  it.

**Next experiments**: real-backend validation (Gemini) remains the
top item — see EXP-004's list, item 3, now updated to note a
`GEMINI_API_KEY` was provided by the user; results pending a dedicated
run (see `docs/experiment-log.md`'s next update, or `results/` for
whether one has landed by the time this is read). Workload-repetition
sweep (H5) and Level 3 prototype remain unstarted, per EXP-004's
original list.

---

## EXP-007 — Real-backend (Gemini) spot-check: a correctness-checker bug found, a hard quota wall hit

**Date**: 2026-08-10
**Goal**: the "real-backend validation" item queued since EXP-004 — confirm
this repo's correctness-checking and cache-outcome logic behave sensibly
against real model output, not just the simulated backend's own
ground-truth-by-construction answers.

**What happened, in order:**

1. A single manual call to `gemini-2.5-flash` succeeded — correct answer,
   real token counts, ~1.4s latency.
2. Four more manual calls, using this project's actual (non-instructed)
   workload phrasings — e.g. "What is 512 plus 937?", no "answer with
   just the number" hint — all returned **correct final answers wrapped
   in worked explanations**: `"To find the sum of 512 and 937, we add
   them:\n\n512\n+ 937\n-----\n1449\n\nSo, 512 plus 937 is **1449**."`
3. **Bug found**: `aicache/metrics/correctness.py`'s `is_correct` parsed
   the *first* number in the response text (`parse_first_number`). On
   response (2) above, that extracts `512` — an input operand — not the
   answer. Every one of the four real responses would have been graded
   incorrect despite being correct. This bug was invisible through
   EXP-001–EXP-006 because the simulated LLM always answers with a
   single bare number (`"1449"`, no surrounding text), so "first number"
   and "last number" were always the same value — the bug had zero
   opportunity to manifest until real model output hit it.
4. **Fix**: renamed to `parse_final_number`, changed to take the *last*
   extracted number instead of the first. Re-verified correct against
   all four real responses (including a compound-interest case with a
   worked multi-line derivation ending in "**$1469.33**"). Re-ran the
   full 51-test suite — all pass unchanged, confirming zero behavioral
   change for the simulated backend (the whole point of picking "last"
   over "first" as the more robust heuristic, not a simulated-vs-real
   special case).
5. Also, while in `aicache/llm/gemini.py`: disabled Gemini's extended
   "thinking" mode (`thinkingConfig.thinkingBudget: 0`) — pure
   arithmetic/formula questions get no accuracy benefit from it, and a
   single throwaway call showed it consuming 312 "thinking" tokens
   against 10 visible-answer tokens, inflating latency, cost, and
   rate-limit pressure for nothing. Added client-side request pacing
   (minimum 4s between requests) and retry-with-backoff for 429/503.
6. **Attempted a scoped, quota-conscious validation run** (36 requests,
   then 9 requests, `--skip-threshold-sweep --skip-external`) — both
   failed with HTTP 429 even with pacing and backoff in place.
   Diagnosed directly: the error body reports
   `quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"`,
   `quotaValue: "20"` — a **hard daily cap of 20 requests/day** for
   `gemini-2.5-flash` on this key's project, not a per-minute burst
   limit pacing could work around. Checked whether other models on the
   same key had separate headroom (`gemini-2.5-flash-lite` — deprecated/
   404 for new users; `gemini-2.0-flash`, `gemini-2.0-flash-lite` — both
   also 429, same daily-exhaustion pattern). Stopped immediately rather
   than continuing to spend retry attempts against a hard cap that
   retries cannot fix — each retry is itself a request that counts
   against the same scarce daily allowance.

**Interpretation**: the real-backend spot-check did exactly the job
"validation" is supposed to do — it found a real bug
(first-vs-last-number extraction) that every prior simulated-backend run
in this repo could never have surfaced, because the simulated backend's
output shape happened to make the bug's two code paths identical. That's
the general argument for real-backend validation this project's docs
made before ever running one: a simulated backend, however carefully
built, can only be as revealing as its own assumptions about output
shape — and this project's own simulated LLM's assumption ("answers are
bare numbers") turned out to be doing more work than intended, quietly
protecting a bug in the harness itself, not just standing in for model
quality. The daily-quota wall is a mundane, expected constraint (flagged
explicitly when the Gemini backend was first built, and again before
this run) — not a finding about the caching hypotheses, just the reason
a full real-backend run of EXP-006's scale doesn't exist yet.

**Next**: retry a real-backend validation run once the daily quota
resets (or with a paid/higher-tier key) — same scoped approach (small
workload, `--skip-threshold-sweep --skip-external`, now with correct
answer parsing and request pacing already in place, so the only
remaining constraint is quota, not code). A single day's 20-request
allowance is enough for roughly the 9-request configuration attempted
here (no_cache + exact_cache + a handful of semantic/computation-cache
misses/fallbacks) but not for anything close to EXP-006's 150-request,
multi-threshold scale — a full real-backend replication of EXP-006
would need either a paid tier or accumulating results across several
days' free quota.

---

## EXP-008 — Full-scale real-backend validation (Ollama, local, no quota) — a bigger finding than expected

**Date**: 2026-08-10
**Motivation**: EXP-007 established the harness works correctly against a
real model but couldn't run at scale — blocked on Gemini's free-tier
daily cap. Ollama, run fully locally (this environment: a MacBook with
an Apple M4 Pro, 16-core CPU / 20-core GPU), has no such ceiling. Added
`aicache/llm/ollama.py` (same `LLMClient` interface as the Gemini
backend; local REST to `http://localhost:11434/api/generate`, no auth,
real token counts reported by Ollama itself rather than a heuristic).
Model used: `llama3` (8B, already present locally — no new pull
needed), default Ollama sampling settings (not pinned to
temperature 0 — see Limitations below).

**A pricing bug caught before this run could mislabel its results**:
while wiring up the Ollama backend, a source read of
`aicache/llm/pricing.py::estimate_cost_usd` found it fell back to
Claude Haiku's rate card for *any* unrecognized model name — meaning a
real Gemini or Ollama result would have been silently priced as if it
were a Claude Haiku call. Fixed to return `$0.00` for models with no
pricing-table entry instead (an honest "not priced," not a wrong
number) before this run — so every `cost_usd` value in this experiment's
results is `0.0`, correctly reflecting that neither model has an entry
in a Claude-only pricing table, not that inference was free. For Ollama
specifically, `$0.00` means real local compute time was spent, not
literal zero cost.

**Configuration**: full default workload (150 requests, 9 task types,
seed 42) through all four core baselines —
`python -m benchmarks.run_experiment --backend ollama --skip-threshold-sweep
--skip-external --output-dir results/real_backend_ollama` (the
`--output-dir` flag was added *because* the first attempt at this run
didn't have it yet and silently overwrote EXP-006's committed
`results/results.csv` / `summary.json` — recovered from git history
before anything was lost; the flag exists so that mistake can't repeat).
Threshold sweep and GPTCache/RedisVL comparison were skipped
deliberately — both are about caching-algorithm/embedding behavior,
already validated exhaustively in EXP-006, and don't depend on which
LLM answers a miss.

**Observed** (full breakdown in `results/real_backend_ollama/`):

| Baseline | Hit rate | Correctness |
|---|---|---|
| A — No cache | 0.0% | 48.7% |
| B — Exact cache | 0.0% | 46.7% |
| C — Semantic cache (0.75) | 64.0% | 54.0% |
| D — Computation cache | 77.3% | **86.7%** |

![Correctness by baseline, real backend](../results/real_backend_ollama/correctness_real_backend.png)

**The headline number isn't the cache comparison — it's `no_cache`'s
48.7%.** Calling a real local 8B model directly, once per request, with
no tool use and no chain-of-thought scaffolding, got these
closed-form arithmetic/formula questions right less than half the time.
Broken down by task type (from the `no_cache` condition specifically —
this isolates the model's own raw accuracy from any caching behavior):

| Task | Accuracy |
|---|---|
| `add` | 100% |
| `subtract` | 94% |
| `revenue` | 56% |
| `percentage` | 44% |
| `multiply` | 44% |
| `tax` | 42% |
| `discounted_price` | 39% |
| `compound_interest` | **0%** |
| `margin` | **0%** |

![Real LLM accuracy by task type](../results/real_backend_ollama/llm_accuracy_by_task_type.png)

Single-digit-friendly addition and subtraction are essentially solved;
anything requiring more than one arithmetic step (a percentage of a
number, a discount then a subtraction, an exponent for compound
interest) degrades fast, and multi-step compound interest and margin
calculations were wrong on **every single one of 18 and 12 attempts
respectively.** This is a well-known, general limitation of LLMs doing
raw arithmetic without a tool (not a novel finding about language
models in general) — the value of measuring it *here* is that it's
this project's own workload, this project's own harness, a real
number instead of received wisdom.

**Why this matters for the cache comparison — a mechanism this
project hadn't identified before this run:**

Breaking down `semantic_cache` by outcome:

| Outcome | n | Correct |
|---|---|---|
| `hit_valid` | 52 | 100% (by definition) |
| `hit_invalid` | 44 | 0% (by definition) |
| `miss` (fresh LLM call) | 54 | **53.7%** |

Even when the cache correctly decided "this is not a safe reuse, ask the
model" (a `miss`), the resulting answer was only right 53.7% of the
time — because that's just the underlying model's own accuracy on this
workload, independent of anything cache-related. **A real semantic
cache's overall correctness is therefore a mix of two unrelated failure
sources**: false-hit reuse (the mechanism EXP-004–006 measured, still
present here — 44 invalid hits out of 96 total hits, 45.8%, in the same
range as the simulated-backend sweep at this threshold) *and* the base
model simply being wrong on a fresh call. The simulated backend
(EXP-001–006) couldn't show this second source at all, because its
"model" was correct by construction on every miss.

The same decomposition explains `computation_cache`'s 86.7%:

| Outcome | n | Correct |
|---|---|---|
| `hit_valid` | 116 | 100% |
| `hit_invalid` | 0 | — |
| `fallback` (LLM call) | 34 | 41.2% |

**The hit path is still exactly what EXP-004–006 found: 0% invalid, 100%
correct, unchanged by which model answers a fallback** — this is the
one number in this whole run that's a pure replication, and it
replicated exactly. What's new is what happens on `no_cache`'s 48.7%
baseline once you route 77.3% of that same traffic around the
unreliable model entirely: `0.773×100% + 0.227×41.2% ≈ 86.7%`, matching
the observed figure. **Computation caching's real-world benefit, on
this run, isn't just "avoids the semantic cache's stale-reuse errors" —
it's "avoids asking an unreliable model to do arithmetic at all, for
most of its traffic," which is a categorically different and arguably
larger benefit than anything the simulated-backend experiments could
surface**, because they gave every condition a perfectly-accurate LLM
by construction.

**An observed anomaly, reported rather than explained away**:
`exact_cache` took 574.85s wall-clock for 150 calls (~3.83s/call) versus
`no_cache`'s 287.97s (~1.92s/call) for the identical workload run
immediately before it — nearly 2× slower for a condition that should
behave almost identically to `no_cache` (near-zero hit rate expected and
observed). No root cause was investigated (thermal throttling under
sustained local GPU load and background system activity are plausible;
neither was measured). Flagged as a real limitation of a single,
uncontrolled local run — see Limitations below — not smoothed over.

**Limitations specific to this run**:

- **Single run, no repeated trials** — same caveat as every other
  experiment in this log, but sharper here: Ollama's default sampling
  isn't pinned to temperature 0, so the *same prompt* can yield a
  *different* answer on a different call (unlike the simulated backend,
  which is fully deterministic). The 48.7%/86.7%/etc. figures are one
  draw from a real distribution, not a fixed ground truth.
- **One model, one size** — `llama3` (8B). A larger model, a model
  fine-tuned or prompted for arithmetic, or a model given a code-execution
  tool would very plausibly score far higher on `no_cache` — this run
  says something true about *this* model configuration, not about LLMs'
  arithmetic ability in general.
- **No prompt engineering attempted** — the workload's phrasings are
  the same natural-language questions used throughout this project;
  no "think step by step" or "use a calculator" instruction was added.
  A production system would very likely add exactly this kind of
  scaffolding, which would change `no_cache`'s baseline and therefore
  the gap this experiment measured.
- **The unexplained 2× latency anomaly** noted above.

**Interpretation, stated carefully**: this result does not mean
"semantic caching is bad" or "computation caching solves LLM math" in
general — a better-prompted or larger real model would likely close
much of the `no_cache` gap on its own. What it does show, on this
project's own workload with this specific real local model: routing
deterministic sub-tasks around the LLM entirely is not merely a
cache-correctness optimization, it's also a hedge against the model's
own unreliability at exactly the kind of task this repo studies — and
that hedge is unavailable to any approach (semantic caching included)
that still asks the model to compute an answer on every miss.

**Next**: a repeated-trials version of this run (several seeds, or
several draws per request) would turn "48.7%" into a distribution
rather than a point estimate — listed as follow-up, not attempted here.
Also worth trying: the same run with a prompt that asks the model to
show its work step-by-step (a cheap, realistic mitigation) to see how
much of the `no_cache` gap is closable without a computation cache at
all.

---

## EXP-009 — Level 3 primary comparison: database-backed enterprise workflows

**Date**: 2026-08-10
**Hypothesis under test**: L1-L5 (`docs/level3.md`) — does the "cache
the plan, not the data" principle established at Level 1/2 generalize
to workflows that must read live, multi-table enterprise state before
answering, and does semantic response caching's false-hit problem get
worse at higher stakes.

**Configuration**: `python -m benchmarks.level3.run_experiment`
(defaults: `--backend simulated --seed 99 --db-seed 42 --n-customers 80
--n-transactions 200 --combinations-per-task 15
--paraphrases-per-combination 3 --semantic-threshold 0.65`) — 225
requests across 5 task families (`refund_eligibility`,
`loyalty_discount`, `shipping_cost`, `customer_spend_summary`,
`transaction_risk`), each of the 5 baselines run against its own
`db.clone()` of an identical starting database. New machinery this run
depends on, all built and tested in this phase: `aicache/enterprise/db.py`
(mutable live database), `aicache/computation/workflow_tasks.py` (5
task specs), `aicache/binding/entity_extractor.py` (per-slot regex
extraction), `aicache/cache/workflow_computation.py` (the D/E pipeline
with 5-stage latency decomposition), `benchmarks/level3/baselines.py`
and `run_experiment.py`.

**Observed**:

| Baseline | Hit% | Valid% | Invalid% | Fallback% | Correct% | Tokens | Cost (USD) | p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| A — No cache | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 7,504 | 0.02398 | 231.5 |
| B — Exact cache | 3.6% | 3.6% | 0.0% | 0.0% | 100.0% | 7,220 | 0.02302 | 228.1 |
| C — Semantic cache (0.65) | 84.0% | 48.4% | 35.6% | 0.0% | 64.4% | 1,210 | 0.00387 | 11.0 |
| D — Computation cache (fresh) | 99.1% | 99.1% | 0.0% | 0.9% | 100.0% | 77 | 0.00025 | 37.4 |
| E — Stale-fetch ablation | 99.1% | 99.1% | 0.0%* | 0.9% | 100.0%* | 77 | 0.00025 | 12.1 |

\*Identical to D in this run only, because this workload doesn't mutate
the database mid-run by design. **Do not cite E's 100% correctness from
this table on its own — it holds only at 0% mutation exposure and is
not evidence E is a safe strategy; see EXP-010, where the identical
baseline scores 0/3 under mutation.**

![Hit rate by baseline](../results/level3/hit_rate_by_baseline.png)
![Latency breakdown, D vs E](../results/level3/latency_breakdown_d_vs_e.png)

Full discussion, including the two genuine `loyalty_discount` →
`refund_eligibility` misidentifications the validation gate caught, and
why semantic caching's invalid-hit rate (35.6%) is dramatically worse
here than the tuned Level 1/2 figure (0.7%, `docs/hypotheses.md` H2), is
in `docs/level3.md`'s Results section rather than duplicated here.

**A bug found and fixed before this run's numbers could be trusted**:
`aicache/enterprise/db.py`'s fetch methods originally returned live
references into the stored tables, not copies — meaning the E ablation
wasn't actually stale (a later mutation would silently change data E
had already "cached"). Caught by a targeted smoke test before any
experiment used the fix. See `docs/level3.md`'s "Bugs found and fixed"
for this and two other bugs (a workload-realism distribution issue, a
renderer/parser field-ordering mismatch) found and fixed during this
phase.

**Interpretation**: the four qualitative findings from Level 1/2
(computation caching's 0% invalid-hit rate, semantic caching's
real-but-tunable false-hit problem, hit rate's unreliability as a
standalone correctness proxy, validation-as-hard-gate catching real
misidentifications) all reproduce at Level 3, on a structurally
different workload family with live external state — not a given in
advance, since Level 3 exercises code paths (fetch, entity extraction,
multi-field structured output) Level 1/2 never touched at all.

**Next**: EXP-010, the dedicated database-mutation experiment — this
run alone cannot show the freshness benefit the whole phase exists to
test, only that D and E behave identically absent a mutation.

---

## EXP-010 — Level 3 freshness experiment: request, mutate, repeat

**Date**: 2026-08-10
**Hypothesis under test**: L2 (`docs/level3.md`) — computation caching's
"always re-fetch" design should produce a correct, updated answer after
a database mutation; both semantic response caching and a memoized-fetch
ablation should not, since neither has any mechanism to notice the
underlying data changed.

**Configuration**: `python -m benchmarks.level3.freshness_experiment` —
three hand-picked, independently-inspectable scenarios (not a
statistical sample; see `docs/level3.md`'s "Freshness model" for why
this experiment is deliberately small), each following the same
protocol: issue a request, mutate the database, issue the identical
request again, compare `semantic_cache`, D (fresh), and E (stale
ablation) against independently-recomputed ground truth at both points
in time.

**A bug in this experiment's own script, found and fixed before its
results could be trusted**: the first run built the simulated LLM's
`task_hint` from the ground-truth *result* dict (e.g. `eligible=True`)
instead of the raw *fetched* data the task executor actually takes as
keyword arguments (e.g. `transaction={...}, policy={...}`), so every
`semantic_cache` call errored (`_execute_refund_eligibility() got an
unexpected keyword argument 'eligible'`) rather than producing a real
answer. Fixed by fetching and executing as two separate steps and
passing the fetched fields, not the executor's own output, into the
hint. A bug in this phase's measurement tooling, not in any module
under test.

**Observed** (`results/level3/freshness_experiment.json` has the full
record):

| Scenario | Mutation | `semantic_cache` after | D (fresh) after | E (stale ablation) after |
|---|---|---|---|---|
| `refund_eligibility_after_status_change` | txn `completed` → `refunded` | "Yes... $500.00" — wrong | "No... (transaction_not_completed)" — correct | "Yes... $500.00" — wrong |
| `loyalty_discount_after_tier_upgrade` | tier `bronze` → `gold` | "0% discount... $400.00" — wrong | "10% discount... $360.00" — correct | "0% discount... $400.00" — wrong |
| `refund_eligibility_after_policy_change` | gold refund window `60` → `30` days | "Yes... $300.00" — wrong | "No... (outside_refund_window)" — correct | "Yes... $300.00" — wrong |

3/3 scenarios: ground truth genuinely changed after the mutation
(verified independently, not read from any baseline under test); D was
the only condition whose served answer changed with it. `semantic_cache`
and E both kept serving the exact pre-mutation text, with no signal to
the caller that anything was stale.

**Aggregate post-mutation correctness** (added after this experiment's
first pass, in response to the review below —
`benchmarks/level3/freshness_experiment.py::_summarize_post_mutation_correctness`,
written to `freshness_experiment.json`'s `summary` key):

| Baseline | Post-mutation correctness |
|---|---|
| `computation_cache_fresh` | 3/3 (100%) |
| `computation_cache_stale_ablation` | 0/3 (0%) |
| `semantic_cache` | 0/3 (0%) |

**⚠️ This number and EXP-009's aggregate correctness for
`computation_cache_stale_ablation` (100%) are NOT the same claim and
must never be quoted interchangeably** — EXP-009 measures 0% mutation
exposure, this measures 100% mutation exposure (every scenario mutates
the queried record). A baseline's correctness here is a function of how
much mutation exposure it faced, not a fixed property of the baseline.
An earlier pass of this documentation reported EXP-009's number as if
it were the whole picture; see `docs/level3.md`'s "Bugs found and
fixed" item 5 and its combined three-way view
(`results/level3/correctness_latency_tradeoff.png`) for the fix.

**Interpretation**: this is the result the entire Level 3 phase was
built to produce — see `docs/level3.md`'s "Core architectural
principle" section. EXP-009 alone could not have shown this (its
workload never mutates mid-run); this experiment's whole design exists
to isolate exactly the property EXP-009's static workload structurally
cannot exercise.

**Limitations**: three scenarios, one mutation each, not a statistical
sample — see `docs/level3.md`'s Limitations section. Doesn't estimate
*how often* staleness would matter in a real request mix (100% mutation
exposure is an extreme, not an estimate of a realistic rate), only that
the mechanism exists and behaves as designed when it fires.

**Next**: a mutation-*rate* sweep (vs. this experiment's fixed 100%
exposure) and a structural-locality sweep (vs. EXP-009's fixed high
task-type repetition) are both proposed, not attempted, in
`docs/level3.md`'s "Next: structural locality and state volatility"
section.
