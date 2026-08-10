# Hypotheses

Written before reading results (in spirit — see `docs/experiment-log.md`
for the actual chronology, including several iterations where bugs had
to be found and fixed, and one where the embedding model itself was
swapped mid-project after an explicit comparison, before these could be
evaluated honestly). Each is evaluated against the final experiment run
documented in `docs/experiment-log.md` (EXP-006; 150 requests, 9 task
types, seed 42, simulated LLM backend, `redis/langcache-embed-v1`
embeddings, plus real GPTCache and RedisVL comparison baselines).
Numbers below are from that run; re-running with a different seed or
workload size will shift exact figures without changing the qualitative
pattern, per the repeated-runs caveat in `docs/evaluation.md`.

### H1 — Semantic response caching will outperform exact caching when requests are semantically equivalent.

**Supported.** Exact cache: 0% hit rate on this workload (paraphrased,
parameter-varying requests are never byte-identical). Semantic cache:
64.0% hit rate. Unsurprising, and exactly why semantic caching exists.

### H2 — Semantic response caching will struggle when parameter values materially affect outputs.

**Supported at a loose threshold; largely refuted at a well-chosen
threshold with a strong embedding model — this hypothesis needed a
second variable added to be evaluated honestly.** The original version
of this experiment (see EXP-004 in `docs/experiment-log.md`) used a
generic sentence-similarity model (`all-MiniLM-L6-v2`) and found a 36.0%
invalid-hit rate at threshold 0.75. After switching to
`redis/langcache-embed-v1` — a model purpose-trained for semantic
caching, not a generic-model swap chosen to flatter the result (see
EXP-005) — that same threshold produces only a **0.7% invalid-hit rate**
(99.3% overall correctness) on this workload. The failure mode H2
predicts is real and reproduces reliably, but the *threshold at which it
appears* moved from ~0.75 down to ~0.55–0.60 once a better embedding
model was in use (see the full sweep in EXP-006). **Revised reading**:
H2 is a property of (embedding model × threshold), not of semantic
caching as an approach in the abstract — and the choice of embedding
model turned out to matter as much as, or more than, the threshold
itself. This is echoed almost exactly by real-system results: RedisVL
(same embedding model as this repo's own reimplementation) lands within
noise of this repo's own numbers at every threshold tested (64.0% hit /
0.7% invalid at 0.75, for both); GPTCache (generic SBERT default)
shows 88.7% hit / **70.7% invalid** at the identical threshold — see H2's
extension into cross-system terms below.

### H3 — Computation caching can achieve reuse when computational structure remains stable while parameters change.

**Supported, more strongly than in the earlier (weaker-embedding-model)
run.** The computation cache reached a 77.3% hit rate (up from 50.0% in
EXP-004, once identification threshold + embedding model were
recalibrated in EXP-005) — every hit executed the identified operation
fresh, on freshly bound parameters, for a **0% invalid-hit rate** (100%
correctness on every request, including the 22.7% that fell back to the
LLM).

### H4 — Computation caching can reduce LLM calls without requiring stale response reuse.

**Supported.** 77.3% of requests never touched the LLM backend at all
(zero tokens, zero cost). Total tokens: 346 vs. 2193 for no-cache — an
84.2% reduction, the largest of any condition tested including the
semantic caches. The remaining 22.7% fell back to a full LLM call by
design, whenever identification or binding couldn't be validated.

### H5 — Computation reuse will produce higher benefits on structurally repetitive workloads.

**Not independently tested in this run.** Unchanged from the prior
assessment — see `docs/experiment-log.md`'s "next experiments" list for
the workload-repetition sweep this would require.

### H6 — Computation reuse will fail when task structure changes substantially.

**Supported in a narrower, now well-characterized form.** With
`all-MiniLM-L6-v2` (EXP-004), identification similarity didn't
monotonically separate correct from incorrect task matches at all — a
wrong match could outscore several correct ones. With
`redis/langcache-embed-v1` (EXP-005), that problem shrank substantially
but didn't disappear: across 150 requests, correct-match similarities
ranged 0.405–0.676 while incorrect-match similarities ranged up to
0.514 — a real overlap zone, just a much narrower one. The calibrated
threshold (0.52) sits inside that zone by design, accepting the
trade-off of zero observed misidentifications against a real fallback
cost (see H4). **Revised reading**: this is the same underlying finding
as the H2 revision — embedding model quality is a first-order variable
for this whole approach, not a detail — but computation caching has a
structural advantage semantic response caching doesn't: a
misidentification here is caught downstream by parameter-count/type
validation whenever binding fails, whereas a semantic cache's false hit
is caught by nothing (see `docs/architecture.md`'s validation-as-hard-gate
design note).

### H7 — Hit rate alone will not correlate perfectly with inference-cost savings.

**Supported, and now demonstrable across real systems, not just this
repo's own baselines.** GPTCache shows the *highest* raw hit rate of any
condition tested (88.7%) and the *worst* correctness (29.3%) — reading
hit rate alone would rank it best; reading it with the valid/invalid
split reverses that completely. Even between the two safe conditions,
hit rate ranks them oppositely from a naive "bigger is better" reading:
computation cache (77.3% hit) beats semantic cache/RedisVL (64.0% hit)
on both hit rate and total cost ($0.00061 vs. $0.00120) simultaneously —
consistent, not contradictory, in this specific comparison, but the
GPTCache case shows why hit rate can't be trusted as a sufficient
statistic on its own.

### H8 — Cost-aware retention can outperform frequency-only policies under heterogeneous computation costs.

**Explicitly out of scope for this prototype — future work, not
tested.** See `docs/eviction-future-work.md`. Unchanged.

---

## Addendum: what the GPTCache/RedisVL comparison adds

Not one of the original eight hypotheses, but the finding those two
real-system baselines produced deserves its own statement, since it's
arguably the single most load-bearing result in this repo:

> **On this workload, this repo's own from-scratch semantic-cache
> reimplementation is statistically indistinguishable from RedisVL's
> production implementation at every threshold tested** (hit rate,
> valid-hit rate, and invalid-hit rate all match within noise — see
> `results/cross_system_correctness.png`), **while GPTCache's
> out-of-the-box configuration — a different, generic embedding model,
> not a different caching algorithm — produces a 70-percentage-point
> worse correctness rate at the identical nominal threshold.**

This does two things for the project's credibility that comparing only
against a self-written baseline couldn't: it validates that this
repo's own semantic-cache implementation isn't a strawman built to lose
to the computation cache (it matches a real, widely-used system almost
exactly), and it isolates *embedding model choice* as a variable that
matters as much as — on this workload, more than — the caching algorithm
itself, a finding this project's own earlier research (the sibling
`semantic caching/docker-tests` spike) had already flagged qualitatively
and this run now confirms quantitatively, on a different workload, with
a controlled sweep.

## Addendum: what the real-backend (Ollama) run adds to H3/H4

EXP-008 (`docs/experiment-log.md`) replaced the simulated LLM with a
real local model (`llama3:8b` via Ollama) across all four core
baselines, at full workload scale, with no rate limit to cut the run
short. It reproduced H3/H4's structural finding exactly — the
computation cache's hit path is still 0% invalid, 100% correct — and
then surfaced something none of the simulated-backend runs *could*
have shown:

> **The real model, asked to compute these arithmetic/formula questions
> directly, was only correct 48.7% of the time overall — and 0% of the
> time on `compound_interest` and `margin` specifically (18/18 and
> 12/12 wrong).** Because 77.3% of the computation cache's traffic never
> reaches the model at all, its overall correctness came out to 86.7% —
> using the *identical* underlying LLM as the 48.7%-correct `no_cache`
> baseline.

This reframes H4 ("computation caching can reduce LLM calls without
requiring stale response reuse") with a dimension the original
hypothesis didn't anticipate: reducing LLM calls for deterministic
sub-tasks isn't only a cost/latency optimization, it's also a
correctness hedge against the model's own unreliability at exactly the
kind of task this repo studies. A semantic cache — even a well-built one
using a strong embedding model, even RedisVL itself — still asks the
model to compute the answer on every miss, and inherits whatever
fraction of the time the model gets that computation wrong. A
computation cache's guarantee (correct by construction on a hit) is
unaffected by model quality in a way no response-caching approach,
however good its embedding model, can match. Stated carefully, per
EXP-008's own limitations section: this is one real model, one size, no
prompt engineering, a single un-repeated run — not a claim that LLMs
"can't do arithmetic," only a measured, honest instance of a
well-documented general phenomenon, on this project's own workload.

## Addendum: Level 3 (database-backed enterprise workflows)

H1-H8 above are Level 1/2-specific (pure arithmetic and business
formulas over static, positional parameters) and are left exactly as
originally evaluated. Level 3 tests a structurally different claim —
whether "cache the plan, not the data" survives contact with live,
mutable external state — with its own hypothesis set (L1-L5), evaluated
against EXP-009/EXP-010 rather than EXP-006. Full statements, evidence,
and discussion are in `docs/level3.md`'s "Hypotheses (Level 3)" section
rather than duplicated here; the headline results:

- **L1/L2 supported**: computation-structure reuse generalizes to
  multi-table workflow tasks (100% correctness on every hit, 0%
  invalid), and the freshness benefit specifically requires an actually
  mutating database to demonstrate — a static workload (EXP-009) shows
  the primary condition (D) and its stale-fetch ablation (E) as
  statistically identical; only a workload that mutates mid-run
  (EXP-010) reveals D correctly reflecting each of 3/3 tested mutations
  while both `semantic_cache` and E silently keep serving pre-mutation
  answers, i.e. **0/3 (0%) post-mutation correctness for E — do not cite
  EXP-009's 100% figure for E without this qualifier attached**; see
  `docs/level3.md`'s "The combined, honest three-way view."
- **L3 supported, and sharper than at Level 1/2**: semantic response
  caching's invalid-hit rate is 35.6% here (vs. 0.7% for Level 1/2's
  tuned configuration, H2) — enterprise identifiers carry the
  decision-relevant information but almost none of the semantic weight
  an embedding model responds to, so "same shape, different customer"
  requests collide in embedding space far more than "same shape,
  different number" ones did.
- **L4 supported**: the validation-as-hard-gate mechanism H6 credited
  with catching Level 1/2 misidentifications fired on two genuine,
  unforced misidentification events in the Level 3 run (0.9% of
  requests) — not just a theoretical backstop.
- **L5 supported**: per-stage latency decomposition (identify / extract
  / validate / fetch / execute) shows fetch time, not identification
  time, dominates a Level 3 hit — the opposite balance from Level 1/2,
  where there was no fetch step at all.

## Addendum: Level 3b (structural locality and state volatility sweeps)

EXP-009/EXP-010 established the Level 3 mechanism at a single workload
shape and two mutation-exposure endpoints. EXP-011/EXP-012
(`docs/level3b.md`) swept both variables directly — see that document's
hypotheses M1-M3 for full statements. Headline: computation-cache hit
rate and correctness are flat across a 40x range of entity-combination
diversity, while semantic caching's correctness *falls* as its raw hit
rate *rises* over that same range (more distinct entities grow the
false-neighbor pool faster than the true-repeat pool) — a sharper
version of H7 than the original single-workload comparison could show.
Correctness under mutation exposure follows `1 − mutation_rate`
precisely for both semantic caching and the stale-fetch ablation, while
the always-fresh condition (D) stays flat at 100% at every rate tested.
