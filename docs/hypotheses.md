# Hypotheses

Written before reading results (in spirit — see `docs/experiment-log.md`
for the actual chronology, including one iteration where a workload bug
had to be found and fixed before these could be evaluated honestly).
Each is evaluated against the single experiment run documented in
`docs/experiment-log.md` (workload: 150 requests, 9 task types, seed 42,
simulated backend). Numbers below are from that run; re-running with a
different seed or workload size will shift exact figures without
changing the qualitative pattern, per the repeated-runs caveat in
`docs/experiment-log.md`.

### H1 — Semantic response caching will outperform exact caching when requests are semantically equivalent.

**Supported.** Exact cache: 0% hit rate on this workload (paraphrased,
parameter-varying requests are never byte-identical). Semantic cache
(threshold 0.75): 72.7% hit rate. This is the expected, unsurprising
result that motivates semantic caching in the first place.

### H2 — Semantic response caching will struggle when parameter values materially affect outputs.

**Supported, more starkly than expected.** Of the semantic cache's 72.7%
hit rate at threshold 0.75, only 36.7 percentage points were valid hits —
36.0 percentage points were invalid hits (a reused answer that doesn't
match the current request's own ground truth). Overall correctness on
this baseline was 64.0%, i.e. more than a third of all requests received
a wrong answer, silently, with no error signal distinguishing a wrong
answer from a right one at the API-response level.

### H3 — Computation caching can achieve reuse when computational structure remains stable while parameters change.

**Supported.** The computation cache reached a 50.0% hit rate — every
hit executed the identified operation fresh, on freshly bound
parameters, for a **0% invalid-hit rate** (100% correctness on every
request, including the fallback half). This is the central positive
result: reuse without correctness risk on this workload.

### H4 — Computation caching can reduce LLM calls without requiring stale response reuse.

**Supported.** 50% of requests never touched the LLM backend at all
(zero tokens, zero cost, sub-millisecond latency on top of a real,
measured embedding-lookup + parameter-extraction + execution cost). The
other 50% fell back to a full LLM call — by design, whenever
identification or binding couldn't be validated, rather than guessing.

### H5 — Computation reuse will produce higher benefits on structurally repetitive workloads.

**Not independently tested in this run.** The workload here already has
maximal structural repetition by construction (9 fixed task types, many
parameter draws and paraphrases each) — there's no lower-repetition
condition in this experiment to compare against. Testing this properly
needs a workload-repetition sweep (e.g. vary the number of distinct task
types per fixed request count), which is listed as a next experiment in
`docs/experiment-log.md` rather than fabricated here.

### H6 — Computation reuse will fail when task structure changes substantially.

**Supported, and observed in a form worth flagging.** The task
*identification* step itself has a real, measured failure mode
independent of "structure changing": for terse arithmetic phrasings,
absolute cosine similarity against canonical examples runs
surprisingly low (roughly 0.3–0.5 for genuinely-correct matches with
`sentence-transformers/all-MiniLM-L6-v2`), and in at least one observed
case a *different* task's phrasing ("subtract") scored higher against
the wrong canonical task ("add", similarity 0.457) than several
*correct* matches scored against their own right task (`multiply` at
0.308, `add` at 0.453). This means similarity magnitude does not
monotonically separate correct from incorrect identification for this
embedding model on this workload — a genuine limitation, not a tuning
inconvenience, documented in `docs/experiment-log.md`'s failure-analysis
section. The threshold chosen for the headline run (0.60) was picked
specifically because it happened to reject the one wrong-match instance
observed in a small sample, at the cost of also rejecting several
correct low-similarity matches (hence the 50% fallback rate) — an
explicit, acknowledged trade-off, not a tuned "best" value.

### H7 — Hit rate alone will not correlate perfectly with inference-cost savings.

**Supported by construction and confirmed in the numbers.** The
semantic cache's 72.7% hit rate is *higher* than the computation cache's
50.0% hit rate, yet the computation cache saves more total cost (tokens:
749 vs. 2193 for no-cache — a computation-cache reduction of ~66%,
vs. semantic cache's reduction to 527 tokens, ~76% token reduction but
attached to a correctness rate of only 64%). Reading hit rate alone
would rank semantic caching as more effective; reading hit rate together
with the valid/invalid split and correctness rate reverses that
conclusion for any use case where wrong answers have a cost.

### H8 — Cost-aware retention can outperform frequency-only policies under heterogeneous computation costs.

**Explicitly out of scope for this prototype — future work, not
tested.** See `docs/eviction-future-work.md`. This prototype's caches
have unbounded, unevicting memory; no admission/eviction policy of any
kind is implemented or compared. Do not read anything in this repo's
results as evidence for or against H8.
