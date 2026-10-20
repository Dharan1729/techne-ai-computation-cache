# Prior Art

> **Update 2026-10-20.** Most results in this file were measured at a single seed.
> A later multi-seed rerun changed some of them. The computation cache is not
> literally 0% invalid: it made 20 wrong hits in 1500 Level 1/2 requests over 10
> seeds (fixed by a guard, EXP-016) and 1 wrong hit in 2250 Level 3 requests
> (EXP-017). Read "0% invalid" or "100% correct" below as "zero at the seed
> measured". Comparisons against a raw, unscaffolded LLM used a weak baseline;
> EXP-015 has the fair one. Details: [experiment-log.md](experiment-log.md),
> EXP-015 to EXP-017.

Purpose: map the existing design space this repo sits inside, not to
adjudicate "who did it first." Every system/pattern below is established;
nothing here claims to improve on any of them in general — only to test
one specific combination (computation-structure reuse + parameter
rebinding + validation) against one specific, narrow workload family.
Where a year or detail below isn't independently verified against the
project's own source, it's marked accordingly rather than stated as fact.

| System / pattern | Cache object | Matching mechanism | Parameter-aware? | Plan/program reuse | Cost-aware policy | Eviction | Freshness handling | Agent support | KV/prefix support | Notes / how this repo relates |
|---|---|---|---|---|---|---|---|---|---|---|
| **Redis (general)** | Arbitrary key→value | Exact key match | N/A (app-defined) | No | No (app-defined TTL/LRU via `maxmemory-policy`) | LRU/LFU/TTL/random, configurable | App-defined TTL | No | No | The substrate most exact and semantic caches are built on top of, including this repo's exact-cache baseline in spirit (in-process dict here, not Redis itself, for zero infra dependency). |
| **Redis LangCache** (managed cloud product) | LLM prompt→response pairs | Semantic (embedding similarity) + exact | No — matches on prompt similarity only | No | Not published in detail as of this writing | Not a focus of the product surface | TTL-based | No | No | The commercial, hosted product — distinct from RedisVL (below), the open-source client library this repo actually benchmarks. Not run in this project; no claims made about it beyond public documentation. |
| **RedisVL `SemanticCache`** ⭐ directly benchmarked | LLM prompt→response pairs, in a real Redis instance | Semantic (embedding similarity via vector search), configurable distance threshold | No — matches on prompt similarity only, same as this repo's own semantic-cache baseline | No | No | Not a focus of this extension | TTL supported (not exercised here) | No | No | **Run directly** as a comparison baseline (`benchmarks/external/redisvl_adapter.py`, real Redis via Docker) — not just described from documentation. On this project's 150-request workload, RedisVL's results matched this repo's own from-scratch semantic-cache reimplementation within noise at every threshold tested (both default to the same embedding model, `redis/langcache-embed-v1` — RedisVL's own out-of-the-box default, confirmed by direct inspection, not assumed). See `docs/experiment-log.md` EXP-006 and `docs/hypotheses.md`'s addendum for what this validates. |
| **GPTCache** ⭐ directly benchmarked | LLM prompt→response pairs | Configurable embedding + vector-store backend, configurable similarity threshold | No | No | No | Configurable (backend-dependent) | TTL | Partial (framework hooks) | No | **Run directly** as a comparison baseline (`benchmarks/external/gptcache_adapter.py`, local FAISS+SQLite, GPTCache's own out-of-the-box default embedding — generic `sentence-transformers/all-MiniLM-L6-v2`, deliberately left unchanged rather than forced to match this repo's own choice). On the same workload and nominal threshold (0.75) RedisVL and this repo's own semantic cache used, GPTCache showed a 70.7% invalid-hit rate vs. their 0.7% — a controlled demonstration that embedding-model choice, not caching-algorithm implementation, explains most of the gap (see EXP-006). An earlier, separate spike (not included in this repository) had benchmarked GPTCache with the same embedding model on a different, short single-sentence workload and found a similar recall-cliff-versus-false-positive pattern; this run is independent, on this project's own arithmetic/formula workload, not a reuse of that earlier result. |
| **vLLM prefix caching (PagedAttention / RadixAttention family)** | KV-cache blocks (attention state), not text | Exact prefix match on tokenized input | N/A — operates below the request-text layer | No | No | Reference-counted block eviction | N/A (attention state, not a "response") | Runs inside any workload, agentic or not | Yes — this *is* KV/prefix caching | A different layer entirely: reuses compute *inside* one inference engine, not across distinct requests at the application layer. This repo's computation cache and this are not competitors — a downstream inference engine's own prefix cache can still fire on top of anything this repo does, if the assembled prompt happens to share a prefix. Out of scope for this repo's benchmark (no real inference engine is exercised). |
| **LMCache** | KV-cache blocks, extended across GPU/CPU/disk tiers and across engine instances | Exact prefix / chunk match | N/A | No | Partial (tiering policy) | Tiered eviction | N/A | Runs inside any workload | Yes, extends vLLM's model to a distributed, multi-tier cache | Same layer as vLLM prefix caching, extended for distributed serving. Not evaluated here for the same reason. |
| **GenCache** *(name as commonly used in recent literature; exact paper/system not independently verified for this doc)* | LLM response, workload-aware | Semantic + learned admission signal | Partially — some variants incorporate usage/cost signals | No | Yes, in variants that use predicted reuse value | Learned/adaptive | TTL or usage-based | Partial | No | Represents the general direction of "workload-aware" semantic cache replacement research this repo's §19/§20 (deferred eviction work) is scoped toward, not a system this repo directly benchmarks against. |
| **Agentic Plan Caching** — [Zhang, Wornow, Wan and Olukotun, "Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents" (arXiv:2506.14852)](https://arxiv.org/abs/2506.14852) | Extracted plan/tool-call templates | Structured plan-template matching + a lightweight adapter model for task-specific context | Yes — templates are adapted with task-specific parameters (their example: fiscal year, company name) at reuse time | Yes — the closest published sibling to this repo's baseline D | Not the paper's focus | Not the paper's focus | Not addressed: the words stale, fresh and up-to-date do not appear in the current version (v2) of the paper | Yes, by construction | No | **The closest prior-art system found to this repo's core mechanism**, identified via literature search after this repo's Level 3 phase, not before — confirms the same core idea (cache the plan, re-fetch the data) independently. Figures from the paper itself (v2, January 2026): its abstract reports 50.31% lower cost and 27.28% lower latency on average while maintaining performance; in the body, a 46% cache hit rate on 100 FinanceBench queries and a 76.42% cost reduction with a 0.61-point accuracy drop on GAIA. These measure different things on different workloads from this repo's hit rates, so they are not compared here. **What this repo adds beyond this paper**: a controlled ablation isolating staleness as the sole variable (baseline E, sharing one class with D via a single flag) and a mutation-rate sweep quantifying exactly how reuse-without-refetch degrades (`docs/level3b.md` EXP-012) — the paper does not discuss data freshness, so it has no comparable controlled freshness experiment. |
| **W5H2 structured intent canonicalization** — [arXiv:2602.18922](https://arxiv.org/html/2602.18922) | Canonicalized (What, Where) action-target pair as cache key; parameters (Who/When/HowMuch) injected separately at execution | Structured field extraction (7-field decomposition) into a canonicalized key, not raw embedding similarity | Yes — this is the paper's central mechanism, very close to this repo's identify-then-extract split | No — classification/routing, not full plan execution | No | No | Not addressed — the paper does not discuss live data freshness or cache invalidation (no mention of stale, fresh or invalidation in the current version) | N/A | No | Independently arrived at a near-identical split to this repo's `TaskIdentifier` (coarse, embedding-based "which known operation") + `entity_extractor.py` (exact, regex-based "which parameters") design. Their own numbers: 91.1% vs. GPTCache's 37.9% accuracy on one benchmark, 55.3% vs. 49.1% for its GPTCache-KMeans variant on another — an independent, real confirmation of this repo's own H2/H6 (`docs/hypotheses.md`) and Level 3 Experiment A findings that structured extraction beats semantic similarity for this failure mode. Does not address freshness at all, which is where this repo's Level 3/3b work goes further. |
| **LangGraph `CachePolicy`** — [reference.langchain.com](https://reference.langchain.com/python/langgraph/types/CachePolicy) | Node output, keyed by a pickle-hash of the node's input | Exact match on hashed input (memoization), with an optional TTL | No — caches whatever the node returned for that exact input, including any live data the node fetched | No — caches one node's output, not a multi-step plan structure | No | TTL-based expiry only | **TTL only — no re-fetch-on-reuse, no live-data awareness**; a cache hit within the TTL window returns the exact same output regardless of whether underlying data changed | Yes — this is a first-party feature of the most widely used agent-graph framework | No | Directly installed and exercised with a minimal reproducible script confirming its mechanism (see `docs/differentiation.md` §3) — architecturally equivalent to this repo's baseline **E** at Level 3 (the stale-fetch ablation), not baseline D. `aicache/cache/ttl_cache.py::TTLCache`, modeled directly on this confirmed mechanism, is now the basis of Level 4's baseline **G** (`docs/level4.md`), which quantifies — across a TTL x volatility sweep, not just a single before/after demo — exactly how correctness degrades as a function of both TTL length and how fast the underlying data changes (EXP-014). LangChain's own GitHub has an open issue (`langchain-ai/docs#3736`) asking to document "when and why to use the cache parameter," and no source found during this research (including this issue) quantifies that relationship the way EXP-014 now does. |
| **CrewAI tool caching** | Tool call result, gated by a user-supplied `cache_function(args, result) -> bool` | Exact match on tool args (implied; cache-key composition not fully documented) | Partial — the `cache_function` can inspect args, but the framework doesn't distinguish "live data" args from "structural" args | No — per-tool, not plan-level | No | Not documented | Not addressed in available documentation — no TTL, no re-fetch signal, invalidation left entirely to the user's `cache_function` | Yes, first-party | No | Described from documentation only, not run. Notable for putting the "should this be cached" decision in user code (`cache_function`) rather than the framework — closer in spirit to this repo's validation gate than LangGraph's blanket TTL, but the decision is made *after* the tool already ran (deciding whether to *store* the result), not *before* execution the way this repo's validate-before-execute gate works, and nothing in the documentation found addresses live-data freshness specifically. |
| **AutoGen / AG2 LLM caching** | Full LLM response, keyed by exact request (disk or Redis-backed) | Exact match only | No | No | No | Not documented | Not addressed — positioned for cost/reproducibility, not freshness | Yes, first-party | No | Described from documentation only, not run. Architecturally equivalent to this repo's baseline B (exact cache) — same structural inadequacy for parameterized traffic that `docs/hypotheses.md` H1 and this repo's own exact-cache results (e.g. 3.6% hit rate in `docs/level3.md` EXP-009) already demonstrate. |
| **MiniCache** | Compressed KV-cache across transformer layers (inter-layer redundancy) | Layer-similarity based merging, not request-level | N/A | No | No | N/A | N/A | N/A | Partial (works on KV representations) | Different problem (model-internal memory compression) at a different layer than anything this repo touches. Listed for completeness of the design-space map, not because it's comparable. |
| **Helium** *(name as commonly used; not independently verified against a specific paper for this doc)* | Varies by context the name is used in | — | — | — | — | — | — | — | — | Included per the brief's prior-art list; this author could not independently verify a single canonical "Helium" caching system at the time of writing, so no specific claims are made about it here — flagged rather than fabricated. |
| **Google PARROT** *(research paper on LLM serving with semantic variable annotations)* | Structured prompt "variables" the serving system tracks explicitly | Application-annotated data flow, not black-box similarity | Yes — this is close to the point: PARROT's semantic variables let the serving system distinguish "the fixed instruction" from "the variable content" | Partially — annotations describe data flow more than reusable procedures | Serving-level optimizations (batching/scheduling), not cache economics per se | Not a cache-eviction paper in the sense of §19 | N/A | Yes | Interacts with prefix caching at the serving layer | The prior-art system conceptually closest to this repo's core distinction (structure vs. parameters), from a different angle (serving-system-level prompt annotation vs. application-level task identification + closed-form re-execution). This repo's approach is much narrower in scope (deterministic arithmetic/formula tasks only) and doesn't require any serving-system integration. |
| **TinyLFU / W-TinyLFU** | Cache admission/eviction policy (object-agnostic) | Frequency-sketch-based admission, not similarity | N/A | N/A | Frequency-aware, not directly cost-aware | Yes — this is the core contribution | N/A | N/A | N/A | Directly relevant to `docs/eviction-future-work.md` §19–21, not to anything implemented in this prototype. TinyLFU-style admission is one of several policies named there as future work; nothing here reimplements or benchmarks it. |
| **CacheLib** | General-purpose cache library (object-agnostic) | Pluggable | N/A | N/A | Pluggable (supports cost-aware policies as a library feature) | Pluggable (LRU/LFU/TinyLFU-style/etc.) | App-defined | N/A | N/A | Infrastructure-layer prior art for `docs/eviction-future-work.md`, not evaluated here. |

## Real-world evidence: a banking semantic-cache deployment

[Reducing False Positives in RAG Semantic Caching: a Banking Case Study (InfoQ)](https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/)
documents a financial-services organization's real production attempt at
exactly this repo's baseline C (semantic response caching), on banking
FAQ/request traffic. Not run by this repo — read and summarized from the
published article, cited as external evidence, not reproduced.

Documented failures, with confidence scores: *"I don't want this card
anymore"* → matched to **investment account closure** (88.7%
confidence); *"I don't want this business account anymore"* → matched
to **automatic payment cancellation** (84.9%); an ATM-location query →
matched to loan-balance instructions (80.9%). The same class of failure
this repo's own semantic-cache baseline demonstrates (`docs/level3.md`
Experiment A, 35.6% invalid-hit rate) — different domain, same
mechanism (same-shape, different-intent/entity collision in embedding
space).

Their measured false-positive rates across four rounds of production
hardening: **up to 99%** zero-shot (two of seven embedding models
tested); 13.4–27.2% after threshold tuning alone; **3.8%**, their best
achieved result, only after a full cache redesign (curated gold-FAQ +
distractor set, quality filtering, best-performing embedding model
selection out of seven tried). Their own stated conclusion: *"Architecture,
not embeddings, is what separates prototypes from production
systems"* — they found pure semantic similarity structurally unable to
resolve intent-vs-consequence confusion or qualifier/entity distinctions,
regardless of embedding model.

**Why this matters for this repo's claims**: this repo's computation
cache achieves **0% invalid-hit rate** on a structurally similar failure
mode (`docs/level3.md` Experiment A; `docs/level3b.md`'s locality
sweep, 0% invalid at every point from 2 to 80 entity combinations) —
better than this real deployment's best result after months of
optimization effort, without threshold tuning or a curated distractor
set. This is offered as supporting evidence, not a controlled
comparison — the workloads, embedding models, and traffic differ, and
no claim is made that this repo's exact numbers would transfer unchanged
to that bank's production traffic.

## What's actually new in this repo, scoped honestly

Per `docs/problem.md`, nothing in the *concepts* above is new. What this
prototype adds, and what the benchmark in `docs/experiment-log.md`
actually measures, is a specific combination at a specific scope:

1. A canonical computation representation (`aicache/computation/tasks.py`)
   distinct from the cached response.
2. Task identification via embedding nearest-neighbor against canonical
   example phrasings (not the request corpus itself) — closer in spirit
   to PARROT's variable/structure distinction than to GPTCache's
   request-to-request similarity, but implemented as a simple, offline,
   non-generative lookup rather than serving-system integration.
3. Explicit, testable parameter binding + validation with a required
   fallback path, so an unsafe binding never silently executes.
4. A same-workload, same-harness comparison against exact and semantic
   response caching, specifically designed to surface the failure mode
   (stale numeric reuse) that motivates the whole investigation.

Labeled a **potentially differentiated hypothesis** — the combination,
tested on Level 1/2 arithmetic and business-formula workloads — not a
novel invention of any of its parts.

## Level 3 addition: the RAG/tool-agent framing

Level 3 (`docs/level3.md`) sits closer to the **Agentic Plan Caching**
row above than Level 1/2 did — a workflow task's `fetch` step is, in
effect, a single hard-coded tool call, and every baseline (A-E, not
just the computation cache) has to perform that call before it can
produce any plausible answer at all, the same prerequisite a real
RAG/tool-using agent faces. The differentiator this repo still tests is
unchanged from the table above: explicit validation of whether a
rebinding is safe before executing on it, with a documented fallback —
now demonstrated, in `docs/level3.md`, catching two genuine
misidentification events on a live-state workload, not only a
closed-form arithmetic one. (This section originally described
"Agentic Plan Caching" as a research direction with no specific system
identified — a later literature search found the actual paper, now
cited in the table above with a direct comparison; the qualitative
claim made here held up.) Level 3 does not add a new prior-art
category to this table; it's the same "potentially differentiated
hypothesis" tested one level further from the closed-form arithmetic
where the comparison is easiest to control.
