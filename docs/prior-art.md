# Prior Art

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
| **GPTCache** ⭐ directly benchmarked | LLM prompt→response pairs | Configurable embedding + vector-store backend, configurable similarity threshold | No | No | No | Configurable (backend-dependent) | TTL | Partial (framework hooks) | No | **Run directly** as a comparison baseline (`benchmarks/external/gptcache_adapter.py`, local FAISS+SQLite, GPTCache's own out-of-the-box default embedding — generic `sentence-transformers/all-MiniLM-L6-v2`, deliberately left unchanged rather than forced to match this repo's own choice). On the same workload and nominal threshold (0.75) RedisVL and this repo's own semantic cache used, GPTCache showed a 70.7% invalid-hit rate vs. their 0.7% — a controlled demonstration that embedding-model choice, not caching-algorithm implementation, explains most of the gap (see EXP-006). An earlier research spike in this author's other project (Kairos) had already benchmarked GPTCache with the same embedding model on a different (short single-sentence) workload and found a similar recall-cliff-vs-false-positive pattern; this run is independent, on this project's own arithmetic/formula workload, not a reuse of that earlier result. |
| **vLLM prefix caching (PagedAttention / RadixAttention family)** | KV-cache blocks (attention state), not text | Exact prefix match on tokenized input | N/A — operates below the request-text layer | No | No | Reference-counted block eviction | N/A (attention state, not a "response") | Runs inside any workload, agentic or not | Yes — this *is* KV/prefix caching | A different layer entirely: reuses compute *inside* one inference engine, not across distinct requests at the application layer. This repo's computation cache and this are not competitors — a downstream inference engine's own prefix cache can still fire on top of anything this repo does, if the assembled prompt happens to share a prefix. Out of scope for this repo's benchmark (no real inference engine is exercised). |
| **LMCache** | KV-cache blocks, extended across GPU/CPU/disk tiers and across engine instances | Exact prefix / chunk match | N/A | No | Partial (tiering policy) | Tiered eviction | N/A | Runs inside any workload | Yes, extends vLLM's model to a distributed, multi-tier cache | Same layer as vLLM prefix caching, extended for distributed serving. Not evaluated here for the same reason. |
| **GenCache** *(name as commonly used in recent literature; exact paper/system not independently verified for this doc)* | LLM response, workload-aware | Semantic + learned admission signal | Partially — some variants incorporate usage/cost signals | No | Yes, in variants that use predicted reuse value | Learned/adaptive | TTL or usage-based | Partial | No | Represents the general direction of "workload-aware" semantic cache replacement research this repo's §19/§20 (deferred eviction work) is scoped toward, not a system this repo directly benchmarks against. |
| **Agentic Plan Caching** *(research direction, not one specific named system as of this writing)* | Multi-step plans / tool-call sequences | Task/goal similarity, sometimes structural | Varies by paper — some rebind parameters, most don't validate rebinding explicitly | Yes — this is the closest prior-art category to this repo's computation cache | Rarely a focus | Rarely a focus | Rarely addressed | Yes, by construction | No | Closest conceptual sibling to baseline D here. The differentiator this repo tests (labeled a *potentially differentiated hypothesis*, not a claim of invention) is explicit parameter validation with a documented fallback-to-LLM path when binding can't be verified safe — most plan-caching writeups this author is aware of treat the plan as reusable without a similarly explicit "is this rebinding actually safe" gate. |
| **MiniCache** | Compressed KV-cache across transformer layers (inter-layer redundancy) | Layer-similarity based merging, not request-level | N/A | No | No | N/A | N/A | N/A | Partial (works on KV representations) | Different problem (model-internal memory compression) at a different layer than anything this repo touches. Listed for completeness of the design-space map, not because it's comparable. |
| **Helium** *(name as commonly used; not independently verified against a specific paper for this doc)* | Varies by context the name is used in | — | — | — | — | — | — | — | — | Included per the brief's prior-art list; this author could not independently verify a single canonical "Helium" caching system at the time of writing, so no specific claims are made about it here — flagged rather than fabricated. |
| **Google PARROT** *(research paper on LLM serving with semantic variable annotations)* | Structured prompt "variables" the serving system tracks explicitly | Application-annotated data flow, not black-box similarity | Yes — this is close to the point: PARROT's semantic variables let the serving system distinguish "the fixed instruction" from "the variable content" | Partially — annotations describe data flow more than reusable procedures | Serving-level optimizations (batching/scheduling), not cache economics per se | Not a cache-eviction paper in the sense of §19 | N/A | Yes | Interacts with prefix caching at the serving layer | The prior-art system conceptually closest to this repo's core distinction (structure vs. parameters), from a different angle (serving-system-level prompt annotation vs. application-level task identification + closed-form re-execution). This repo's approach is much narrower in scope (deterministic arithmetic/formula tasks only) and doesn't require any serving-system integration. |
| **TinyLFU / W-TinyLFU** | Cache admission/eviction policy (object-agnostic) | Frequency-sketch-based admission, not similarity | N/A | N/A | Frequency-aware, not directly cost-aware | Yes — this is the core contribution | N/A | N/A | N/A | Directly relevant to `docs/eviction-future-work.md` §19–21, not to anything implemented in this prototype. TinyLFU-style admission is one of several policies named there as future work; nothing here reimplements or benchmarks it. |
| **CacheLib** | General-purpose cache library (object-agnostic) | Pluggable | N/A | N/A | Pluggable (supports cost-aware policies as a library feature) | Pluggable (LRU/LFU/TinyLFU-style/etc.) | App-defined | N/A | N/A | Infrastructure-layer prior art for `docs/eviction-future-work.md`, not evaluated here. |

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
closed-form arithmetic one. Level 3 does not add a new prior-art
category to this table; it's the same "potentially differentiated
hypothesis" tested one level further from the closed-form arithmetic
where the comparison is easiest to control.
