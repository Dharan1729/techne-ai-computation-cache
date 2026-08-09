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
| **Redis LangCache** | LLM prompt→response pairs | Semantic (embedding similarity) + exact | No — matches on prompt similarity only | No | Not published in detail as of this writing | Not a focus of the product surface | TTL-based | No | No | Closest commercial analog to this repo's baseline C (semantic response cache). This repo's semantic baseline is a from-scratch minimal reimplementation for controlled experimentation, not a comparison against the product itself. |
| **GPTCache** | LLM prompt→response pairs | Configurable embedding + vector-store backend, configurable similarity threshold | No | No | No | Configurable (backend-dependent) | TTL | Partial (framework hooks) | No | Directly informs this repo's semantic-cache baseline design (embeddings + configurable threshold + response storage). An earlier research spike in this author's other project (Kairos) benchmarked GPTCache with `sentence-transformers/all-MiniLM-L6-v2` and found a recall cliff between similarity threshold 0.80–0.85 for that embedding model — this repo's own threshold sweep (0.80–0.97, see `docs/experiment-log.md`) is a similar-in-spirit but independently-run sweep on a different (arithmetic/paraphrase) workload, not a reuse of that earlier result. |
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
