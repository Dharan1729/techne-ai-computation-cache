# Architecture

## Scope of this prototype

The brief this project follows (see project history) defines four
workload levels of increasing difficulty:

1. **Level 1 — deterministic arithmetic** (`add`, `subtract`, `multiply`,
   `percentage`, `compound_interest`)
2. **Level 2 — parameterized business formulas** (`revenue`,
   `discounted_price`, `tax`, `margin`)
3. **Level 3 — database-backed workflows** (live state: customer,
   pricing, transaction lookups feeding a formula)
4. **Level 4 — agent workflows** (multi-step plans over live state)

**This prototype implements and benchmarks Levels 1 and 2 only.** That's
a deliberate scope decision, not an oversight: Levels 1/2 are closed-form
and deterministic, which is what makes "reuse the computation, rebind
fresh parameters, execute directly — no LLM call needed on a hit" a
coherent, testable design. Levels 3/4 need a different computation
representation (a workflow/plan with data-fetch steps and freshness
requirements, not a pure function) and introduce failure modes (stale
data, authorization context, tool availability) this prototype's
validation layer doesn't model. Extending to Level 3/4 is future work,
not a small parameter change — see the note at the end of this document.

## The four conditions

```
A. No cache          Request -> LLM -> Answer
B. Exact cache        Request -> exact match? -> cached answer : (LLM -> store)
C. Semantic cache      Request -> embed -> nearest neighbor above threshold?
                              -> cached answer : (LLM -> store)
D. Computation cache   Request -> identify task -> retrieve computation
                              -> extract + bind parameters -> validate
                              -> valid? execute directly : fall back to LLM
```

All four share one workload and one LLM backend per run, so results are
directly comparable (`benchmarks/baselines.py` runs each condition over
identical `WorkloadItem`s).

## Computation representation

`aicache/computation/tasks.py` defines a `TaskSpec` per canonical
operation:

```python
TaskSpec(
    task_type="compound_interest",
    params=[ParamSpec("principal"), ParamSpec("rate", transform=pct_to_fraction), ParamSpec("years")],
    executor=lambda principal, rate, years: principal * (1 + rate) ** years,
    output_schema="number",
    freshness_requirements="static",   # always true at Level 1/2; would vary at Level 3+
    example_phrasings=[...],           # used for embedding-based task identification
)
```

This is deliberately the simplest representation that works — a task
type, required parameters (in natural left-to-right phrasing order), a
pure-Python executor that *is* the canonical computation, and a handful
of example phrasings for identification. Per the brief this project
follows, representation design is treated as an experimental variable,
not an assumed-optimal design; the brief's own richer schema (workflow
steps, tool dependencies, execution constraints) is documented here as
the eventual shape for Level 3/4, not implemented, because nothing at
Level 1/2 exercises those fields.

## The four-stage pipeline (baseline D)

1. **Task/computation identification** (`aicache/matching/task_identification.py`)
   — embed the request, compare against embeddings of each canonical
   task's example phrasings (not the request corpus), take the
   best-matching task if its similarity clears a threshold. No LLM call
   — identification has to stay cheap, or the hit path stops avoiding
   the very LLM call it exists to avoid.
2. **Retrieve the reusable computation** — a `TASK_REGISTRY` dict lookup
   by task type. At Level 1/2 this "cache" is really a fixed, small
   registry of known procedures rather than a store that grows from
   observed traffic; growing it adaptively from fallback traffic is
   future work (see below).
3. **Extract + bind parameters** (`aicache/binding/extractor.py`) —
   numbers are pulled out of the request text left-to-right and zipped
   positionally against the task's declared parameter order. This is a
   real, explicit limitation: a phrasing that states its numbers out of
   the canonical order will silently mis-bind. Two instances of exactly
   this bug were caught by this repo's own test suite during
   development (see `docs/experiment-log.md`) and fixed at the workload
   level (paraphrase templates are now constrained to preserve
   parameter order) rather than hidden.
4. **Validate** (`aicache/validation/validator.py`) — parameter count,
   finiteness, type. On failure: fall back to a full LLM call rather
   than executing on an unvalidated binding. This is the brief's "never
   optimize latency at the expense of silently incorrect answers"
   requirement, implemented as a hard gate, not a best-effort check.
5. **Execute** — call the task's pure Python executor with the bound,
   validated parameters. No LLM call on a hit, at all — the fresh answer
   comes from direct execution, not from anything cached from a prior
   request.

## Outcome taxonomy

Every request across every baseline is classified into exactly one
outcome (`aicache/metrics/types.py`):

| Outcome | Meaning |
|---|---|
| `NO_CACHE` | Baseline A only — always calls the LLM, by definition. |
| `HIT_VALID` | A reuse (cache hit or computation execution) whose answer matches ground truth. |
| `HIT_INVALID` | A reuse whose answer does **not** match ground truth — the false-hit failure mode. |
| `MISS` | No reusable entry found (exact/semantic) or task successfully identified with no matching registry entry — fell through to a fresh LLM call. |
| `FALLBACK` | Computation cache only — identification or parameter binding/validation failed, so the request fell back to a full LLM call rather than risk an unvalidated execution. |

Correctness (`aicache/metrics/correctness.py`) is checked independently
of outcome, for every request, by parsing a number out of whatever text
was ultimately served and comparing against the workload's own ground
truth with a small tolerance (see that module's docstring for why the
tolerance isn't zero). This is what makes it possible to measure, not
assume, whether a "hit" was actually correct.

## External comparison baselines (GPTCache, RedisVL)

Two real, independently-implemented semantic-caching systems run
alongside this repo's own four conditions, as an optional fifth/sixth
data point rather than a fifth/sixth *design* — `benchmarks/external/`.
Both are driven through the identical request-by-request protocol as
this repo's own `semantic_cache` baseline (see
`benchmarks/external/protocol.py`): on a miss, store the same
precomputed, ground-truth-correct answer text every baseline in this
repo would have stored; on a hit, report whatever the external system
actually served, so its correctness can be checked exactly the way this
repo checks its own.

Each runs **out-of-process**, as a subprocess into its own dedicated
virtualenv (`runner_gptcache.py`, `runner_redisvl.py`), specifically so
GPTCache's dependency tree (torch, onnxruntime, faiss-cpu) or RedisVL's
(a Redis client) can never conflict with this repo's own environment —
a real risk given GPTCache alone pulls in a different major version of
several packages this repo's own `sentence-transformers` dependency also
uses. RedisVL additionally requires a reachable Redis instance with
vector search support (e.g. `redis/redis-stack` via Docker). Both are
optional: `benchmarks/run_experiment.py` checks availability up front
and skips either one gracefully (printing why) rather than failing the
whole run — this repo's own four baselines never depend on either being
present. See `docs/experiment-log.md` (EXP-006) for what running them
found.

## Deferred: hierarchical caching (brief §16)

The brief poses a hierarchical-cache hypothesis — L0 exact response, L1
semantic response, L2 reusable computation, L3 tool-result cache, L4
intermediate-artifact cache, cooperating rather than competing. This
repo treats that as an **architectural hypothesis to document, not
implement**: the four conditions here are run as independent,
competing alternatives (each baseline owns its own cache, none consult
each other), specifically so the comparison between them is clean. A
combined system (e.g., "G: semantic response cache + computation cache"
from the brief's §15) — where a computation-cache miss/fallback checks a
semantic response cache before calling the LLM, or vice versa — is a
natural next experiment once the standalone baselines are well
understood, and is called out as such in `docs/eviction-future-work.md`
and the experiment log's "next experiment" field rather than attempted
here.

## Deferred: adaptive computation registry

`TASK_REGISTRY` is fixed at import time from nine hand-written task
specs. A more realistic system would grow this registry from observed
fallback traffic (an LLM-based extraction step turns a novel,
successfully-answered request into a new canonical task) — this is
exactly the boundary the brief's §14 draws ("start with controlled
canonical tasks... add LLM-based task extraction only after the
deterministic version works"). This prototype deliberately stays on the
near side of that boundary; LLM-based task extraction is future work,
not attempted here.
