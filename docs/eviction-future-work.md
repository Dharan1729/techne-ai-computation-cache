# Eviction & Admission — Deferred Future Work

This document exists to be explicit about a scope boundary: **this
prototype implements no eviction and no admission policy of any kind.**
Every cache in this repo (`ExactCache`, `SemanticCache`,
`ComputationCache`'s registry) grows without bound for the duration of a
run and is discarded at process exit. Nothing here should be read as
evidence for or against any eviction policy, including H8 in
`docs/hypotheses.md`.

## Why this is deferred, not solved

Eviction policies require realistic scale and workload distributions to
evaluate meaningfully. This repo's benchmark runs 150 requests across 9
task types — nowhere near the scale (10K–10M+ requests, realistic
popularity distributions, temporal locality, workload shifts) needed to
say anything credible about which entries should be evicted under memory
pressure. Fabricating eviction numbers from this benchmark's scale would
be worse than not reporting them at all.

## Admission vs. eviction — two different questions

These are worth keeping conceptually distinct in future work:

- **Admission**: should this object enter the cache at all? A very
  expensive computation observed once may not justify caching (the
  amortization never pays off). A moderately expensive computation
  observed repeatedly may.
- **Eviction**: given the cache is full and something must leave, which
  object leaves?

Most classical policies (LRU, LFU, FIFO, random) only really address
eviction. TinyLFU/W-TinyLFU (see `docs/prior-art.md`) is notable for
addressing admission explicitly via a frequency sketch.

## Candidate policies (documented, not implemented or compared)

- **LRU** — evict least recently used.
- **LFU** — evict least frequently used.
- **FIFO** — evict oldest entry.
- **Random** — simple baseline, sometimes competitive in practice.
- **TinyLFU / W-TinyLFU** — frequency-aware admission + replacement.
- **Size-aware** — prefer retaining smaller or larger objects depending
  on the objective.
- **Cost-aware** — weight by recomputation cost (how expensive was this
  to produce the first time).
- **Token-aware** — weight by tokens actually saved on reuse (this
  repo's own metrics — see `aicache/metrics/`— already track
  per-request token savings, which is the raw material a token-aware
  policy would need; nothing here turns that into a policy).
- **Workload-aware** — weight by predicted future reuse probability.
- **Economic-value-aware** — a composite:

  ```
  expected future reuse × recomputation cost × validity probability
  ────────────────────────────────────────────────────────────────
                        memory footprint
  ```

  possibly also incorporating freshness, latency value, quality risk, and
  dependency count. This is a research question in its own right
  ("which cache object produces the greatest expected future value under
  a constrained memory budget"), not something this repo attempts to
  answer.

## What a real load test would need

- Request traces at 10K / 100K / 1M / 10M+ scale.
- Controlled cache capacities and varying working-set sizes.
- Realistic popularity distributions: Zipfian, uniform, bursty,
  periodic, workload-shifting, long-tail.
- Object sizes and per-object (re)computation costs that actually vary —
  this repo's Level 1/2 tasks are all cheap to compute directly, which
  is exactly why eviction economics don't show up as interesting at this
  scale: there's nothing expensive enough to fight over.

## Embedding model migration (a related, also-deferred concern)

If the semantic layer's embedding model changes (e.g. swapping
`all-MiniLM-L6-v2` for a newer model), every stored vector in a live
semantic cache becomes incomparable to newly-computed query vectors —
dimensionality may differ, and even same-dimensionality models don't
share a vector space. A real system would need dual indexing during
migration, a backfill process, an A/B evaluation period, and a rollback
path. This repo's semantic-cache baseline is rebuilt fresh on every
benchmark run, so this concern never actually arises here — flagged as
an engineering reality outside this prototype's scope, not solved.

## What would have to be true before this is worth building

Per the brief this project follows: don't build eviction speculatively.
Build it when there's a real cache that's actually filling up under real
traffic, with real per-object cost variance worth fighting over — at
that point, the metrics infrastructure already in this repo
(`aicache/metrics/`) is the natural place to start, since it already
tracks the per-request cost/token/latency signals any of the
cost-aware policies above would need as input.
