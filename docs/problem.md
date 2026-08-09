# Problem

## The observation

Ordinary caching assumes:

```
REQUEST → EXACT SAME REQUEST → REUSE RESPONSE
```

Semantic caching relaxes the middle step: embed the request, find the
nearest *semantically similar* prior request, reuse its response if
similarity clears a threshold.

That relaxation is safe when semantically equivalent requests really do
share an answer. It is not safe when a request's answer depends on
parameters that vary independently of its wording. Consider:

> "What is X + Y?"

where `X` and `Y` come from live state (a form field, a database row, a
sensor reading). Across three requests —

```
X=10,  Y=20
X=51,  Y=93
X=712, Y=381
```

— the wording can be near-identical ("what is X plus Y", "calculate the
sum of X and Y", "can you add X and Y") while the correct answer is
different every time. Exact-match caching correctly never reuses these
(the requests aren't identical). Semantic caching, if the paraphrase
similarity clears its threshold, *will* reuse one request's answer for
another — which is wrong whenever the parameters differ, regardless of
how similar the wording is.

What doesn't change across those three requests is the underlying
**computational structure**: `ADD(x, y)`. The operation is invariant; the
operands aren't.

## The hypothesis

> Some AI workloads may benefit more from caching reusable computation
> structure — plans, programs, workflows, or intermediate artifacts —
> than from caching final responses.

Concretely: for a task whose *procedure* is reusable but whose
*parameters* aren't, a system that (1) identifies which procedure applies,
(2) binds fresh parameters, and (3) executes should be both cheaper and
more correct than a system that tries to reuse *outputs* under the same
conditions.

## Central research questions

This repo investigates (see `docs/hypotheses.md` for the corresponding
falsifiable hypotheses):

- **RQ1 — Response reuse.** How much can exact-match caching save on a
  workload where requests are paraphrased and parameterized?
- **RQ2 — Semantic response reuse.** How much *additional* reuse does
  semantic caching provide over exact matching?
- **RQ3 — Parameter variation.** What happens when requests share
  computational structure but have materially different parameters —
  does semantic response reuse stay safe?
- **RQ4 — Computation reuse.** Can reusable computational structure be
  identified and reused safely?
- **RQ5 — Parameter binding.** Can fresh parameters be extracted and
  bound to a cached computation reliably?
- **RQ6 — Correctness.** When does computation reuse produce incorrect
  results?
- **RQ7 — Cost.** How much inference cost (tokens, calls) can be avoided?
- **RQ8 — Evaluation.** Does cache hit rate alone describe value for LLM
  workloads, or is it misleading on its own?
- **RQ9 — Workload dependence.** Under what workload distributions does
  computation reuse outperform semantic response caching?
- **RQ10 — Eviction.** (Explicitly deferred — see
  `docs/eviction-future-work.md`.) If the cache becomes large, what
  should determine admission and eviction?

## What this repo does *not* claim

This is an investigation, not an announcement. Exact caching, semantic
caching, plan/program caching, and cost-aware eviction are all
established ideas with prior art — see `docs/prior-art.md`. Nothing here
claims to have invented computation caching, semantic caching, or
cost-aware eviction. Where a specific combination of representation,
binding, validation, and reuse policy tested here appears to behave in a
way not obviously covered by that prior art, it's labeled a *potentially
differentiated hypothesis*, not a novel invention — and it's evaluated
against a workload narrow enough (deterministic arithmetic and business
formulas — see `docs/architecture.md` § Scope) that broader claims aren't
supported by what's actually been run.
