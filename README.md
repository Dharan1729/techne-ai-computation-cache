# ai-computation-cache

A research prototype that tests one idea: for LLM requests whose answer depends on parameters, **cache the computation, not the answer**.

A semantic cache decides two requests are "the same" because their wording is close. That fails when the answer depends on numbers or ids that the wording barely shows.

```
"What is 10 + 20?"     -> 30
"What is 712 + 381?"   -> 1093
```

An embedding model sees these as nearly the same sentence. A semantic cache can return 30 for the second one, with full confidence.

The alternative tested here: identify the operation, extract the current parameters, validate them, and run deterministic code. If validation fails, fall back to the LLM.

**Status: a controlled investigation, not a product, and not a new technique.** Plan caching and structured-intent caching already exist. See [Prior art](#prior-art). The question here was when this approach beats the alternatives and where it breaks.

## What the experiments found

All numbers below come from [docs/experiment-log.md](docs/experiment-log.md), entries EXP-015 to EXP-017. Each can be reproduced with [scripts/reproduce.sh](scripts/reproduce.sh).

### 1. Against a fair baseline, the main gain is latency, not accuracy

Same 150 requests, same local Llama 3 8B model via Ollama, temperature 0.

| System | Correct | Median latency | Requests that called the LLM |
|---|---|---|---|
| LLM answers directly | 71/150 (47%) | 1.1 s | 100% |
| LLM answers with a step-by-step prompt | 71/150 (47%) | 2.5 s | 100% |
| **LLM picks the tool and parameters, code does the math** (the fair baseline) | 139/150 (93%) | 1.2 s | 100% |
| Computation cache, raw LLM as fallback | 131/150 (87%) | 129 ms | 23% |
| Computation cache, the router above as fallback | 146/150 (97%) | 129 ms | 23% |

![Accuracy and latency against the fair baseline](results/figures/fig1_baselines.png)

- The 93% and 97% accuracy intervals overlap at this sample size, so the accuracy difference is not conclusive.
- The latency gap is real: about 9x lower median, and 77% of requests needed no LLM call.
- The 129 ms median was measured while a local model shared the same laptop. A lookup takes about 11 ms on an idle machine.

### 2. My first cache was wrong, and I fixed it

Over 10 seeds the cache made **20 wrong hits in 1,500 requests**. Requests for add, subtract and multiply share a parameter signature. The embedding match sometimes picked the wrong one, and my validation checked parameter shape only. For example, "508 + 780 = ?" was matched to subtract.

A small lexical-cue guard brought this to **0 in 1,500**, at a cost of about one point of hit rate.

```
python -m benchmarks.guard_ab    # under a minute, offline
```

That is encouraging, not proof of general safety. The guard was written after the failures were seen, and the evaluation uses the same workload templates. Seeds 6 to 10 were not looked at while writing it.

### 3. Four claims I made earlier did not survive re-testing

![Four claims that changed](results/figures/fig3_corrections.png)

### 4. A real open-source app: where the router fails silently

The two risk tools from [wshobson/financial-chat](https://github.com/wshobson/financial-chat) are imported unmodified and tested with 144 positive and 20 negative requests. **The tool router in this test is a local Llama 3 8B, not the model the app itself uses.** Nothing here is a claim about how that app behaves with its own model.

| System | Correct (of 144) | Wrong answers executed | Median latency |
|---|---|---|---|
| LLM router over the real tool schemas | 133 (92%) | 10 | 1.6 s |
| Cache, extractor v2 | 111 (77%) | 0 | 12 ms |
| Hybrid: cache first, router on reject | 141 (98%) | 2 | 12 ms median, 385 ms mean |

- In the five of the router's ten wrong executions that I inspected, "0.5%" was passed as 0.005. The tool expects percent, so it returned a position size 100x too small, with no error. A clearer tool description would probably help. That was not tested.
- The cache refused 23% of requests it could not safely bind. Refusing is safer than a plausible wrong answer, but it is not free.
- Phrasing templates were split into calibration, development and a fresh holdout. On the holdout the cache executed 41 of 64 requests with 0 wrong.
- Embedding similarity alone could not gate this domain. Scores for real and unrelated requests overlapped (0.46 to 0.67 against 0.37 to 0.70). A structural check did the real work: every number must bind to exactly one parameter, with none left over.

### 5. When a semantic cache is fine, and when it is not

A good semantic cache stayed about 99% correct on plain arithmetic, using `redis/langcache-embed-v1` at similarity 0.75. What changed is how much traffic it could serve.

![Cache usefulness against repeat rate](results/figures/fig2_repeats.png)

When no request repeats an earlier request's exact numbers, the semantic cache served 0% from cache and the computation cache about 75%. When numbers repeat often, the two get close. If your parameters repeat, you may not need this.

### 6. Database and branching workflows

Over 10 seeds, with the semantic cache at the harness default similarity of 0.65, looser than the 0.75 above:

| | Computation cache | Semantic cache |
|---|---|---|
| Level 3, database workflows (225 requests per seed) | 100.0% correct, 1 wrong hit in 2,250 | 64.1% correct |
| Level 4, branching workflows (180 requests per seed) | 100.0% correct, 0 wrong hits in 1,800 | 38.2% correct |

The looser threshold explains part of the semantic cache's lower score, so these are not directly comparable to section 5. Correctness under database mutation was tested in only three hand-picked scenarios (EXP-010).

## Limitations

- Templated phrasings, not real user traffic.
- One 8B model on one laptop. A stronger model would narrow the accuracy gap.
- The arithmetic validator still cannot catch out-of-order numbers. "Over 3 years at 5 percent, what does $1000 grow to?" binds to (3, 5, 1000) and returns 4.6e21. The benchmark never states numbers out of order, so it does not test this. The financial-chat extractor does handle order.
- The lexical-cue guard covers add, subtract and multiply only.
- The extractor and guard were built after seeing failures, so development-set numbers are optimistic.
- No eviction or admission policy. See [docs/eviction-future-work.md](docs/eviction-future-work.md).
- The 1B-model result (52 of 60 with a proper router prompt) is a small slice of 60 requests.

Older entries in the experiment log (EXP-001 to EXP-014) are single-seed. Where they disagree with EXP-015 to EXP-017, trust the later entries.

## Use it in your own workflow

`ComputationRouter` lets you put your own deterministic tool in front of an LLM. It is experimental and the API may change.

```python
from aicache import ComputationRouter, Slot

router = ComputationRouter(min_similarity=0.5)

@router.task(
    examples=["What is X percent of Y?", "Calculate X% of Y."],
    slots=[Slot("pct", after=r"%|percent"), Slot("base", before=r"of")],
)
def percent_of(pct, base):
    return base * pct / 100

result = router.run("What is 15% of 240?", fallback=ask_your_llm)
result.source   # "computation", "fallback" or "refused"
result.value    # 36.0
result.reason   # why it was not computed, when it was not
```

What happens on each request:

1. The closest registered tool is found by embedding similarity.
2. Every number in the request must be claimed by exactly one slot, with none left over. Order in the sentence does not matter.
3. Your function runs. If any step is unsure, or your function raises, the request goes to `fallback`. With no fallback it is refused. It is never guessed.

A complete runnable example, with a position-size tool, is in [examples/position_size_router.py](examples/position_size_router.py):

```bash
python examples/position_size_router.py
```

Before you trust it on real traffic:

- **Calibrate on your own requests.** `router.identify(text)` returns the best task and its similarity. Run it on real requests and on unrelated ones to pick `min_similarity`. In the finance test, similarity alone could not separate the two. The slot check did the real rejecting.
- **Measure wrong executions.** Count how often it computes a wrong answer on your traffic. [benchmarks/real_app/financial_chat_integration.py](benchmarks/real_app/financial_chat_integration.py) shows the pattern, with a calibration set, held-out phrasings and negative requests.
- **Use `cues` when two tools take the same parameters.** Add and subtract both take two numbers, and the embedding match confused them. `add_task(..., cues=[r"\bplus\b|\+"])` requires an operator word for that tool.
- **Keep units explicit.** A slot gives you the number as written. Use `transform` only when your tool expects something else.

It is a poor fit for tools with side effects, tools that depend on context outside the request text, or answers that need judgment.

## Reproduce it

Python 3.10 or newer. The first run downloads the embedding model from Hugging Face.

```bash
pip install -e ".[dev]"

bash scripts/reproduce.sh quick       # 142 tests + the wrong-hit guard comparison, about 5 minutes, offline
bash scripts/reproduce.sh sweep       # repeat-rate sweep and figure 2, about 15 minutes, offline
bash scripts/reproduce.sh realmodel   # fair-baseline comparison, needs Ollama and llama3
bash scripts/reproduce.sh realapp     # financial-chat integration, needs Ollama, llama3 and network
```

Everything writes to `out/` and never overwrites the committed `results/`.

For the two real-model steps:

```bash
ollama serve &
ollama pull llama3
```

Latency numbers depend on your hardware and on whatever else is using it. Expect the same ordering, not the same milliseconds. Model sampling is pinned (temperature 0, fixed seed), but a different Ollama or model build can still change individual answers.

| Claim | How to check |
|---|---|
| 20 wrong hits in 1,500 without the guard, 0 with it | `python -m benchmarks.guard_ab` |
| Fair-baseline accuracy and latency | `bash scripts/reproduce.sh realmodel`, committed copy in `results/real_baselines/` |
| Financial-chat comparison | `bash scripts/reproduce.sh realapp`, committed copy in `results/real_app_financial_chat.json` |
| Repeat-rate sweep | `bash scripts/reproduce.sh sweep`, raw summaries in `results/sweeps/repetition/` |
| Level 3 and 4 across 10 seeds | raw summaries in `results/sweeps/level34/`, protocol in EXP-017 |
| Figures | `python -m benchmarks.plot_linkedin --rep-dir results/sweeps/repetition --out out/figures` |

`results/real_baselines_preguard/` holds the same real-model run from before the guard was added, kept for comparison.

## How it works

Four conditions run over the same workload and the same backend:

```
A. No cache          Request -> LLM -> Answer
B. Exact cache       Request -> exact match? -> cached answer : (LLM -> store)
C. Semantic cache    Request -> embed -> nearest neighbor above threshold?
                            -> cached answer : (LLM -> store)
D. Computation cache Request -> identify task -> extract + bind parameters
                            -> validate -> valid? execute directly : fall back to LLM
```

The workload has 9 task types (add, subtract, multiply, percentage, compound interest, revenue, discounted price, tax, margin), seeded paraphrase templates and randomized parameters. Levels 3 and 4 add database-backed and branching workflows. Architecture detail: [docs/architecture.md](docs/architecture.md).

Backends: a deterministic simulated model (the default, offline), local Ollama, and Gemini. External baselines are real GPTCache and real RedisVL, each in its own isolated subprocess. They are optional and skipped when unavailable.

## Prior art

Nothing here claims to invent computation caching, semantic caching or plan caching. The full map of 17 systems is in [docs/prior-art.md](docs/prior-art.md). The closest published work:

- Zhang, Wornow, Wan, Olukotun, [Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](https://arxiv.org/abs/2506.14852). Caches reusable plan templates. Reported figures in the current abstract: 50.31% lower cost, 27.28% lower latency.
- Basu, [Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning](https://arxiv.org/abs/2602.18922). Splits a request into a cache key plus separately injected parameters.
- Daivam, [Reducing False Positives in RAG Semantic Caching: a Banking Case Study](https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/), InfoQ, November 2025. A production semantic cache matched "I don't want this card anymore" to investment account closure at 88.7% confidence.

What this repo adds is a controlled comparison of one combination, identify then bind then validate then fall back, against exact and semantic caching, with the failures reported.

## Repository layout

```
docs/         problem, prior art, architecture, hypotheses, evaluation,
              level3, level3b, level4, differentiation, experiment-log
src/aicache/  the library: router.py (public API), cache/, matching/,
              computation/, binding/, validation/, metrics/, llm/, enterprise/
workloads/    workload generators and templates for levels 1 to 4
benchmarks/   run_experiment.py, real_baselines.py, guard_ab.py,
              real_app/ (financial-chat), level3/, level4/, external/
tests/        142 unit and integration tests
examples/     a runnable position-size router example
results/      committed raw results and figures
scripts/      reproduce.sh
```

## License

MIT. See [LICENSE](LICENSE).
