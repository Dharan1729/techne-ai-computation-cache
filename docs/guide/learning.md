# Learning

Learning is off by default. Turn it on and the gateway (or `ActionCache`) learns from the answers your main LLM gives.

```yaml
learning:
  enabled: true
  store: aicache.db                  # SQLite file: actions, phrasings, embeddings, evidence
  normalizer:                        # the small LLM; any OpenAI-compatible endpoint
    base_url: http://localhost:11434/v1
    model: llama3
upstream: {base_url: http://localhost:11434/v1, model: llama3}   # the main LLM
```

```bash
aicache serve aicache.yaml
```

Requests the cache cannot answer go to the upstream as usual. After each answer, in the background, the small model proposes an action. Nothing changes for your users except that, over time, some requests stop reaching the upstream.

## Choosing the small model

Any OpenAI-compatible chat model. It must name an operation and write a one-line formula. In our tests an 8B model was good at it and a 1B model was weaker; see [the evidence](../experiment-log.md) for the numbers. A weaker model does not make the cache unsafe, because every proposal is verified. It makes the cache learn less.

The normalizer is not the main model. The main model's answer is what proposals are verified against, so it should be your best model. The normalizer can be small and cheap.

## The lifecycle of a phrasing

```
learned from one request  ->  probation  ->  active
                                  |
                                  +--> rejected   (two disagreements with the LLM)
```

- **probation.** The phrasing computes silently next to the LLM when a request has exactly that wording. The LLM still answers. Each comparison is recorded.
- **active.** After `min_agreements` (default 5) *different* sets of numbers with the LLM agreeing and no disagreement, the phrasing answers requests itself. With `promotion: manual` it waits for you instead.
- **rejected.** After `reject_after` (default 2) disagreements. It never answers, and the same phrasing is not learned again.
- **audit.** With `audit_rate: 0.02`, 2% of live answers are also checked against the LLM in the background. A disagreement sends the phrasing back to probation. Audits need `upstream.model` set, because the gateway makes the call itself.

## Settings

| Setting | Default | |
|---|---|---|
| `learning.enabled` | `false` | |
| `learning.store` | `aicache.db` | SQLite file. Copy it to move what was learned. |
| `learning.min_agreements` | 5 | Distinct number sets the LLM must agree on. Lower learns faster and risks more. |
| `learning.reject_after` | 2 | Disagreements that reject a phrasing. |
| `learning.promotion` | `auto` | `manual` leaves promotion to you. |
| `learning.audit_rate` | 0.0 | Share of live answers re-checked against the LLM. |
| `learning.max_actions` | 500 | Stop learning new actions at this size. |
| `learning.max_request_chars` | 300 | Longer requests are not learned from. |
| `learning.max_numbers` | 5 | Requests with more numbers are not learned from. |
| `learning.skip_patterns` | `[]` | Extra regexes. A match means the request is never learned from. |
| `learning.synchronous` | `false` | Learn inline instead of on a background thread (tests, batch runs). |
| `reply.llm_when` | explain / why / ... | Requests matching this get an LLM-written reply from the computed value. |

## Privacy

Learning stores a *template* of each request it learns from: the request with its numbers removed. The wording stays. That is the point, and it also means wording is kept on disk.

- Requests that look like they contain an email address, a long number (cards, phone numbers), a social security number, or the words password, secret, token or api key are never learned from. Add your own patterns with `skip_patterns`.
- Requests are sent to the small model. If it is a hosted service, that is a disclosure to that service. Use a local model if that matters.
- The decision log still holds only a hash of the request unless you turn on `telemetry.log_text`.
- Delete what was learned by deleting the store file, or one action with `aicache reject`.

## Reviewing what was learned

```bash
aicache actions aicache.yaml                       # every action and its phrasings, with the evidence
aicache action aicache.yaml percent_of
aicache approve aicache.yaml percent_of            # or --phrasing "What is {pct}% of {base}?"
aicache reject  aicache.yaml percent_of
aicache export  aicache.yaml learned.yaml          # review it as a file, keep it in git
aicache import  aicache.yaml learned.yaml
```

On a running gateway: `GET /v1/aicache/actions` and `POST /v1/aicache/actions/{name}/approve|reject` (admin token). The CLI edits the store file, which a running gateway reads at startup.

## Starting from history

If you already have logs of requests and the answers your model gave:

```bash
aicache learn aicache.yaml history.jsonl          # lines of {"text": "...", "answer": "..."}
```

Everything it learns starts on probation, so nothing answers a request until it has been confirmed on live traffic or approved.

## Using it from Python

```python
from aicache.config import load_config
from aicache.engine import ActionCache

cache = ActionCache.from_config(load_config("aicache.yaml"))
out = cache.handle("What is 15% of 240?", llm=lambda text: my_model(text))
out.source    # "computation", "computation+llm" or "llm"
out.text      # the reply
cache.flush() # wait for background learning (tests)
```

## Failure modes

| What happens | Why | What you see |
|---|---|---|
| The model proposes a wrong formula | One example agreed by chance | The phrasing is rejected on probation. `demoted` in the metrics. |
| The model writes a formula that does not reproduce the answer | | `rejected:expression_disagrees_with_answer` in `learning` stats. Nothing is stored. |
| The main LLM is wrong on the request being learned | A correct formula disagrees with a wrong answer | The proposal is rejected. Learning is lost, never a wrong answer. |
| The main LLM is wrong *consistently* | The cache learns its mistake | The one failure verification cannot see. Use a strong main model, manual promotion, or audits. |
| The same operation is worded many ways | Each wording is learned once | Slower coverage. Each wording costs one extra call. |
