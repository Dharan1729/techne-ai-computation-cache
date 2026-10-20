# Quickstart

You have an LLM app. Some requests are really calculations: a percentage, a price after a discount, a position size. A model can get these wrong, and it takes a second or more to answer them. aicache answers them with your own code when it can bind every number safely, and sends everything else to the model.

## 1. Install

```bash
pip install "ai-computation-cache[embeddings,serve]"
```

The core package is small. `embeddings` adds the local embedding model, `serve` adds the gateway, `langchain` adds the LangChain helper. No embedding download? Use `embedder: {type: hashing}` for a first trial.

## 2. Describe one tool

```bash
aicache init aicache.yaml
```

This writes a starter config with one tool, `percent_of`:

```yaml
tools:
  - name: percent_of
    examples:                       # how users phrase this request
      - "What is X percent of Y?"
      - "Calculate X% of Y."
      - "How much is X percent of the amount Y?"
    slots:                          # where each number sits in the sentence
      - {name: pct,  after: "%|percent", min: 0, max: 100}
      - {name: base, before: "of"}
    compute: "base * pct / 100"     # a safe arithmetic expression, not code
    format: "{value:.2f}"
    tests:                          # requests that must compute, and must not
      - {text: "What is 15% of 240?", expect: 36}
      - {text: "What is 15% of that?", expect: refuse}
```

## 3. Check it

```bash
aicache validate aicache.yaml       # schema and lint
aicache test aicache.yaml           # runs the tests inside the config
aicache run aicache.yaml "Of 240, what is 15 percent?"
```

## 4. Measure it on your own requests

Make a file of real requests with the answer you expect, or `refuse` for requests that must go to the model:

```json
{"text": "What is 15% of 240?", "expect": 36}
{"text": "What does percent mean?", "expect": "refuse"}
```

```bash
aicache eval aicache.yaml --labeled requests.jsonl
aicache calibrate aicache.yaml --labeled requests.jsonl
```

`eval` reports wrong executions, the number that matters. It exits non-zero if there are any, so you can run it in CI. See [rollout](rollout.md) before sending real traffic.

## 5. Put it in front of your model

As a gateway. Existing apps change only their base URL:

```yaml
upstream: {base_url: http://localhost:11434/v1, model: llama3}   # any OpenAI-compatible server
```

```bash
aicache serve aicache.yaml --port 8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="anything")
client.chat.completions.create(model="m", messages=[{"role": "user", "content": "What is 15% of 240?"}])
```

Or in Python, with any model call as the fallback:

```python
from aicache import ComputationRouter
from aicache.config import build_router, load_config

router = build_router(load_config("aicache.yaml"))
result = router.run("What is 15% of 240?", fallback=ask_my_llm)
result.source   # "computation" or "fallback"
```

Or in a LangChain chain: see [integrations](integrations.md).
