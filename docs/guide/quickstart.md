# Quickstart

You have an LLM app. Some of its requests are really calculations: a percentage, a price after a discount, a loan payment. A model answers them slowly and sometimes wrongly, and it answers the same *kind* of request again and again with different numbers.

aicache sits in front of your model. It learns the operation behind such requests from the answers your model already gives, and once it has checked that it is right, it computes them itself. Everything else goes to your model, unchanged.

Read [how it works](how-it-works.md) first if you want the idea. This page gets it running.

## 1. Install

```bash
pip install "ai-computation-cache[embeddings,serve]"
```

The core package is small. `embeddings` adds the local embedding model, `serve` adds the gateway, `langchain` adds the LangChain helper.

## 2. Pick two models

- **Your main model.** The one your app already calls. Its answers are what learning is checked against, so use your best one.
- **A small model** that identifies the operation behind a request. Any OpenAI-compatible endpoint. A local 8B model through Ollama works well, and a smaller one learns less but stays safe.

```bash
ollama pull llama3        # used as both below, to try it
```

## 3. Write a config

```yaml
# aicache.yaml
embedder: {type: sentence-transformers}
min_similarity: 0.6

upstream:                                  # your main model
  base_url: http://localhost:11434/v1
  model: llama3

learning:
  enabled: true
  store: aicache.db                        # where learned actions are kept
  normalizer:                              # the small model
    base_url: http://localhost:11434/v1
    model: llama3
```

```bash
aicache validate aicache.yaml
```

## 4. Run the gateway and point your app at it

```bash
aicache serve aicache.yaml --port 8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="anything")
client.chat.completions.create(model="m", messages=[{"role": "user", "content": "What is 15% of 240?"}])
```

`aicache serve` listens on `127.0.0.1`, so only your machine can reach it. To listen on another address you must set `server.api_key_env` (see [the gateway](gateway.md)); otherwise it refuses to start.

## 5. Watch it learn

Send a few requests that share one wording and differ in the numbers:

```
What is 15% of 240?     -> goes to your model. aicache learns an action from the answer.
What is 20% of 50?      -> goes to your model. The new action is on probation: it computes
                           silently, and its answer is compared with your model's.
What is 10% of 90?      -> same
...
What is 12% of 500?     -> answered by aicache. No model call.
```

```bash
aicache actions aicache.yaml
#   percent_of  [active]  base * pct / 100   params: pct, base
#       active    What is {pct}% of {base}?   (agree 5, disagree 0, answered 3)
```

Every answer the gateway gives carries `x-aicache-source: computation` or `fallback`, so you can see which was which.

## 6. Decide how much to trust it

By default a phrasing goes live after the model agreed with it on 5 different sets of numbers and never disagreed. To check each one yourself first:

```yaml
learning:
  promotion: manual
```

```bash
aicache actions aicache.yaml                 # see what is waiting, with the evidence
aicache approve aicache.yaml percent_of
```

See [learning](learning.md) for audits, privacy settings and the review workflow.

## Tools you write yourself

If you already know the operations, define them directly. They answer from the first request and need no learning:

```yaml
tools:
  - name: percent_of
    examples: ["What is X percent of Y?", "Calculate X% of Y.", "How much is X percent of the amount Y?"]
    slots:
      - {name: pct,  after: "%|percent", min: 0, max: 100}
      - {name: base, before: "of"}
    compute: "base * pct / 100"
    tests:
      - {text: "What is 15% of 240?", expect: 36}
      - {text: "What is 15% of that?", expect: refuse}
```

```bash
aicache test aicache.yaml
aicache run aicache.yaml "What is 15% of 240?"
```

See [tools you write yourself](how-it-decides.md) and the [config reference](config-reference.md). Both kinds work together.

## From Python

```python
from aicache.config import load_config
from aicache.engine import ActionCache

cache = ActionCache.from_config(load_config("aicache.yaml"))
out = cache.handle("What is 15% of 240?", llm=lambda text: call_my_model(text))
out.source   # "computation" or "llm"
out.text     # the reply
```
