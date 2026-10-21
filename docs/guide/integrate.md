# Putting it in your product

There are three ways in, from no code to full control. Pick the lowest one that fits. All three learn the same way and share the same store.

## 1. The gateway: no code

Your app already talks to an OpenAI-compatible endpoint. Run `aicache serve` in front of it and change the base URL. This works with any language, SDK or framework that lets you set one. See [the gateway](gateway.md).

## 2. The two-call interface: any framework

If you call your model yourself, two calls are all you need. Ask the cache; if it cannot answer, call your model and tell the cache what it said:

```python
from aicache.config import load_config
from aicache.engine import ActionCache

cache = ActionCache.from_config(load_config("aicache.yaml"))

def answer(text: str) -> str:
    r = cache.lookup(text)                     # 1. ask
    if r.source == "computation":
        return r.text                          #    answered, no model call
    reply = call_my_model(text)                # 2. your model, however you call it
    cache.record(text, r, reply)               #    hand the answer back: this is how it learns
    return reply
```

That is the whole contract. Nothing else about your stack matters: any model, any prompt, any framework, sync or async.

Shortcuts for common shapes:

| You have | Use |
|---|---|
| A function from text to text | `ask = cache.wrap(my_llm)` |
| An async app | `out = await cache.ahandle(text, my_async_llm)` (the model may be sync or async) |
| A synchronous call you want handled for you | `cache.handle(text, my_llm)` returns an `Outcome` with `.source`, `.text`, `.action` |
| A LangChain or LangGraph chain | `computation_first(cache, llm)`, below |

`Outcome.source` is `"computation"`, `"computation+llm"` (the LLM wrote an explanation from the computed value) or `"llm"`.

## 3. LangChain and LangGraph

```python
from aicache.integrations.langchain import computation_first

chain = computation_first(cache, ChatOpenAI(model="gpt-4o-mini"))
chain.invoke("What is 15% of 240?")
```

It is a normal Runnable, so `prompt | computation_first(cache, llm) | parser` works, sync or async, and as a LangGraph node. It routes the last human message, and hands the model's answer back so the cache learns. Computed replies carry `response_metadata["aicache"]` with the source and the action. It also accepts a plain `ComputationRouter` if you only use tools you wrote.

## What you can swap

Every part that touches the outside world is replaceable.

| Part | How to replace it |
|---|---|
| **Your main model** | Anything you can call from a function: `llm(text) -> str`. Or an endpoint for the gateway. |
| **The small model that identifies operations** | Any OpenAI-compatible endpoint in the config, or in code `Normalizer(complete=my_function)`, where `my_function` takes chat messages and returns the model's text. |
| **Embeddings** | Config: local model, any `/v1/embeddings` server, or none. In code, any object with `encode(list[str])` and `encode_one(str)` returning normalized vectors. |
| **Where learned actions live** | One SQLite file (`learning.store`). Copy it, back it up, or `aicache export` it to YAML and keep that in git. |
| **Who approves actions** | Automatic after agreement, or a person: `aicache approve`, the admin endpoints, or `cache.approve(name)`. |
| **What it logs** | A JSONL decision log, Prometheus metrics at `/metrics`, and `cache.router.telemetry.snapshot()`. |
| **Formulas you already know** | Define them as tools in the config or in Python. They answer from the first request and sit beside the learned ones. |

```python
# a custom small model: any function from chat messages to text
from aicache.normalizer import Normalizer
normalizer = Normalizer(complete=lambda messages: my_client.chat(messages))

# custom embeddings: any object with these two methods
class MyEmbedder:
    def encode(self, texts): ...        # (N, D) array, L2-normalized
    def encode_one(self, text): ...     # (D,) array

from aicache.router import ComputationRouter
from aicache.actions import ActionStore
from aicache.engine import ActionCache
cache = ActionCache(ComputationRouter(MyEmbedder(), min_similarity=0.6), ActionStore("aicache.db"), normalizer)
```

## Things to decide before you ship

- **Where the small model runs.** Requests are sent to it for learning. A local model keeps them in your network.
- **Who can see the store.** It holds the wording of requests it learned from, with numbers removed. Treat it like request logs.
- **How you approve.** Start with `promotion: manual`; move to automatic with audits once the formulas it learns look right. See [rolling it out](rollout.md).
- **Threading.** `ActionCache` is safe to share between threads. Learning runs on one background thread.
- **Shutdown.** Call `cache.close()` to stop the learner and close the store. `cache.flush()` waits for pending learning (useful in tests and batch jobs).
