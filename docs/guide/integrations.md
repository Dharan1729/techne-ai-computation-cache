# Integrations

## Any OpenAI client: the gateway

Anything that lets you set a base URL works: the OpenAI Python and JavaScript SDKs, LangChain's `ChatOpenAI(base_url=...)`, LiteLLM, curl. See [the gateway](gateway.md).

## Python, with learning

```python
from aicache.config import load_config
from aicache.engine import ActionCache

cache = ActionCache.from_config(load_config("aicache.yaml"))
out = cache.handle(user_text, llm=lambda text: call_my_model(text))
out.source   # "computation", "computation+llm" or "llm"
out.text     # the reply
```

`handle` answers from a stored action if it can and calls your `llm` otherwise, then learns from the answer in the background. `cache.flush()` waits for learning to finish. `cache.approve(name)` and `cache.reject(name)` review what was learned. See [learning](learning.md).

## Python, tools you write yourself

```python
from aicache.config import build_router, load_config

router = build_router(load_config("aicache.yaml"))

result = router.run(user_text, fallback=lambda text: call_my_llm(text))
if result.source == "computation":
    reply = result.text            # rendered with the tool's format
else:
    reply = result.value           # whatever your fallback returned
```

| Call | Use |
|---|---|
| `router.route(text)` | Decide without calling any fallback. `result.source` is `computation` or `fallback`. Use this when you manage the model call yourself. |
| `router.run(text, fallback)` | `route` plus your fallback. Without a fallback an uncomputable request returns `source="refused"`. |
| `await router.arun(text, fallback)` | Async. Routing runs in a worker thread. The fallback may be sync or async. |
| `router.identify(text)` | Closest tool and similarity, for your own calibration. |
| `router.set_mode(name, mode)` | Change a tool's mode at runtime. |
| `router.telemetry.snapshot()` | Counters. |

`Result` fields: `source`, `value`, `text`, `task`, `similarity`, `args`, `reason`, `request_id`, `latency_ms`, and in shadow mode `shadow_value` and `shadow_agrees`.

The router is safe to share between threads.

### Without a config file

```python
from aicache import ComputationRouter, Slot

router = ComputationRouter(min_similarity=0.5)

@router.task(
    examples=["What is X percent of Y?", "Calculate X% of Y."],
    slots=[Slot("pct", after=r"%|percent"), Slot("base", before=r"of")],
    format="{value:.2f}",
)
def percent_of(pct, base):
    return base * pct / 100
```

`Slot` takes `before`, `after`, `transform` and `validate` callables. `TextSlot(name, pattern)` finds non-numeric values. This is the same engine the config uses.

## LangChain and LangGraph

Needs `pip install "ai-computation-cache[langchain]"`.

```python
from aicache.integrations.langchain import computation_first

chain = computation_first(router, ChatOpenAI(model="gpt-4o-mini"))
chain.invoke("What is 15% of 240?")        # AIMessage("36.00"), no API call
chain.invoke("Explain compound interest")  # goes to the model unchanged
```

It accepts a string, a list of messages or a prompt value, and routes the last human message. It is a normal Runnable, so `prompt | computation_first(router, llm) | parser` works, sync or async. In LangGraph, use it as a node in place of the model call. Computed replies carry `response_metadata["aicache"]` with the source, task and request id.

## Bring your own tool

A tool is a function of its slots. For anything the expression language cannot express, use `python: "mypackage.tools:position_size"` in the config, or `router.add_task(...)` in code. Keep tools pure: no side effects, no network calls, because a request may be refused and retried by the model path.
