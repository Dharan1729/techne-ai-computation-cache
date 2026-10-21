"""LangChain / LangGraph integration.

``computation_first(x, llm)`` returns a Runnable. ``x`` is an ``ActionCache`` (which
also learns from the model's answers) or a ``ComputationRouter`` (tools you wrote). Give it a string, a list
of messages or a prompt value. If a registered tool can answer, it returns an
``AIMessage`` computed by that tool with no model call. Otherwise it calls
``llm`` with the input unchanged and returns whatever the model returns.

    from aicache.integrations.langchain import computation_first
    chain = computation_first(router, ChatOpenAI(model="gpt-4o-mini"))
    chain.invoke("What is 15% of 240?")        # AIMessage("36.00"), no API call
    chain.invoke("Explain compound interest")  # goes to the model

It composes like any Runnable: ``prompt | computation_first(router, llm) | parser``
is valid, because the router sees the final rendered prompt.

Needs the ``langchain`` extra (langchain-core).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables import Runnable, RunnableLambda
except ImportError as exc:  # pragma: no cover - depends on install
    raise ImportError("this integration needs langchain-core: pip install 'ai-computation-cache[langchain]'") from exc

from ..router import ComputationRouter, Result


def _text_of(value: Any) -> str | None:
    """The request text to route, or None if the input is not plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, PromptValue):
        value = value.to_messages()
    if isinstance(value, list) and value and all(isinstance(m, BaseMessage) for m in value):
        last = value[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            return last.content
    return None


def _content(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    return c if isinstance(c, str) else "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)


def computation_first(cache: Any, llm: Runnable) -> Runnable:
    """A Runnable that tries ``cache`` before ``llm``.

    With an ``ActionCache`` it also gives the model's answer back to the cache
    (``record``), which is how it learns. With a plain ``ComputationRouter`` it
    only answers from the tools you registered."""
    learns = hasattr(cache, "lookup") and hasattr(cache, "record")
    router = cache.router if learns else cache

    def meta(r: Result) -> dict:
        task = cache.action_name(r.task) if learns else r.task
        return {"aicache": {"source": r.source, "task": task, "request_id": r.request_id}}

    def decide(value: Any) -> tuple[str | None, Result | None]:
        text = _text_of(value)
        if text is None:
            return None, None
        return text, (cache.lookup(text) if learns else router.route(text))

    def computed(r: Result) -> AIMessage:
        return AIMessage(content=r.text if r.text is not None else str(r.value), response_metadata=meta(r))

    def sync(value: Any) -> Any:
        text, r = decide(value)
        if r is not None and r.source == "computation":
            return computed(r)
        out = llm.invoke(value)
        if learns and r is not None and text is not None:
            cache.record(text, r, _content(out))
        return out

    async def asynchronous(value: Any) -> Any:
        import asyncio

        text, r = await asyncio.to_thread(decide, value)
        if r is not None and r.source == "computation":
            return computed(r)
        out = await llm.ainvoke(value)
        if learns and r is not None and text is not None:
            await asyncio.to_thread(cache.record, text, r, _content(out))
        return out

    return RunnableLambda(sync, afunc=asynchronous, name="computation_first")
