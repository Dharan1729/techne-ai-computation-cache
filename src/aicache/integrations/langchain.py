"""LangChain / LangGraph integration.

``computation_first(router, llm)`` returns a Runnable. Give it a string, a list
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

from ..router import ComputationRouter


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


def computation_first(router: ComputationRouter, llm: Runnable) -> Runnable:
    def meta(r: Any) -> dict:
        return {"aicache": {"source": r.source, "task": r.task, "request_id": r.request_id}}

    def sync(value: Any) -> Any:
        text = _text_of(value)
        if text is not None:
            r = router.route(text)
            if r.source == "computation":
                return AIMessage(content=r.text if r.text is not None else str(r.value), response_metadata=meta(r))
        return llm.invoke(value)

    async def asynchronous(value: Any) -> Any:
        import asyncio

        text = _text_of(value)
        if text is not None:
            r = await asyncio.to_thread(router.route, text)
            if r.source == "computation":
                return AIMessage(content=r.text if r.text is not None else str(r.value), response_metadata=meta(r))
        return await llm.ainvoke(value)

    return RunnableLambda(sync, afunc=asynchronous, name="computation_first")
