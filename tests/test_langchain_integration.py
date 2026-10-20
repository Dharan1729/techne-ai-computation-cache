import asyncio

import pytest

pytest.importorskip("langchain_core")
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from aicache import ComputationRouter, Slot  # noqa: E402
from aicache.embedders import HashingEmbedder  # noqa: E402
from aicache.integrations.langchain import computation_first  # noqa: E402


def _chain():
    router = ComputationRouter(embedder=HashingEmbedder(), min_similarity=0.0)
    router.add_task("percent_of", lambda pct, base: base * pct / 100,
                    examples=["What is X percent of Y?", "Calculate X% of Y.", "X percent of Y"],
                    slots=[Slot("pct", after=r"%|percent"), Slot("base", before="of")], format="{value:.2f}")
    llm = FakeListChatModel(responses=["from the model"] * 10)
    return router, llm, computation_first(router, llm)


def test_computes_without_calling_the_model_for_strings_and_messages():
    router, llm, chain = _chain()
    out = chain.invoke("What is 15% of 240?")
    assert isinstance(out, AIMessage) and out.content == "36.00"
    assert out.response_metadata["aicache"]["source"] == "computation"
    out2 = chain.invoke([SystemMessage(content="be brief"), HumanMessage(content="What is 20% of 50?")])
    assert out2.content == "10.00"
    assert llm.i == 0  # the fake model was never called


def test_falls_back_to_the_model_unchanged():
    _, llm, chain = _chain()
    assert chain.invoke("What is 15% of that?").content == "from the model"
    assert chain.invoke([HumanMessage(content="hello")]).content == "from the model"
    assert llm.i == 2


def test_composes_in_a_chain_and_works_async():
    router, llm, chain = _chain()
    prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
    composed = prompt | chain
    assert composed.invoke({"q": "What is 15% of 240?"}).content == "36.00"
    assert composed.invoke({"q": "tell me a joke"}).content == "from the model"
    assert asyncio.run(chain.ainvoke("What is 15% of 240?")).content == "36.00"
    assert asyncio.run(chain.ainvoke("tell me a joke")).content == "from the model"
