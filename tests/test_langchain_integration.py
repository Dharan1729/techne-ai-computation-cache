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


def test_langchain_with_an_action_cache_learns_from_the_models_answers():
    import json
    import re

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from aicache.actions import ActionStore
    from aicache.engine import ActionCache
    from aicache.learner import LearnPolicy
    from aicache.normalizer import Normalizer
    from aicache.router import ComputationRouter

    NUM = r"\$?\d[\d,]*(?:\.\d+)?"

    def oracle(messages):
        req = re.findall(r"Request: (.*)\nAnswer:", messages[-1]["content"])[-1]
        n = len(re.findall(NUM, req))
        return json.dumps({"action": "percent_of", "names": dict(zip([f"n{i + 1}" for i in range(n)], ["pct", "base"])),
                           "expression": "base * pct / 100", "response": "{value}"})

    class Math(FakeListChatModel):
        def _call(self, messages, *a, **k):
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", messages[-1].content)]
            return f"The answer is {nums[0] * nums[1] / 100:g}."

    model = Math(responses=["x"])
    cache = ActionCache(ComputationRouter(HashingEmbedder(), min_similarity=0.0), ActionStore(":memory:"),
                        Normalizer(complete=oracle), LearnPolicy(min_agreements=1), synchronous=True)
    chain = computation_first(cache, model)
    assert chain.invoke("What is 15% of 240?").content == "The answer is 36."      # the model answered; the cache learned
    chain.invoke("What is 20% of 50?")                                              # probation: compared, model still answers
    out = chain.invoke("What is 12% of 500?")
    assert out.content == "60.0" and out.response_metadata["aicache"]["task"] == "percent_of"
    prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
    assert (prompt | chain).invoke({"q": "What is 5% of 80?"}).content == "4.0"      # composes like any Runnable
