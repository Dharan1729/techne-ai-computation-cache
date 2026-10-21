"""The gateway learns from traffic it proxies, and then answers without the upstream."""

import json
import re

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aicache.actions import ActionStore  # noqa: E402
from aicache.config import load_config  # noqa: E402
from aicache.embedders import HashingEmbedder  # noqa: E402
from aicache.engine import ActionCache  # noqa: E402
from aicache.normalizer import Normalizer  # noqa: E402
from aicache.server import create_app  # noqa: E402

NUM = r"\$?\d[\d,]*(?:\.\d+)?"


def oracle(messages):
    req = re.findall(r"Request: (.*)\nAnswer:", messages[-1]["content"])[-1]
    n = len(re.findall(NUM, req))
    return json.dumps({"action": "percent_of", "names": dict(zip([f"n{i + 1}" for i in range(n)], ["pct", "base"])),
                       "expression": "base * pct / 100", "response": "{value}"})


class Upstream:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        text = body["messages"][-1]["content"]
        nums = [float(x.replace("$", "").replace(",", "")) for x in re.findall(NUM, text)]
        answer = f"The answer is {nums[0] * nums[1] / 100:g}." if len(nums) >= 2 else "Hello!"
        if body.get("stream"):
            parts = [f'data: {json.dumps({"choices": [{"delta": {"content": answer[:8]}}]})}\n\n',
                     f'data: {json.dumps({"choices": [{"delta": {"content": answer[8:]}}]})}\n\n', "data: [DONE]\n\n"]
            return httpx.Response(200, content="".join(parts).encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "u", "object": "chat.completion", "model": "m", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]})


def make(monkeypatch=None, **learning):
    lc = {"enabled": True, "synchronous": True, "min_agreements": 2,
          "normalizer": {"base_url": "http://n/v1", "model": "tiny"}}
    lc.update(learning)
    cfg = load_config({"min_similarity": 0.0, "embedder": {"type": "hashing"}, "learning": lc,
                       "upstream": {"base_url": "http://up/v1"}, "server": {"admin_token_env": "ADM"}})
    up = Upstream()
    cache = ActionCache.from_config(cfg, embedder=HashingEmbedder(), normalizer=Normalizer(complete=oracle),
                                    store=ActionStore(":memory:"))
    app = create_app(cfg, cache=cache, upstream_transport=httpx.MockTransport(up))
    return TestClient(app), up, cache


def ask(client, text, **extra):
    return client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": text}], **extra})


H = {"authorization": "Bearer adm"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ADM", "adm")


def test_the_gateway_learns_from_proxied_answers_then_answers_by_itself():
    c, up, cache = make()
    r = ask(c, "What is 15% of 240?")
    assert r.headers["x-aicache-source"] == "fallback" and "36" in r.json()["choices"][0]["message"]["content"]
    assert cache.store.get("percent_of").status == "probation"
    ask(c, "What is 20% of 50?")
    ask(c, "What is 10% of 90?")                                  # second distinct agreement: promoted
    assert cache.store.get("percent_of").status == "active"
    n = len(up.requests)
    r = ask(c, "What is 12% of 500?")
    assert r.headers["x-aicache-source"] == "computation" and r.headers["x-aicache-task"] == "percent_of"
    assert r.json()["choices"][0]["message"]["content"] == "60.0" and len(up.requests) == n
    s = c.get("/v1/aicache/stats", headers=H).json()
    assert s["learning"]["learned"] == 1 and s["learning"]["promoted"] == 1


def test_a_streamed_answer_is_learned_from_too():
    c, up, cache = make()
    r = ask(c, "What is 15% of 240?", stream=True)
    assert "data: [DONE]" in r.text
    assert cache.store.get("percent_of") is not None


def test_requests_the_router_must_not_touch_are_not_learned_from():
    c, up, cache = make()
    ask(c, "What is 15% of 240?", response_format={"type": "json_object"})
    ask(c, "What is 15% of 240?", n=2)
    assert cache.store.count() == 0


def test_sensitive_requests_are_not_learned_from():
    c, up, cache = make()
    ask(c, "My card 4111111111111111: what is 15% of 240?")
    assert cache.store.count() == 0
    assert c.get("/v1/aicache/stats", headers=H).json()["learning"] == {"skipped:sensitive_pattern": 1}


def test_review_endpoints_list_approve_and_reject():
    c, up, cache = make(promotion="manual")
    for r in ("What is 15% of 240?", "What is 20% of 50?", "What is 10% of 90?"):
        ask(c, r)
    a = c.get("/v1/aicache/actions", headers=H).json()
    assert a[0]["name"] == "percent_of" and a[0]["status"] == "probation"
    assert a[0]["phrasings"][0]["agree"] == 2 and "ready for approval" in a[0]["phrasings"][0]["note"]
    assert ask(c, "What is 7% of 100?").headers["x-aicache-source"] == "fallback"
    assert c.post("/v1/aicache/actions/percent_of/approve", headers=H, json={}).json()["status"] == "active"
    assert ask(c, "What is 7% of 100?").headers["x-aicache-source"] == "computation"
    assert c.post("/v1/aicache/actions/percent_of/reject", headers=H).json()["status"] == "rejected"
    assert ask(c, "What is 7% of 100?").headers["x-aicache-source"] == "fallback"
    assert c.post("/v1/aicache/actions/nope/approve", headers=H).status_code == 404
    assert c.post("/v1/aicache/actions/percent_of/bless", headers=H).status_code == 404
    assert c.post("/v1/aicache/actions/percent_of/approve").status_code == 401


def test_an_explanation_request_is_written_by_the_llm_from_the_computed_value():
    c, up, cache = make(min_agreements=1)
    for r in ("Explain what 15% of 240 is", "Explain what 20% of 50 is"):
        ask(c, r)
    n = len(up.requests)
    r = ask(c, "Explain what 12% of 500 is")
    assert r.headers["x-aicache-source"] == "computation+llm" and len(up.requests) == n + 1
    sent = up.requests[-1]["messages"][-1]["content"]
    assert "60.0" in sent and "do not recompute" in sent


def test_the_health_endpoint_reports_learning():
    c, up, cache = make()
    ask(c, "What is 15% of 240?")
    h = c.get("/healthz").json()
    assert h["learning"] is True and h["learned_actions"] == 1
