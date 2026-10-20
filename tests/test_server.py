"""The OpenAI-compatible gateway, with a mock upstream."""

import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aicache.config import build_router, load_config  # noqa: E402
from aicache.embedders import HashingEmbedder  # noqa: E402
from aicache.server import create_app  # noqa: E402


def cfg_dict(**server):
    return {
        "min_similarity": 0.0,
        "embedder": {"type": "hashing"},
        "upstream": {"base_url": "http://upstream/v1", "api_key_env": "TEST_UPSTREAM_KEY"},
        "server": server,
        "tools": [{
            "name": "percent_of",
            "examples": ["What is X percent of Y?", "Calculate X% of Y.", "X percent of Y"],
            "slots": [{"name": "pct", "after": "%|percent"}, {"name": "base", "before": "of"}],
            "compute": "base * pct / 100",
            "format": "{value:.2f}",
        }],
    }


class Upstream:
    """Records what the gateway forwards; replies like an OpenAI server."""

    def __init__(self, status=200, content="LLM says hi"):
        self.requests: list[httpx.Request] = []
        self.status, self.content = status, content

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body.get("stream"):
                data = (b'data: {"choices":[{"delta":{"content":"streamed"}}]}\n\n' b"data: [DONE]\n\n")
                return httpx.Response(self.status, content=data, headers={"content-type": "text/event-stream"})
            return httpx.Response(self.status, json={
                "id": "up-1", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}]})
        return httpx.Response(404)


def make(upstream=None, monkeypatch=None, **server):
    cfg = load_config(cfg_dict(**server))
    up = upstream or Upstream()
    app = create_app(cfg, router=build_router(cfg, embedder=HashingEmbedder()),
                     upstream_transport=httpx.MockTransport(up))
    return TestClient(app), up, app.state.router


def chat(client, text, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": text}], **extra})


def test_computed_request_gets_an_openai_shaped_answer_without_calling_upstream():
    client, up, _ = make()
    r = chat(client, "What is 15% of 240?")
    assert r.status_code == 200 and up.requests == []
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "m"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "36.00"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert r.headers["x-aicache-source"] == "computation" and r.headers["x-aicache-task"] == "percent_of"
    assert r.headers["x-aicache-request-id"]


def test_unanswerable_request_is_forwarded_unchanged_with_the_clients_auth():
    client, up, _ = make()
    r = client.post("/v1/chat/completions", headers={"authorization": "Bearer client-key"},
                    json={"model": "m", "messages": [{"role": "user", "content": "What is 15% of that?"}], "temperature": 0.2})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "LLM says hi"
    assert r.headers["x-aicache-source"] == "fallback" and r.headers["x-aicache-reason"] == "missing"
    sent = up.requests[0]
    assert sent.headers["authorization"] == "Bearer client-key"
    assert json.loads(sent.content)["temperature"] == 0.2 and str(sent.url) == "http://upstream/v1/chat/completions"


def test_configured_upstream_key_wins_and_model_override(monkeypatch):
    monkeypatch.setenv("TEST_UPSTREAM_KEY", "server-key")
    d = cfg_dict()
    d["upstream"]["model"] = "forced-model"
    cfg = load_config(d)
    up = Upstream()
    app = create_app(cfg, router=build_router(cfg, embedder=HashingEmbedder()), upstream_transport=httpx.MockTransport(up))
    r = TestClient(app).post("/v1/chat/completions", headers={"authorization": "Bearer client-key"},
                             json={"model": "ignored", "messages": [{"role": "user", "content": "hello there"}]})
    assert r.status_code == 200
    assert up.requests[0].headers["authorization"] == "Bearer server-key"
    assert json.loads(up.requests[0].content)["model"] == "forced-model"


def test_streaming_computed_answer_is_valid_sse():
    client, up, _ = make()
    r = chat(client, "What is 15% of 240?", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream") and up.requests == []
    events = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first = json.loads(events[0])
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["content"] == "36.00"
    assert json.loads(events[1])["choices"][0]["finish_reason"] == "stop"


def test_streaming_fallback_is_relayed():
    client, up, _ = make()
    r = chat(client, "tell me a story", stream=True)
    assert "streamed" in r.text and r.text.strip().endswith("data: [DONE]")
    assert r.headers["x-aicache-source"] == "fallback"


@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "f"}}]},
    {"response_format": {"type": "json_object"}},
    {"n": 2},
])
def test_requests_the_router_must_not_touch_go_straight_upstream(extra):
    client, up, router = make()
    r = chat(client, "What is 15% of 240?", **extra)
    assert r.headers["x-aicache-source"] == "fallback" and len(up.requests) == 1
    assert router.telemetry.snapshot()["fallback_reasons"]["passthrough:" + list(extra)[0]] == 1


def test_non_text_and_non_user_messages_pass_through():
    client, up, _ = make()
    img = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is 15% of 240?"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}
    assert client.post("/v1/chat/completions", json=img).headers["x-aicache-source"] == "fallback"
    sys_last = {"model": "m", "messages": [{"role": "assistant", "content": "What is 15% of 240?"}]}
    assert client.post("/v1/chat/completions", json=sys_last).headers["x-aicache-source"] == "fallback"
    parts = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 15% of 240?"}]}]}
    assert client.post("/v1/chat/completions", json=parts).headers["x-aicache-source"] == "computation"


def test_only_the_last_user_message_is_routed():
    client, _, _ = make()
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "What is 20% of 50?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "10.00"


def test_too_long_text_skips_the_router():
    client, up, _ = make(max_text_chars=20)
    r = chat(client, "What is 15% of 240? " + "padding " * 10)
    assert r.headers["x-aicache-source"] == "fallback" and len(up.requests) == 1


def test_bad_bodies_are_rejected_cleanly():
    client, _, _ = make(max_body_bytes=2000)
    assert client.post("/v1/chat/completions", content=b"not json").status_code == 400
    assert client.post("/v1/chat/completions", json=[1, 2]).status_code == 400
    assert client.post("/v1/chat/completions", content=b"x" * 5000).status_code == 413


def test_upstream_failure_and_missing_upstream_are_reported():
    client, _, _ = make(upstream=Upstream(status=500))
    assert chat(client, "hello").status_code == 500  # the upstream's own status is passed through

    def boom(request):
        raise httpx.ConnectError("down")

    cfg = load_config(cfg_dict())
    app = create_app(cfg, router=build_router(cfg, embedder=HashingEmbedder()), upstream_transport=httpx.MockTransport(boom))
    r = TestClient(app).post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 502 and r.json()["error"]["type"] == "aicache_upstream_error"

    d = cfg_dict()
    d.pop("upstream")
    cfg2 = load_config(d)
    app2 = create_app(cfg2, router=build_router(cfg2, embedder=HashingEmbedder()))
    r2 = TestClient(app2).post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hello"}]})
    assert r2.status_code == 503 and "no upstream" in r2.json()["error"]["message"]
    ok = TestClient(app2).post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "What is 15% of 240?"}]})
    assert ok.status_code == 200  # computable requests work even with no upstream


def test_client_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("GW_KEY", "s3cret")
    client, up, _ = make(api_key_env="GW_KEY")
    body = {"model": "m", "messages": [{"role": "user", "content": "What is 15% of 240?"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    assert client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer s3cret"}).status_code == 200
    assert up.requests == []


def test_admin_endpoints_are_disabled_without_a_token_and_protected_with_one(monkeypatch):
    client, _, _ = make()
    assert client.get("/v1/aicache/stats").status_code == 403
    monkeypatch.setenv("ADMIN_T", "adm")
    client, _, _ = make(admin_token_env="ADMIN_T")
    assert client.get("/v1/aicache/stats").status_code == 401
    assert client.get("/v1/aicache/stats", headers={"authorization": "Bearer nope"}).status_code == 401
    H = {"authorization": "Bearer adm"}
    chat(client, "What is 15% of 240?")
    chat(client, "hello")
    s = client.get("/v1/aicache/stats", headers=H).json()
    assert s["requests_total"] == 2 and s["llm_calls_avoided"] == 1
    d = client.get("/v1/aicache/decisions?limit=1", headers=H).json()
    assert len(d) == 1 and "text" not in d[0]  # request text is not stored by default
    t = client.get("/v1/aicache/tools", headers=H).json()
    assert t[0]["name"] == "percent_of" and t[0]["mode"] == "active"
    rt = client.post("/v1/aicache/route", json={"text": "What is 10% of 50?"}, headers=H).json()
    assert rt["source"] == "computation" and rt["value"] == 5.0
    assert client.post("/v1/aicache/route", json={"nope": 1}, headers=H).status_code == 422


def test_kill_switch_changes_behaviour_at_runtime(monkeypatch):
    monkeypatch.setenv("ADMIN_T", "adm")
    client, up, _ = make(admin_token_env="ADMIN_T")
    H = {"authorization": "Bearer adm"}
    assert chat(client, "What is 15% of 240?").headers["x-aicache-source"] == "computation"
    assert client.post("/v1/aicache/tools/percent_of/mode", json={"mode": "off"}, headers=H).status_code == 200
    r = chat(client, "What is 15% of 240?")
    assert r.headers["x-aicache-source"] == "fallback" and len(up.requests) == 1
    assert client.post("/v1/aicache/tools/percent_of/mode", json={"mode": "loud"}, headers=H).status_code == 422
    assert client.post("/v1/aicache/tools/nope/mode", json={"mode": "off"}, headers=H).status_code == 404


def test_shadow_mode_serves_the_llm_answer_and_records_agreement(monkeypatch):
    monkeypatch.setenv("ADMIN_T", "adm")
    client, up, router = make(upstream=Upstream(content="The answer is 36."), admin_token_env="ADMIN_T")
    router.set_mode("percent_of", "shadow")
    r = chat(client, "What is 15% of 240?")
    assert r.json()["choices"][0]["message"]["content"] == "The answer is 36."  # the LLM's answer is what is served
    assert len(up.requests) == 1
    up.content = "It is 40."
    chat(client, "What is 15% of 240?")
    assert router.telemetry.snapshot()["shadow"] == {"percent_of:agree": 1, "percent_of:disagree": 1}


def test_health_and_metrics_are_open():
    client, _, _ = make()
    chat(client, "What is 15% of 240?")
    assert client.get("/healthz").json() == {"status": "ok", "tools": 1, "upstream": True}
    m = client.get("/metrics")
    assert m.status_code == 200 and 'aicache_requests_total{source="computation"} 1' in m.text
