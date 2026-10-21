"""OpenAI-compatible gateway.

Point an existing app's ``base_url`` at this server. For each chat completion
it asks the router whether a registered tool can answer. If so it returns an
OpenAI-shaped response itself, with no LLM call. Otherwise it forwards the
request unchanged to the configured upstream (OpenAI, Ollama, vLLM, anything
that speaks ``/v1/chat/completions``).

Only plain chat requests are intercepted. Requests that use tools, functions,
structured output, images or ``n > 1`` go straight to the upstream.

Endpoints
    POST /v1/chat/completions       the gateway (bearer auth if server.api_key_env is set)
    GET  /healthz                   liveness
    GET  /metrics                   Prometheus text (restrict at your network edge)
    GET  /v1/aicache/stats          counters, computed share, reasons
    GET  /v1/aicache/decisions      recent routing decisions
    GET  /v1/aicache/tools          tools and their modes
    POST /v1/aicache/route          route text without calling the upstream
    POST /v1/aicache/tools/{n}/mode change a tool's mode at runtime (kill switch)
The /v1/aicache/* endpoints are disabled unless server.admin_token_env names an
environment variable that holds a token.

Every response carries x-aicache-source (computation or fallback),
x-aicache-task and x-aicache-request-id.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from .actions import ActionStore
from .config import Config
from .embedders import EmbedderLike
from .engine import ActionCache
from .normalizer import Normalizer
from .router import MODES, ComputationRouter, Result

_INTERCEPT_BLOCKERS = ("tools", "functions", "tool_choice", "function_call", "response_format")


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _last_user_text(body: dict) -> tuple[str | None, str]:
    """(text, why_not). Returns the last user message as plain text, or (None, reason)."""
    for blocker in _INTERCEPT_BLOCKERS:
        if body.get(blocker):
            return None, f"passthrough:{blocker}"
    if (body.get("n") or 1) != 1:
        return None, "passthrough:n"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, "passthrough:no_messages"
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None, "passthrough:last_not_user"
    content = last.get("content")
    if isinstance(content, str):
        return content, ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            else:
                return None, "passthrough:non_text_content"
        return "\n".join(parts), ""
    return None, "passthrough:content"


def _completion(model: str, text: str, rid: str) -> dict:
    return {
        "id": f"chatcmpl-aicache-{rid}", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(model: str, text: str, rid: str) -> AsyncIterator[bytes]:
    base = {"id": f"chatcmpl-aicache-{rid}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}

    async def gen() -> AsyncIterator[bytes]:
        first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
        last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(first)}\n\n".encode()
        yield f"data: {json.dumps(last)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return gen()


def _delta_text(line: bytes) -> str:
    """The text in one server-sent ``data:`` line of a streamed chat completion."""
    if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
        return ""
    try:
        d = json.loads(line[5:])
        return d["choices"][0].get("delta", {}).get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def create_app(cfg: Config, *, router: ComputationRouter | None = None, cache: ActionCache | None = None,
               embedder: EmbedderLike | None = None, normalizer: Normalizer | None = None,
               upstream_transport: Any = None) -> FastAPI:
    if cache is None:
        if router is not None:
            cache = ActionCache(router, ActionStore(":memory:"))   # tools only, no learning
        else:
            cache = ActionCache.from_config(cfg, embedder=embedder, normalizer=normalizer)
    router = cache.router
    srv = cfg.server

    upstream: httpx.AsyncClient | None = None
    audit_client: httpx.Client | None = None
    up_key = os.environ.get(cfg.upstream.api_key_env, "") if cfg.upstream and cfg.upstream.api_key_env else ""
    if cfg.upstream:
        upstream = httpx.AsyncClient(base_url=cfg.upstream.base_url.rstrip("/"),
                                     timeout=cfg.upstream.timeout_s, transport=upstream_transport)
        if cfg.upstream.model:  # audits call the upstream themselves, so they need a model to ask for
            audit_client = httpx.Client(base_url=cfg.upstream.base_url.rstrip("/"), timeout=cfg.upstream.timeout_s,
                                        headers={"authorization": f"Bearer {up_key}"} if up_key else {},
                                        transport=upstream_transport)

    def audit_llm(text: str) -> str:
        assert audit_client is not None and cfg.upstream is not None
        r = audit_client.post("/chat/completions", json={"model": cfg.upstream.model,
                                                          "messages": [{"role": "user", "content": text}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if upstream is not None:
            await upstream.aclose()
        if audit_client is not None:
            audit_client.close()
        cache.close()

    app = FastAPI(title="aicache gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.router = router
    app.state.cache = cache

    def _token(env: str | None) -> str:
        return os.environ.get(env, "") if env else ""

    def _authorized(request: Request, env: str | None) -> bool:
        want = _token(env)
        got = request.headers.get("authorization", "")
        return bool(want) and got.startswith("Bearer ") and hmac.compare_digest(got[7:], want)

    def admin(request: Request) -> None:
        if not _token(srv.admin_token_env):
            raise HTTPException(403, "admin endpoints are disabled: set server.admin_token_env")
        if not _authorized(request, srv.admin_token_env):
            raise HTTPException(401, "invalid admin token")

    def _headers(r: Result, source: str | None = None) -> dict[str, str]:
        h = {"x-aicache-source": source or r.source, "x-aicache-request-id": r.request_id}
        action = cache.action_name(r.task)
        if action:
            h["x-aicache-task"] = action
        if r.reason and r.source != "computation":
            h["x-aicache-reason"] = r.reason.split(":")[0]
        return h

    # ---- the gateway ------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        if srv.api_key_env and not _authorized(request, srv.api_key_env):
            return _error(401, "invalid or missing bearer token", "invalid_api_key")
        raw = await request.body()
        if len(raw) > srv.max_body_bytes:
            return _error(413, "request body too large", "request_too_large")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return _error(400, "body must be a JSON object", "invalid_request_error")

        text, why = _last_user_text(body)
        if text is not None and len(text) > srv.max_text_chars:
            text, why = None, "passthrough:too_long"
        if text is None:
            decision = Result("fallback", reason=why)
            router.telemetry.record(source="fallback", task=None, similarity=-1.0, reason=why,
                                    latency_ms=0.0, text="")
        else:
            decision = await asyncio.to_thread(router.route, text)

        model = body.get("model") or "aicache"
        if decision.source == "computation":
            content = decision.text if decision.text is not None else str(decision.value)
            await asyncio.to_thread(cache.note_computed, text or "", decision, audit_llm if audit_client else None)
            if upstream is not None and cache.wants_llm_reply(text or ""):
                # the request asks for an explanation: the LLM writes it, from the computed value
                return await _proxy(request, _with_result(body, content), decision, source="computation+llm",
                                    learn_text=None)
            rid = decision.request_id or uuid.uuid4().hex[:12]
            if body.get("stream"):
                return StreamingResponse(_sse(model, content, rid), media_type="text/event-stream",
                                         headers=_headers(decision))
            return JSONResponse(_completion(model, content, rid), headers=_headers(decision))

        if upstream is None:
            return _error(503, f"this request could not be computed ({decision.reason}) and no upstream is configured",
                          "aicache_no_upstream")
        return await _proxy(request, body, decision, learn_text=text)

    def _with_result(body: dict, content: str) -> dict:
        msgs = [dict(m) for m in body["messages"]]
        last = msgs[-1]
        note = f"\n\nA verified calculation gives {content}. Use that value in your reply and do not recompute it."
        last["content"] = (last["content"] if isinstance(last["content"], str)
                           else "\n".join(p.get("text", "") for p in last["content"] if isinstance(p, dict))) + note
        return {**body, "messages": msgs}

    async def _proxy(request: Request, body: dict, decision: Result, *, learn_text: str | None,
                     source: str = "fallback") -> Response:
        assert upstream is not None and cfg.upstream is not None
        if cfg.upstream.model:
            body = {**body, "model": cfg.upstream.model}
        headers = {"content-type": "application/json"}
        if up_key:
            headers["authorization"] = f"Bearer {up_key}"
        elif request.headers.get("authorization"):
            headers["authorization"] = request.headers["authorization"]
        h = _headers(decision, source)

        async def learn_from(answer: str) -> None:
            if learn_text is not None and answer:
                await asyncio.to_thread(cache.after_llm, learn_text, decision, answer)

        try:
            if body.get("stream"):
                req = upstream.build_request("POST", "/chat/completions", json=body, headers=headers)
                resp = await upstream.send(req, stream=True)

                async def relay() -> AsyncIterator[bytes]:
                    seen: list[str] = []
                    buf = b""
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                            buf += chunk
                            *lines, buf = buf.split(b"\n")
                            seen.extend(_delta_text(ln) for ln in lines)
                    finally:
                        await resp.aclose()
                    if resp.status_code == 200:
                        await learn_from("".join(seen))

                return StreamingResponse(relay(), status_code=resp.status_code,
                                         media_type=resp.headers.get("content-type", "text/event-stream"), headers=h)
            resp = await upstream.post("/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            return _error(502, f"upstream request failed: {type(exc).__name__}", "aicache_upstream_error")
        if resp.status_code == 200:
            try:
                await learn_from(resp.json()["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        return Response(resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"), headers=h)

    # ---- operations -------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "tools": len(cache.pinned), "learned_actions": cache.store.count(),
                "learning": cache.learner is not None, "upstream": upstream is not None}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(router.telemetry.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/aicache/stats", dependencies=[Depends(admin)])
    async def stats() -> dict:
        return router.telemetry.snapshot()

    @app.get("/v1/aicache/decisions", dependencies=[Depends(admin)])
    async def decisions(limit: int = 50) -> list[dict]:
        return router.telemetry.recent(max(1, min(limit, 1000)))

    @app.get("/v1/aicache/tools", dependencies=[Depends(admin)])
    async def tools() -> list[dict]:
        return [{"name": t.name, "mode": t.mode, "description": t.description, "examples": len(t.examples),
                 "slots": [s.name for s in t.slots] + [s.name for s in t.text_slots]}
                for n, t in router.tasks.items() if n in cache.pinned]

    @app.get("/v1/aicache/actions", dependencies=[Depends(admin)])
    async def actions() -> list[dict]:
        return [{"name": a.name, "status": a.status, "expression": a.compute, "params": a.params, "used": a.used,
                 "phrasings": [{"text": t.text, "status": t.status, "agree": t.agree, "disagree": t.disagree,
                                "used": t.used, "note": t.note} for t in a.templates]}
                for a in cache.store.list()]

    @app.post("/v1/aicache/actions/{name}/{verb}", dependencies=[Depends(admin)])
    async def review(name: str, verb: str, payload: dict | None = None) -> dict:
        """verb is approve or reject; body may name one phrasing: {"phrasing": "..."}"""
        if cache.learner is None:
            raise HTTPException(409, "learning is not enabled")
        if verb not in ("approve", "reject"):
            raise HTTPException(404, "verb must be approve or reject")
        if cache.store.get(name) is None:
            raise HTTPException(404, f"no action named {name!r}")
        phrasing = (payload or {}).get("phrasing")
        (cache.approve if verb == "approve" else cache.reject)(name, phrasing)
        return {"name": name, "status": cache.store.get(name).status}  # type: ignore[union-attr]

    @app.post("/v1/aicache/route", dependencies=[Depends(admin)])
    async def route(payload: dict) -> dict:
        text = payload.get("text")
        if not isinstance(text, str):
            raise HTTPException(422, "body must be {\"text\": \"...\"}")
        r = await asyncio.to_thread(router.route, text)
        return {"source": r.source, "task": cache.action_name(r.task), "value": r.value, "text": r.text,
                "similarity": round(r.similarity, 4), "reason": r.reason, "args": r.args,
                "shadow_value": r.shadow_value, "request_id": r.request_id}

    @app.post("/v1/aicache/tools/{name}/mode", dependencies=[Depends(admin)])
    async def set_mode(name: str, payload: dict) -> dict:
        mode = payload.get("mode")
        if name not in cache.pinned:
            raise HTTPException(404, f"no tool named {name!r}")
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        router.set_mode(name, mode)
        return {"name": name, "mode": mode}

    return app
