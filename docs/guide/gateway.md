# The gateway

`aicache serve` runs an OpenAI-compatible server. Point an existing app's `base_url` at it. Requests a tool can answer come back from the gateway with no model call. Everything else is forwarded unchanged to your upstream.

```yaml
upstream:
  base_url: https://api.openai.com/v1        # or Ollama, vLLM, any OpenAI-compatible server
  api_key_env: OPENAI_API_KEY
server:
  api_key_env: GATEWAY_KEY                    # clients must send this as a bearer token
  admin_token_env: GATEWAY_ADMIN              # enables /v1/aicache/*
```

```bash
export OPENAI_API_KEY=... GATEWAY_KEY=... GATEWAY_ADMIN=...
aicache serve aicache.yaml --host 0.0.0.0 --port 8080
```

## Learning in the gateway

With `learning.enabled`, the gateway forwards every request it cannot answer, returns the upstream's answer to the client unchanged, and then, in the background, gives the (request, answer) pair to the learner. That includes streamed answers, which are collected as they pass through. It does not learn from requests it passes straight through (tools, structured output, images, very long text).

A phrasing on probation sits next to the upstream: the gateway computes its result, still returns the upstream's answer, and records whether they agree. Once a phrasing is trusted the gateway answers it itself and the response carries `x-aicache-source: computation` and `x-aicache-task: <action>`.

For a request that asks for an explanation (`reply.llm_when`), the gateway forwards the request with the computed value added and an instruction not to recompute it. The response reports `x-aicache-source: computation+llm`.

Audits (`learning.audit_rate`) re-ask the upstream about a share of live answers and compare. They need `upstream.model`, because the gateway makes that call itself.

## What is intercepted

Plain chat requests: the last message must be from the user and contain text. The router sees only that last user message.

Requests are forwarded untouched, without asking the router, if they use `tools`, `functions`, `tool_choice`, `response_format`, `n` other than 1, non-text content such as images, or text longer than `server.max_text_chars`. Routing reasons for these appear as `passthrough:*` in the metrics.

## Responses

A computed answer is a normal chat completion (`object: chat.completion`, one choice, `finish_reason: stop`, usage zeros). With `stream: true` it is sent as server-sent events ending in `data: [DONE]`.

Every response carries headers:

| Header | Meaning |
|---|---|
| `x-aicache-source` | `computation` or `fallback` |
| `x-aicache-task` | the tool that computed it, or the closest candidate on a fallback |
| `x-aicache-reason` | on a fallback, why it was not computed |
| `x-aicache-request-id` | matches the decision log |

## Upstream behaviour

- If `upstream.api_key_env` is set the gateway sends that key upstream. Otherwise it forwards the client's `Authorization` header.
- If `upstream.model` is set it replaces the model in every forwarded request.
- Upstream HTTP errors are passed through with their status. If the upstream cannot be reached the gateway returns 502. With no upstream configured, an uncomputable request returns 503 while computable ones still work.

## Security

- **Safe by default.** `aicache serve` binds to `127.0.0.1`. Binding to any other address without `server.api_key_env` is refused unless you pass `--allow-unauthenticated`, for example when a private network already restricts access. If `server.api_key_env` names a variable that is empty or unset, it refuses to start, because every request would be rejected.
- Set `server.api_key_env` whenever the gateway can be reached by anyone other than your own services. Without it, anyone who can reach the port can use your upstream key through the gateway.
- `/v1/aicache/*` is disabled unless `server.admin_token_env` is set. These endpoints can read decisions and change tool modes.
- `/healthz` and `/metrics` are open. Restrict `/metrics` at your network edge.
- Request text is not stored by default. Turn on `telemetry.log_text` only if you have decided that is acceptable.
- The gateway does no TLS. Put it behind your usual proxy.
- `python:` tools run code named in the config. Treat the config like code.

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | client key if configured | the gateway |
| `GET /healthz` | none | liveness, tool count, whether an upstream is set |
| `GET /metrics` | none | Prometheus text |
| `GET /v1/aicache/stats` | admin | counters and computed share |
| `GET /v1/aicache/decisions?limit=50` | admin | recent decisions, newest first |
| `GET /v1/aicache/tools` | admin | tools you wrote, modes, slot names |
| `GET /v1/aicache/actions` | admin | learned actions with each phrasing's status and evidence |
| `POST /v1/aicache/actions/{name}/approve` | admin | let an action answer requests. Optional body `{"phrasing": "..."}` for one wording |
| `POST /v1/aicache/actions/{name}/reject` | admin | stop an action answering requests |
| `POST /v1/aicache/route` | admin | `{"text": "..."}` decision without calling the upstream |
| `POST /v1/aicache/tools/{name}/mode` | admin | `{"mode": "active"\|"shadow"\|"off"}` |

## Metrics

```
aicache_requests_total{source="computation"|"fallback"}
aicache_fallback_reason_total{reason="missing:account_size"|"below_similarity"|...}
aicache_computed_total{task="position_size"}
aicache_shadow_total{task="...",result="agree"|"disagree"}
aicache_learning_total{outcome="learned"|"merged"|"promoted"|"demoted"|"rejected:<why>"|"skipped:<why>"}
aicache_route_latency_ms  (histogram: routing decision time, not upstream time)
```

## Docker

```bash
docker build -t aicache .
docker run -p 8080:8080 -e GATEWAY_KEY=... -e GATEWAY_ADMIN=... \
  -v $PWD/aicache.yaml:/config/aicache.yaml:ro aicache
```

The container listens on `0.0.0.0` inside Docker, so your config must set `server.api_key_env: GATEWAY_KEY` (as above) or the container refuses to start. Publish the port only on interfaces you intend, for example `-p 127.0.0.1:8080:8080`.

The image is about 370 MB, runs as a non-root user and has a health check on `/healthz`. It does not include torch. Use `embedder: {type: openai, ...}` or `hashing` in the config, or build with `--build-arg EXTRAS=serve,embeddings` to include the local model (much larger). To reach an Ollama on the Docker host, use `http://host.docker.internal:11434/v1` as the upstream.

## Limits

- One process. Counters and the decision ring are in memory, per process. With several replicas each has its own, so aggregate with Prometheus, or use `telemetry.log_file` and `aicache stats`.
- Routing embeds each request. The local model takes about 10 ms per request on a laptop when otherwise idle. Use the `openai` embedder to move that cost elsewhere.
- Streaming fallbacks are relayed without shadow comparison.
