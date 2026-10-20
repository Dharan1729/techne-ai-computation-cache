# Changelog

## 0.2.0

First release of the product layer on top of the research code.

- `ComputationRouter`: register your own deterministic tools. A request is computed only if every number binds to exactly one parameter with none left over; otherwise it goes to a fallback. Tries every candidate tool above the similarity threshold and refuses when two bind with different answers.
- Declarative YAML config with validation, per-tool golden tests, and a linter (`aicache validate`, `aicache test`).
- Safe expression language for `compute:`. No `eval`; unsafe expressions fail at load time.
- Rollout modes per tool: `active`, `shadow` (compare with the LLM, serve nothing computed), `off`. Runtime kill switch.
- `aicache eval` and `aicache calibrate`: measure wrong executions and pick `min_similarity` on your own labeled requests. `eval` exits non-zero on wrong executions, for CI.
- OpenAI-compatible gateway (`aicache serve`): streaming, client auth, admin endpoints, Prometheus metrics, decision log. Request text is not stored by default.
- Pluggable embedders: local sentence-transformers, any OpenAI-compatible `/embeddings` server, or a dependency-free hashing embedder.
- LangChain / LangGraph runnable (`aicache.integrations.langchain`).
- The core install no longer pulls in torch or the plotting stack. Use extras: `embeddings`, `serve`, `langchain`, `research`.
- `aicache serve` is local-only by default: it refuses a non-loopback bind without `server.api_key_env`, and refuses an empty key variable.
- Lexical-cue guard for tools that differ only by operation (EXP-016).
- Reproducible research results: fair-baseline comparison, 10-seed reruns, real-app integration (EXP-015 to EXP-018).

## 0.1.0

Research prototype: exact, semantic and computation caching compared across four workload levels, with real GPTCache and RedisVL baselines.
