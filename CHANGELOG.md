# Changelog

## 0.3.0

The product now follows the original design: it learns the operation behind a request instead of needing every tool written by hand.

- **Learning (`aicache.engine.ActionCache`).** When the main LLM answers a request the cache could not, a small LLM identifies the operation (names the roles and writes a formula) while code removes the values. Proposals are verified deterministically, embedded, and stored in SQLite.
- **Probation, per wording.** A new wording computes silently next to the LLM and answers requests only after agreeing with it on several different sets of numbers and never disagreeing. A trusted action's new wording earns trust by itself. Promotion can be manual. Audits re-check live answers.
- **Exact wording decides.** A learned wording answers a request only if the request is that wording with different numbers. Embeddings retrieve candidates and never decide. (An earlier, looser design bound requests for the wrong operation in a real-model test and was replaced before release.)
- **Equivalent operations merge by computation:** the same function under another name, or with roles in another order, becomes another wording of the same action.
- Gateway learns from proxied answers, including streams; new admin endpoints list, approve and reject learned actions; LLM-written replies for explanation requests; audits.
- CLI: `learn`, `actions`, `action`, `approve`, `reject`, `export`, `import`.
- Privacy: requests that look like they contain emails, long numbers, SSNs or secrets are never learned from.
- Config: `learning` and `reply` sections; tools are optional when learning is on; lint rules for risky learning settings.
- `ActionCache.lookup` / `record` (a two-call interface for any framework), `wrap`, `ahandle`; the LangChain runnable now learns.
- Experiments EXP-019 (the learning loop with real small models) and EXP-020 (against semantic and exact caches and a hand-written registry).

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
