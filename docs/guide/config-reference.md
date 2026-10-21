# Config reference

A config is one YAML file. `aicache validate` checks every field below and lists all problems at once.

```yaml
version: 1
min_similarity: 0.5
ambiguity_margin: 0.03
embedder: {type: sentence-transformers}
upstream: {base_url: http://localhost:11434/v1}
telemetry: {log_file: decisions.jsonl}
server: {api_key_env: GATEWAY_KEY, admin_token_env: GATEWAY_ADMIN}
tools: [...]
```

## Top level

| Field | Default | Meaning |
|---|---|---|
| `version` | `1` | Config format version. |
| `min_similarity` | `0.5` | A tool is only tried if its closest example scores at least this. Per-tool override below. Run `aicache calibrate` to choose it. |
| `ambiguity_margin` | `0.03` | If two tools both bind and give different answers within this similarity of each other, the request is refused as ambiguous. |
| `embedder` | sentence-transformers | See below. |
| `upstream` | none | Needed for `aicache serve` to forward requests the router cannot answer. |
| `telemetry` | see below | Decision log and counters. |
| `server` | see below | Gateway security and limits. |
| `learning` | off | Learn actions from the main LLM's answers. See below and [learning](learning.md). |
| `reply` | see below | When the LLM writes the reply. |
| `tools` | `[]` | Tools you write yourself. Give at least one tool, or turn learning on. |

## `embedder`

| `type` | Notes |
|---|---|
| `sentence-transformers` (default) | Local model. `model` defaults to `redis/langcache-embed-v1`, whose model card states no license: check its terms before commercial use, or set `model` to another sentence-transformers model. Needs the `embeddings` extra. |
| `openai` | Any server with `POST /v1/embeddings`. Needs `base_url` and `model`; `api_key_env` names the environment variable holding the key. |
| `hashing` | No dependencies, no download, about 0.1 ms per request. Matches spelling, not meaning. See the note below on thresholds. |

**Each embedder needs its own `min_similarity`.** On our 131-request finance set, `aicache calibrate` found both safe: `hashing` at a threshold of 0.1 computed 91 of 96 computable requests with no wrong answers and no false triggers, and the neural model at 0.45 computed 90 of 96 with none. Used at the neural model's threshold (0.45), `hashing` computed only 31 of 96. A looser embedder leans more on the slot check, so measure on your own requests. We did not test paraphrases looser than these templates, more tools, or more varied unrelated requests, where the neural model may matter more.

## `learning`

See [learning](learning.md) for what each setting does.

```yaml
learning:
  enabled: true
  store: aicache.db
  normalizer: {base_url: http://localhost:11434/v1, model: llama3, api_key_env: NORMALIZER_KEY, timeout_s: 60, json_mode: true}
  min_agreements: 5
  reject_after: 2
  promotion: auto            # or manual
  audit_rate: 0.0
  max_actions: 500
  max_request_chars: 300
  max_numbers: 5
  skip_patterns: []
  queue_size: 100
  synchronous: false
```

With `learning.enabled`, `normalizer.base_url` and `normalizer.model` are required. `json_mode` asks the endpoint for JSON output; turn it off for an endpoint that rejects `response_format`.

## `reply`

```yaml
reply:
  llm_when: "\\b(explain|why|summari[sz]e|walk me through)"
```

A request matching this regex gets its reply written by the LLM from the computed value (the LLM is told not to recompute it). Every other request is answered from the action's reply template with no LLM call. Set `llm_when: null` to never use the LLM for replies.

## `upstream`

| Field | Meaning |
|---|---|
| `base_url` | Include `/v1`. Requests go to `{base_url}/chat/completions`. |
| `api_key_env` | Environment variable holding the key to send upstream. If unset, the client's `Authorization` header is forwarded. |
| `model` | Replaces the model named in every forwarded request. |
| `timeout_s` | Default 60. |

## `telemetry`

| Field | Default | Meaning |
|---|---|---|
| `log_file` | none | Append one JSON record per decision. Read it with `aicache stats`. |
| `log_text` | `false` | Store request text in records. Off by default: records hold a short hash instead, so logs are safe to ship. |
| `ring_size` | 1000 | Recent decisions kept in memory for `/v1/aicache/decisions`. |

## `server`

| Field | Meaning |
|---|---|
| `api_key_env` | If set, clients must send `Authorization: Bearer <value of that variable>` to `/v1/chat/completions`. |
| `admin_token_env` | Token for `/v1/aicache/*`. If unset those endpoints are disabled. |
| `max_body_bytes` | Default 1,000,000. Larger bodies get 413. |
| `max_text_chars` | Default 4000. Longer requests skip the router and go upstream. |

## Tools

```yaml
- name: position_size            # identifier
  description: ...               # optional, shown by /v1/aicache/tools
  examples: [...]                # at least 1; 3 or more recommended
  slots: [...]                   # numbers, at least 1
  text_slots: [...]              # optional non-numeric values
  cues: [...]                    # optional regexes; one must match
  compute: "..."                 # or python: "pkg.module:function"
  format: "{value:.2f} shares of {symbol}"
  mode: active                   # active | shadow | off
  min_similarity: 0.45           # optional per-tool threshold
  tests: [...]
```

### `slots`

Each slot says where one number sits relative to words around it.

| Field | Meaning |
|---|---|
| `name` | Identifier. This is the variable name inside `compute`. |
| `before` | Regex for words that usually come just before the number, for example `entry|buy|in at`. |
| `after` | Regex for what follows it, for example `%|percent` or `years?`. |
| `min`, `max` | Reject values outside the range. Checked on the number as written. |
| `integer` | Reject non-integers. |
| `scale` | Multiply the number before use. `0.01` turns 5 (percent) into 0.05. |

Give `before`, `after`, or both. A keyword right after a number beats one before it. A request is computed only if every number in the text is claimed by exactly one slot and every slot has a number. Order in the sentence does not matter.

### `text_slots`

`{name, pattern}`. The regex must match exactly one distinct value in the request. If it has a capture group, group 1 is the value. Use it for tickers or names. Text slots can appear in `format` and in `python` tools, not in `compute` expressions.

### `compute`

A safe expression over the numeric slot names. Allowed: `+ - * / // % **`, comparisons, `and`/`or`/`not`, `a if c else b`, numbers, and these functions: `abs round min max sqrt floor ceil log log10 exp`, plus the constants `pi` and `e`. Nothing else: no attribute access, indexing, strings, comprehensions or other calls. A bad expression fails at load time. Exponents above 1000 are refused.

### `python`

`python: "package.module:function"` calls your own function with the slots as keyword arguments (text slots included). The function must be importable where aicache runs. **This imports and runs code named in the config, so only load configs you trust.**

### `format`

A template for the reply text. Fields: `{value}`, `{value:.2f}`, and any slot name, `{symbol}`. Only simple fields and format specs are supported. Default `{value}`.

### `cues`

Regexes, at least one of which must match the request. Use them when two tools take the same parameters and differ by operation, such as add and subtract. In our benchmark, embeddings confused such tools and cues removed the errors. `aicache validate` warns when same-shaped tools have no cues.

### `mode`

| Mode | Behaviour |
|---|---|
| `active` | Computes and answers. |
| `shadow` | Computes but lets the model answer, and records whether the model's number agrees. Use it first on real traffic. |
| `off` | Ignored. |

### `tests`

`{text, expect}` where `expect` is a number or `refuse`. Run by `aicache test`, with every tool treated as active. Have at least one of each kind per tool.
