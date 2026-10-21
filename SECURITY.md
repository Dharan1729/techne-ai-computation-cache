# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's "Report a vulnerability" on this repository's Security tab, not in a public issue. Include steps to reproduce. Expect a first reply within a few days. This is a small project, so there is no formal SLA.

## What to know when deploying

- **Configs are code.** `python: "pkg.module:function"` imports and runs code named in the config. Load configs only from sources you trust. `compute:` expressions do not execute code: they are parsed with a whitelist and rejected at load time if they use anything else.
- **Regexes in configs run on request text.** A badly written pattern can be slow on adversarial input. Requests longer than `server.max_text_chars` (default 4000) skip the router for this reason.
- **Gateway access.** Without `server.api_key_env`, anyone who can reach the port can use your upstream credentials through the gateway. `/v1/aicache/*` is disabled unless `server.admin_token_env` is set. `/metrics` and `/healthz` are open, so restrict them at your network edge. The gateway does not terminate TLS.
- **Privacy.** Decision records hold a short hash of the request, not the text, unless `telemetry.log_text` is enabled.
- **Learning stores wording.** A learned action keeps a template of each request it learned from: the request with its numbers removed. Requests that look like emails, long numbers, social security numbers or secrets are never learned from, and you can add patterns. Requests are also sent to the small model you configure; use a local model if that matters. Delete the store file (`learning.store`) to forget everything.
- **Learned formulas are checked, not trusted.** A formula comes from a model, but it is parsed by the same whitelist as `compute:` (no code execution), verified against the main LLM's answer, and kept on probation until it has agreed with the LLM on several different sets of numbers. A model that is *consistently* wrong can still teach a wrong formula; use a strong main model, `promotion: manual` or audits where that matters.
- **Not a security control.** The router decides who answers a request. It does not authenticate users or filter content.
