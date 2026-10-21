"""The ``aicache`` command line.

    aicache init [path]                       write a starter config
    aicache validate CONFIG [--strict]        check the config and lint it
    aicache test CONFIG                       run the golden tests inside the config
    aicache run CONFIG "text"                 route one request and show the decision
    aicache eval CONFIG --labeled FILE        wrong executions on your labeled requests
    aicache calibrate CONFIG --labeled FILE   sweep min_similarity
    aicache serve CONFIG                      OpenAI-compatible gateway (needs the 'serve' extra)
    aicache stats LOGFILE                     summarise a decision log

    aicache learn CONFIG FILE                 learn actions from {"text", "answer"} pairs (JSONL)
    aicache actions CONFIG                    list learned actions and their phrasings
    aicache action CONFIG NAME                show one action with its evidence
    aicache approve CONFIG NAME               let an action (or one --phrasing) answer requests
    aicache reject CONFIG NAME                stop an action (or one --phrasing) answering requests
    aicache export CONFIG FILE                write learned actions to YAML for review
    aicache import CONFIG FILE                load actions from such a file

Exit codes: 0 success, 1 a check failed (tests, lint with --strict, wrong
executions above --max-wrong), 2 bad usage or an invalid config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .actions import ActionStore
from .config import ConfigError, build_router, lint_config, load_config, run_golden_tests
from .embedders import make_embedder
from .tooling import calibrate, evaluate, load_labeled, summarize_log

STARTER = """\
# aicache starter config. Check it, then try it:
#   aicache validate aicache.yaml
#   aicache test     aicache.yaml
#   aicache run      aicache.yaml "What is 15% of 240?"
version: 1
min_similarity: 0.5
embedder: {type: sentence-transformers}   # or `hashing` (no download; then lower min_similarity: run `aicache calibrate`)
# upstream:                                # your main LLM, needed for `aicache serve`
#   base_url: http://localhost:11434/v1
#   model: llama3
# learning:                                # learn more actions from your LLM's answers
#   enabled: true
#   normalizer: {base_url: http://localhost:11434/v1, model: llama3}
tools:
  - name: percent_of
    description: A percentage of an amount.
    examples:
      - "What is X percent of Y?"
      - "Calculate X% of Y."
      - "How much is X percent of the amount Y?"
    slots:
      - {name: pct,  after: "%|percent", min: 0, max: 100}
      - {name: base, before: "of"}
    compute: "base * pct / 100"
    format: "{value:.2f}"
    tests:
      - {text: "What is 15% of 240?", expect: 36}
      - {text: "Of 240, what is 15 percent?", expect: 36}
      - {text: "What is 15% of that?", expect: refuse}
"""


def _load(args: argparse.Namespace):
    """The config and a router that includes the learned actions in its store."""
    from .engine import ActionCache

    cfg = load_config(Path(args.config))
    emb = make_embedder(args.embedder) if getattr(args, "embedder", None) else None
    if cfg.learning.enabled:
        return cfg, ActionCache.from_config(cfg, embedder=emb).router
    return cfg, build_router(cfg, embedder=emb)


def _store(args: argparse.Namespace) -> ActionStore:
    cfg = load_config(Path(args.config))
    return ActionStore(cfg.learning.store)


def cmd_init(args: argparse.Namespace) -> int:
    p = Path(args.path)
    if p.exists() and not args.force:
        print(f"{p} exists; use --force to overwrite", file=sys.stderr)
        return 2
    p.write_text(STARTER)
    print(f"wrote {p}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    issues = lint_config(cfg)
    for i in issues:
        print(f"{i.level:7s} {i.tool}: {i.message}")
    warn = sum(i.level == "warning" for i in issues)
    print(f"ok: {len(cfg.tools)} tools, {warn} warnings")
    return 1 if (args.strict and warn) else 0


def cmd_test(args: argparse.Namespace) -> int:
    cfg, router = _load(args)
    out = run_golden_tests(cfg, router)
    for o in out:
        print(f"{'PASS' if o.passed else 'FAIL'}  {o.tool}: {o.text!r}" + ("" if o.passed else f"\n      {o.detail}"))
    failed = sum(not o.passed for o in out)
    print(f"{len(out) - failed}/{len(out)} passed")
    return 1 if failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    _, router = _load(args)
    r = router.route(args.text)
    print(json.dumps({"source": r.source, "task": r.task, "value": r.value, "text": r.text,
                      "similarity": round(r.similarity, 4), "reason": r.reason, "args": r.args,
                      "latency_ms": round(r.latency_ms, 3)}, indent=2, default=str))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    _, router = _load(args)
    rep = evaluate(router, load_labeled(args.labeled))
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        print(f"requests            {rep.n}  ({rep.positives} should compute, {rep.negatives} should refuse)")
        print(f"computed correctly  {rep.computed_correct}  (coverage {rep.coverage:.1%} of the ones that should compute)")
        print(f"WRONG executions    {rep.wrong_executions}")
        print(f"false triggers      {rep.false_triggers}")
        print(f"refused (missed)    {rep.missed}   correct refusals {rep.correct_refusals}")
        print(f"bad executions      {rep.bad_executions} of {rep.n}; 95% upper bound on the rate {rep.bad_execution_upper_95:.2%}")
        print(f"route latency       p50 {rep.latency_p50_ms:.1f} ms, p95 {rep.latency_p95_ms:.1f} ms")
        if rep.miss_reasons:
            print("why requests were refused:", ", ".join(f"{k} x{v}" for k, v in rep.miss_reasons.items()))
        for w in rep.wrong_examples[:10]:
            print("  BAD", json.dumps(w, default=str))
    return 1 if rep.bad_executions > args.max_wrong else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    _, router = _load(args)
    items = load_labeled(args.labeled)
    negs = sum(i.expect == "refuse" for i in items)
    cal = calibrate(router, items)
    print("threshold  coverage  wrong  false_triggers")
    for p in cal.points:
        print(f"  {p.threshold:.2f}      {p.coverage:6.1%}   {p.wrong_executions:4d}   {p.false_triggers:4d}")
    if cal.recommended is None:
        print("\nNo threshold gives zero wrong executions and zero false triggers on this data.\n"
              "Fix the tools (slots, cues, examples) rather than raising the threshold; `aicache eval` shows which requests go wrong.")
    else:
        print(f"\nsafe from {cal.lowest_safe:.2f} upward on this data; recommended min_similarity: {cal.recommended:.2f}")
        print("(the highest safe threshold that keeps at least 98% of the best coverage. A higher threshold filters\n"
              " more unrelated traffic you did not label. Confirm with `aicache eval` after changing it.)")
    if negs < 20:
        print(f"note: only {negs} requests are labeled 'refuse'. With fewer than about 20 negatives this recommendation is weak.")
    return 0


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:
        print(f"serve needs the 'serve' extra: pip install 'ai-computation-cache[serve]' ({exc})", file=sys.stderr)
        return 2
    cfg = load_config(Path(args.config))
    key_env = cfg.server.api_key_env
    if key_env and not os.environ.get(key_env):
        print(f"error: server.api_key_env is '{key_env}' but that environment variable is empty or unset. "
              "Every request would be rejected. Set it, or remove server.api_key_env.", file=sys.stderr)
        return 2
    if not _is_loopback(args.host) and not key_env and not args.allow_unauthenticated:
        print(f"error: refusing to listen on {args.host} without client authentication: anyone who can reach the port "
              "could use your upstream credentials through the gateway.\n"
              "Set server.api_key_env in the config (clients then send it as a bearer token), "
              "bind to 127.0.0.1, or pass --allow-unauthenticated if the network already restricts access.",
              file=sys.stderr)
        return 2
    if not key_env:
        print("note: client authentication is off. Fine for local use; set server.api_key_env before exposing this.",
              file=sys.stderr)
    emb = make_embedder(args.embedder) if args.embedder else None
    app = create_app(cfg, embedder=emb)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    from collections import Counter

    from .engine import ActionCache

    cfg = load_config(Path(args.config))
    if not cfg.learning.enabled:
        print("error: learning is not enabled in this config (learning.enabled)", file=sys.stderr)
        return 2
    emb = make_embedder(args.embedder) if args.embedder else None
    cache = ActionCache.from_config(cfg, embedder=emb, synchronous=True)
    outcomes: Counter[str] = Counter()
    n = 0
    for line in Path(args.file).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        outcomes[cache.learner.consider(str(d["text"]), str(d["answer"])).split(":")[0]] += 1  # type: ignore[union-attr]
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"{n} pairs: " + ", ".join(f"{k} x{v}" for k, v in outcomes.most_common()))
    for a in cache.store.list():
        print(f"  {a.name:22s} {a.compute}   phrasings: {len(a.templates)} (all on probation)")
    print("Learned phrasings start on probation: they answer requests only after agreeing with the LLM on live traffic.\n"
          "Review with `aicache actions`, or approve with `aicache approve`.")
    return 0


def _print_action(a, detail: bool) -> None:
    print(f"{a.name}  [{a.status}]  {a.compute}   params: {', '.join(a.params)}   answered: {a.used}")
    for t in a.templates:
        ev = f"agree {t.agree}, disagree {t.disagree}, answered {t.used}"
        print(f"    {t.status:9s} {t.text}   ({ev})" + (f"   note: {t.note}" if detail and t.note else ""))


def cmd_actions(args: argparse.Namespace) -> int:
    actions = _store(args).list()
    if args.status:
        actions = [a for a in actions if a.status == args.status]
    for a in actions:
        _print_action(a, False)
    print(f"{len(actions)} actions")
    return 0


def cmd_action(args: argparse.Namespace) -> int:
    a = _store(args).get(args.name)
    if a is None:
        print(f"error: no action named {args.name!r}", file=sys.stderr)
        return 2
    _print_action(a, True)
    print(f"reply: {a.response}")
    return 0


def _set_status(args: argparse.Namespace, status: str) -> int:
    store = _store(args)
    a = store.get(args.name)
    if a is None:
        print(f"error: no action named {args.name!r}", file=sys.stderr)
        return 2
    note = "approved by a person" if status == "active" else "rejected by a person"
    hit = [t for t in a.templates if (args.phrasing is None or t.text == args.phrasing)
           and not (status == "active" and t.status == "rejected")]
    if not hit:
        print("error: no matching phrasing", file=sys.stderr)
        return 2
    for t in hit:
        store.set_template_status(a.name, t.text, status, note)
    print(f"{a.name}: {len(hit)} phrasing(s) now {status}. A running gateway picks this up after a restart; "
          "use its admin endpoint to change it live.")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    return _set_status(args, "active")


def cmd_reject(args: argparse.Namespace) -> int:
    return _set_status(args, "rejected")


def cmd_export(args: argparse.Namespace) -> int:
    import yaml

    items = _store(args).export()
    Path(args.file).write_text(yaml.safe_dump({"actions": items}, sort_keys=False, allow_unicode=True))
    print(f"wrote {len(items)} actions to {args.file}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    import yaml

    items = yaml.safe_load(Path(args.file).read_text()).get("actions", [])
    added, skipped = _store(args).import_actions(items, overwrite=args.overwrite)
    print(f"added {added}, skipped {skipped} (already present; use --overwrite to replace)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    s = summarize_log(args.logfile)
    print(json.dumps(s, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aicache", description="Computation-first routing for LLM requests.")
    ap.add_argument("--version", action="version", version=f"aicache {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_config(name: str, help_: str, fn):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config")
        p.add_argument("--embedder", choices=["sentence-transformers", "hashing"],
                       help="override the config's embedder (for example hashing for a quick offline run)")
        p.set_defaults(fn=fn)
        return p

    p = sub.add_parser("init", help="write a starter config")
    p.add_argument("path", nargs="?", default="aicache.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("validate", help="check and lint a config")
    p.add_argument("config")
    p.add_argument("--strict", action="store_true", help="exit 1 if there are warnings")
    p.set_defaults(fn=cmd_validate)

    with_config("test", "run the golden tests in a config", cmd_test)
    p = with_config("run", "route one request", cmd_run)
    p.add_argument("text")
    p = with_config("eval", "measure wrong executions on labeled requests", cmd_eval)
    p.add_argument("--labeled", required=True, help="JSONL: {\"text\": ..., \"expect\": number|\"refuse\"}")
    p.add_argument("--max-wrong", type=int, default=0, help="exit 1 if bad executions exceed this (default 0)")
    p.add_argument("--json", action="store_true")
    p = with_config("calibrate", "sweep min_similarity on labeled requests", cmd_calibrate)
    p.add_argument("--labeled", required=True)
    p = with_config("serve", "run the OpenAI-compatible gateway", cmd_serve)
    p.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (local only)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--allow-unauthenticated", action="store_true",
                   help="allow a non-local host without server.api_key_env (only if the network restricts access)")
    p.add_argument("--log-level", default="info")
    p = with_config("learn", "learn actions from request and answer pairs", cmd_learn)
    p.add_argument("file", help='JSONL: {"text": ..., "answer": ...}')
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("actions", help="list learned actions")
    p.add_argument("config")
    p.add_argument("--status", choices=["probation", "active", "rejected"])
    p.set_defaults(fn=cmd_actions)
    p = sub.add_parser("action", help="show one learned action")
    p.add_argument("config")
    p.add_argument("name")
    p.set_defaults(fn=cmd_action)
    for verb, fn, helptext in (("approve", cmd_approve, "let a learned action answer requests"),
                               ("reject", cmd_reject, "stop a learned action answering requests")):
        p = sub.add_parser(verb, help=helptext)
        p.add_argument("config")
        p.add_argument("name")
        p.add_argument("--phrasing", help="only this phrasing (the exact template text)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("export", help="write learned actions to a YAML file")
    p.add_argument("config")
    p.add_argument("file")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("import", help="load learned actions from a YAML file")
    p.add_argument("config")
    p.add_argument("file")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("stats", help="summarise a decision log (telemetry.log_file)")
    p.add_argument("logfile")
    p.set_defaults(fn=cmd_stats)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:  # a missing optional extra: say which, without a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
