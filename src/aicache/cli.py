"""The ``aicache`` command line.

    aicache init [path]                       write a starter config
    aicache validate CONFIG [--strict]        check the config and lint it
    aicache test CONFIG                       run the golden tests inside the config
    aicache run CONFIG "text"                 route one request and show the decision
    aicache eval CONFIG --labeled FILE        wrong executions on your labeled requests
    aicache calibrate CONFIG --labeled FILE   sweep min_similarity
    aicache serve CONFIG                      OpenAI-compatible gateway (needs the 'serve' extra)
    aicache stats LOGFILE                     summarise a decision log

Exit codes: 0 success, 1 a check failed (tests, lint with --strict, wrong
executions above --max-wrong), 2 bad usage or an invalid config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
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
embedder: {type: sentence-transformers}   # or `hashing` for a no-download trial
# upstream:                                # needed for `aicache serve`
#   base_url: http://localhost:11434/v1
#   model: llama3
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
    cfg = load_config(Path(args.config))
    emb = None
    if getattr(args, "embedder", None):
        emb = make_embedder(args.embedder)
    return cfg, build_router(cfg, embedder=emb)


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


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:
        print(f"serve needs the 'serve' extra: pip install 'ai-computation-cache[serve]' ({exc})", file=sys.stderr)
        return 2
    cfg = load_config(Path(args.config))
    emb = make_embedder(args.embedder) if args.embedder else None
    app = create_app(cfg, embedder=emb)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
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
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--log-level", default="info")
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
