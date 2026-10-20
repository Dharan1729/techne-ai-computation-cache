"""Real-app integration: aicache vs. an LLM tool-router, on wshobson/financial-chat.

financial-chat (https://github.com/wshobson/financial-chat) routes every
risk-math request through an LLM tool-selection call, even though
``calculate_r_multiples`` and ``calculate_position_size`` are pure
arithmetic. This script imports those two tools UNMODIFIED from a
checkout of that repo (a few compatibility shims for a newer installed
langchain; see ``_install_shims``) and compares:

  A. cache path  : embedding identification -> slot extraction ->
                   validation gate -> call the real tool directly
  B. router path : a local LLM (Ollama) is shown the real tools' names,
                   descriptions and argument schemas and must choose a
                   tool and fill its arguments (a stand-in for the
                   repo's ``llm.bind_tools(...)`` step)
  C. hybrid      : A first; whenever A rejects the request, fall back to B

Methodology (chosen so the result can survive hostile review):
  * Phrasing templates are split: TRAIN templates are used for registry
    example phrasings and threshold calibration; the numbers reported are
    on TEST templates that neither the registry nor the calibration saw.
  * Negative queries (requests that should NOT trigger either tool) are
    included, because a router/cache that fires on everything would
    otherwise look perfect.
  * Correctness = the tool's real output string equals the output of the
    same tool called with ground-truth arguments. Not a proxy.
  * Latency is steady-state (model and embedder warmed first).

Usage:
    python -m benchmarks.real_app.financial_chat_integration \
        --repo-path /path/to/financial-chat --model llama3 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Compatibility shims so financial-chat's tools import under a newer langchain
# --------------------------------------------------------------------------


def _install_shims(repo_path: Path) -> None:
    sys.path.insert(0, str(repo_path))
    import langchain.agents as la
    import pydantic.v1 as pv1
    from langchain_core.tools import tool as lc_tool

    if not hasattr(la, "tool"):
        la.tool = lc_tool  # type: ignore[attr-defined]
    sys.modules.setdefault("langchain_core.pydantic_v1", pv1)
    # Only data-fetching tools use these; the two math tools do not.
    sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
    utils = types.ModuleType("app.tools.utils")

    def fetch_stock_data(*a, **k):  # pragma: no cover - never exercised
        raise RuntimeError("market data disabled in this benchmark")

    utils.fetch_stock_data = fetch_stock_data  # type: ignore[attr-defined]
    sys.modules["app.tools.utils"] = utils


# --------------------------------------------------------------------------
# Request generation: train / test template split, plus negatives
# --------------------------------------------------------------------------

SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "SHOP"]

R_TEMPLATES = [  # first 3 = TRAIN, rest = TEST
    "Calculate the {r}R profit target for {sym}, entry ${entry}, stop ${stop}",
    "I'm buying {sym} at ${entry} with a stop at ${stop}. Where is my {r}R target?",
    "What are the R multiple targets for {sym} if I enter at {entry} and my stop loss is {stop}? Use a {r}R multiple.",
    "Give me a {r}:1 reward-to-risk profit target on {sym}: in at {entry}, out at {stop} if wrong.",
    "{sym} trade idea, stop {stop}, entry {entry}. Compute the {r}R take-profit.",
    "Using a stop price of ${stop} and an entry price of ${entry} on {sym}, what's the profit target at {r} times my risk?",
    "how far is a {r}R target on {sym} when I get in at {entry} and bail at {stop}",
    "R-multiple calc: ticker {sym}, entry {entry}, stop {stop}, multiple {r}",
]
P_TEMPLATES = [
    "How many shares of {sym} should I buy with a ${acct} account risking {rp}% per trade? Entry ${entry}, stop ${stop}.",
    "Position size for {sym}: account ${acct}, risk {rp}%, entry {entry}, stop {stop}.",
    "I have ${acct} and want to risk {rp}% on {sym}. I'd enter at ${entry} with a stop at ${stop}, what size?",
    "With a {rp}% risk limit on a ${acct} portfolio, how many {sym} shares can I take if I buy at {entry} and stop out at {stop}?",
    "Size my {sym} trade: stop {stop}, entry {entry}, risking {rp} percent of ${acct}.",
    "Entry price ${entry}, stop price ${stop}, {sym}. Account size is ${acct}; risk percent {rp}. Shares?",
    "how big should my {sym} position be? {acct} account, {rp}% risk, buy {entry}, stop {stop}",
    "Given a ${acct} account and a risk of {rp}% of it, determine the share count for {sym}, in at {entry}, protective stop at {stop}.",
]
N_TRAIN = 3

# Fresh holdout phrasings. Written BEFORE extractor v2 was run against
# anything, so v2 cannot have been tuned on them. (Templates 3..7 above are
# "dev": v1's failures on them were observed and v2 was built in response.)
R_HOLDOUT = [
    "If I go long {sym} at {entry} and my stop sits at {stop}, what does a {r}R exit look like?",
    "{sym}: buy {entry}, stop-loss {stop}. I want the {r}R level.",
    "Can you work out my target for a {r} to 1 payoff on {sym}? Entry is {entry} and I'd stop out at {stop}.",
    "stop {stop}, entry {entry}, {sym}, {r}R please",
]
P_HOLDOUT = [
    "I'm risking {rp}% of my ${acct} account on {sym}. Buy at {entry}, stop at {stop}. How many shares?",
    "{sym} position sizing: ${acct} account, {rp} percent risk, entry ${entry}, stop ${stop}",
    "My stop on {sym} is {stop} and I'm getting in at {entry}; with {acct} in the account and {rp}% risk, what's the size?",
    "how many {sym} shares if account size = {acct}, risk percent = {rp}, entry price = {entry}, stop price = {stop}",
]

# Disjoint calibration negatives vs. test negatives.
NEG_CAL = [
    "What is the news sentiment on AAPL this week?",
    "Show me a price chart for TSLA over the last year.",
    "Which stocks are the top gainers today?",
    "What does an R multiple mean in trading?",
    "Explain what position sizing is.",
    "Is NVDA overvalued right now?",
    "What is the 50-day moving average for MSFT?",
    "Tell me about the company behind AMZN.",
    "Give me the RSI for AMD.",
    "What's the relative strength of META against the S&P 500?",
    "Hello, what can you do?",
    "Why do traders use stop losses?",
    "My account is $50,000, which sectors look strong?",
    "TSLA is trading at 250, is that a good level?",
    "What are the top losers today?",
]
NEG_TEST = [
    "What are analysts saying about GOOGL?",
    "Plot NFLX against its 200-day average.",
    "Define risk percent in portfolio management.",
    "What was the 52 week high for SHOP?",
    "Should I hold AMD through earnings?",
    "How do R multiples relate to expectancy?",
    "Compare AAPL and MSFT fundamentals.",
    "I have $25,000 to invest, any ideas?",
    "What is a good risk to reward ratio?",
    "NVDA gapped up at 480, what does that mean?",
    "Which stocks have the strongest momentum?",
    "Can you explain how stop losses get hunted?",
    "Summarize the latest TSLA earnings call.",
    "What's the support and resistance for META?",
    "Is a 2% risk per trade too aggressive in general?",
    "Who is the CEO of AMZN?",
    "What happened to the market today?",
    "List the biggest losers in tech.",
    "Teach me about technical analysis.",
    "How much is 3 times 7?",
]


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


@dataclass
class Case:
    kind: str  # "r_multiples" | "position_size" | "negative"
    text: str
    args: dict | None = None
    template: int = -1
    split: str = ""


def make_positive(rng: random.Random, kind: str, template_idx: int, split: str) -> Case:
    sym = rng.choice(SYMBOLS)
    entry = round(rng.uniform(20, 480), 2)
    stop = round(entry * rng.uniform(0.85, 0.97), 2)
    if kind == "r_multiples":
        r = rng.choice([2, 3, 4, 5])
        text = R_TEMPLATES[template_idx].format(sym=sym, r=r, entry=_fmt(entry), stop=_fmt(stop))
        args = dict(symbol=sym, entry_price=entry, stop_price=stop, risk_multiple=r)
    else:
        acct = rng.choice([10000, 25000, 50000, 75000, 100000, 250000, 500000])
        rp = rng.choice([0.5, 1, 1.5, 2, 3])
        text = P_TEMPLATES[template_idx].format(
            sym=sym, acct=acct, rp=_fmt(rp), entry=_fmt(entry), stop=_fmt(stop)
        )
        args = dict(
            symbol=sym, entry_price=entry, stop_price=stop, account_size=float(acct), risk_percent=float(rp)
        )
    return Case(kind, text, args, template_idx, split)


def build_cases(seed: int, per_template: int) -> tuple[list[Case], list[Case]]:
    """Returns (calibration cases, evaluation cases).

    Evaluation cases carry split="dev" (templates 3..7: extractor v1's
    failures on these were observed, v2 was built in response) or
    split="holdout" (fresh templates, never used to build anything), plus
    split="test" negatives.
    """
    rng = random.Random(seed)
    calib: list[Case] = []
    test: list[Case] = []
    for kind, templates, hold in (
        ("r_multiples", R_TEMPLATES, R_HOLDOUT),
        ("position_size", P_TEMPLATES, P_HOLDOUT),
    ):
        for ti in range(len(templates)):
            for _ in range(per_template):
                if ti < N_TRAIN:
                    calib.append(make_positive(rng, kind, ti, "calib"))
                else:
                    test.append(make_positive(rng, kind, ti, "dev"))
        for hi in range(len(hold)):
            for _ in range(per_template):
                c = make_positive_from(rng, kind, hold[hi])
                c.template = 100 + hi
                c.split = "holdout"
                test.append(c)
    calib += [Case("negative", t, split="calib") for t in NEG_CAL]
    test += [Case("negative", t, split="test") for t in NEG_TEST]
    return calib, test


def make_positive_from(rng: random.Random, kind: str, template: str) -> Case:
    sym = rng.choice(SYMBOLS)
    entry = round(rng.uniform(20, 480), 2)
    stop = round(entry * rng.uniform(0.85, 0.97), 2)
    if kind == "r_multiples":
        r = rng.choice([2, 3, 4, 5])
        text = template.format(sym=sym, r=r, entry=_fmt(entry), stop=_fmt(stop))
        args = dict(symbol=sym, entry_price=entry, stop_price=stop, risk_multiple=r)
    else:
        acct = rng.choice([10000, 25000, 50000, 75000, 100000, 250000, 500000])
        rp = rng.choice([0.5, 1, 1.5, 2, 3])
        text = template.format(sym=sym, acct=acct, rp=_fmt(rp), entry=_fmt(entry), stop=_fmt(stop))
        args = dict(symbol=sym, entry_price=entry, stop_price=stop, account_size=float(acct), risk_percent=float(rp))
    return Case(kind, text, args)


# --------------------------------------------------------------------------
# Path A: slot extraction + validation gate
# --------------------------------------------------------------------------

NUM = r"\$?\s?(?P<n>\d[\d,]*(?:\.\d+)?)"
TOKEN_RE = re.compile(r"(?P<ratio>\d+)\s*:\s*1\b|\$?(?P<num>\d[\d,]*(?:\.\d+)?)")


def numeric_tokens(text: str) -> list[tuple[tuple[int, int], float]]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        if m.group("ratio") is not None:
            toks.append((m.span("ratio"), float(m.group("ratio"))))
            # the ":1" is part of the ratio notation, not a stray number
        else:
            toks.append((m.span("num"), float(m.group("num").replace(",", ""))))
    return toks


SLOT_PATTERNS: dict[str, list[str]] = {
    "entry_price": [
        r"(?:entry(?: price)?|enter(?:ing)?|buy(?:ing)?|get(?:ting)? in|in at|long)\D{0,25}?" + NUM,
        NUM + r"\s*entry",
    ],
    "stop_price": [
        r"(?:stop(?:[- ]loss)?(?: price)?|stops? out|stopped out|bail(?:s)?(?: out)?|out at)\D{0,25}?" + NUM,
    ],
    "account_size": [
        r"(?:account(?: size)?|portfolio|have|capital|balance)\D{0,25}?" + NUM,
        NUM + r"\s*(?:account|portfolio)",
    ],
    "risk_percent": [
        NUM + r"\s*(?:%|percent|pct)",
        r"risk(?:ing)?(?: percent| limit)?\D{0,15}?" + NUM,
    ],
    "risk_multiple": [
        NUM + r"\s*(?:r\b|:\s*1|times)",
        r"multiple\D{0,10}?" + NUM,
    ],
}
SLOTS_FOR = {
    "r_multiples": ["entry_price", "stop_price", "risk_multiple"],
    "position_size": ["entry_price", "stop_price", "account_size", "risk_percent"],
}
TICKER_STOP = {"R", "USD", "ETF", "RSI", "SMA", "CEO", "AND", "THE", "I"}


@dataclass
class Extraction:
    ok: bool
    args: dict | None = None
    reason: str = ""


def extract_args(kind: str, text: str) -> Extraction:
    tokens = numeric_tokens(text)
    claimed: dict[str, tuple[int, int]] = {}
    values: dict[str, float] = {}
    for slot in SLOTS_FOR[kind]:
        spans = set()
        for pat in SLOT_PATTERNS[slot]:
            for m in re.finditer(pat, text, flags=re.I):
                spans.add(m.span("n"))
        # map regex group spans back to numeric tokens
        hit = [(sp, v) for (sp, v) in tokens if sp in spans]
        if not hit:
            return Extraction(False, reason=f"missing:{slot}")
        if len({sp for sp, _ in hit}) > 1:
            return Extraction(False, reason=f"ambiguous:{slot}")
        claimed[slot] = hit[0][0]
        values[slot] = hit[0][1]
    if len(set(claimed.values())) != len(claimed):
        return Extraction(False, reason="token_claimed_twice")
    leftover = [sp for sp, _ in tokens if sp not in set(claimed.values())]
    if leftover:
        return Extraction(False, reason="unclaimed_number")
    syms = [s for s in re.findall(r"\b[A-Z]{2,5}\b", text) if s not in TICKER_STOP]
    if len(set(syms)) != 1:
        return Extraction(False, reason="symbol")
    args: dict = {"symbol": syms[0]}
    args.update(values)
    # domain sanity (cheap, deterministic)
    if args["entry_price"] <= 0 or args["stop_price"] <= 0 or args["entry_price"] == args["stop_price"]:
        return Extraction(False, reason="price_domain")
    if kind == "r_multiples":
        m = args["risk_multiple"]
        if m != int(m) or not 1 <= m <= 20:
            return Extraction(False, reason="multiple_domain")
        args["risk_multiple"] = int(m)
    else:
        if args["account_size"] <= 0 or not 0 < args["risk_percent"] <= 100:
            return Extraction(False, reason="risk_domain")
    return Extraction(True, args)


# ---- extractor v2: every number is assigned to its NEAREST keyword -----------
# Built after seeing v1 reject many "dev" phrasings (v1's fixed windows let a
# far-away keyword claim the wrong number, which the gate then refused).

PREFIX_KEYS = {
    "entry_price": r"entry(?: price)?|enter(?:ing)?|buy(?:ing)?|get(?:ting)? in|in at|long",
    "stop_price": r"stop(?:[- ]loss)?(?: price)?|stops? out|stopped out|bail(?:s)?(?: out)?|out at",
    "account_size": r"account(?: size)?|portfolio|capital|balance|have|(?:%|percent|pct)\s+of",
    "risk_percent": r"risk(?:ing)?(?: percent| limit)?",
    "risk_multiple": r"multiple",
}
SUFFIX_KEYS = {
    "entry_price": r"\s*entry",
    "account_size": r"\s*(?:in\s+(?:the|my)\s+)?(?:account|portfolio)",
    "risk_percent": r"\s*(?:%|percent|pct)",
    "risk_multiple": r"\s*(?:r\b|:\s*1|times)",
}
MAX_GAP = 25


def extract_args_v2(kind: str, text: str) -> Extraction:
    tokens = numeric_tokens(text)
    slots = SLOTS_FOR[kind]
    cands: list[tuple[int, int, str]] = []  # (distance, token index, slot)
    for ti, (sp, _v) in enumerate(tokens):
        # make the span cover a leading "$" / space so gaps measure real text
        for slot in slots:
            # suffix keyword immediately after the number -> strongest signal
            if slot in SUFFIX_KEYS and re.match(SUFFIX_KEYS[slot], text[sp[1]:], flags=re.I):
                cands.append((-1, ti, slot))
            # prefix keyword before the number, with no digit or sentence break in between
            for m in re.finditer(PREFIX_KEYS[slot], text[: sp[0]], flags=re.I):
                gap = text[m.end(): sp[0]]
                if len(gap.strip("$ ")) <= MAX_GAP and not re.search(r"[\d.;?!]", gap.replace("$", "")):
                    cands.append((len(gap.strip("$ ")), ti, slot))
    cands.sort()
    tok_for: dict[str, int] = {}
    used: set[int] = set()
    dist_for: dict[str, int] = {}
    for d, ti, slot in cands:
        if slot in tok_for or ti in used:
            continue
        tok_for[slot] = ti
        dist_for[slot] = d
        used.add(ti)
    for slot in slots:
        if slot not in tok_for:
            return Extraction(False, reason=f"missing:{slot}")
    # competition check: an unassigned token that is just as close to an assigned slot -> ambiguous
    for d, ti, slot in cands:
        if ti not in used and slot in dist_for and d <= dist_for[slot]:
            return Extraction(False, reason=f"ambiguous:{slot}")
    if len(used) != len(tokens):
        return Extraction(False, reason="unclaimed_number")
    values = {slot: tokens[tok_for[slot]][1] for slot in slots}
    return _finalize(kind, text, values)


def _finalize(kind: str, text: str, values: dict) -> Extraction:
    syms = [s for s in re.findall(r"\b[A-Z]{2,5}\b", text) if s not in TICKER_STOP]
    if len(set(syms)) != 1:
        return Extraction(False, reason="symbol")
    args: dict = {"symbol": syms[0]}
    args.update(values)
    if args["entry_price"] <= 0 or args["stop_price"] <= 0 or args["entry_price"] == args["stop_price"]:
        return Extraction(False, reason="price_domain")
    if kind == "r_multiples":
        m = args["risk_multiple"]
        if m != int(m) or not 1 <= m <= 20:
            return Extraction(False, reason="multiple_domain")
        args["risk_multiple"] = int(m)
    else:
        if args["account_size"] <= 0 or not 0 < args["risk_percent"] <= 100:
            return Extraction(False, reason="risk_domain")
    return Extraction(True, args)


# Registry for the (unmodified) TaskIdentifier: generic example phrasings
# written from the TRAIN style only. They are not copies of any template.
@dataclass
class _Spec:
    example_phrasings: list[str] = field(default_factory=list)


REGISTRY = {
    "r_multiples": _Spec([
        "Calculate the profit target and stop for a trade using R multiples.",
        "Where is my R-multiple target given my entry price and stop price?",
        "What is the 3R profit target if I enter at a price with a stop loss?",
        "Compute reward-to-risk targets for a stock trade from entry and stop.",
    ]),
    "position_size": _Spec([
        "How many shares should I buy given my account size and risk percent?",
        "Calculate the position size for a trade with an entry price and a stop price.",
        "What size position can I take risking a percent of my account?",
        "Work out the share count for a trade from account size, risk and stop.",
    ]),
}


# --------------------------------------------------------------------------
# Path B: LLM router over the real tools' schemas
# --------------------------------------------------------------------------


def _sig(t) -> str:
    ty = {"string": "string", "number": "number", "integer": "integer"}
    parts = [f"{k}: {ty.get(v.get('type'), 'any')} ({v.get('description', '')})" for k, v in t.args.items()]
    return "(" + "; ".join(parts) + ")"


def router_prompt(tools: dict, question: str) -> str:
    descs = [f"- {name}{_sig(t)}: {t.description}" for name, t in tools.items()]
    first = next(iter(tools))
    return (
        "You are the tool-selection step of a trading assistant. Decide whether one of "
        "the tools below should be called for the user's message and fill in its arguments.\n"
        "Tools:\n" + "\n".join(descs) + "\n\n"
        'Reply with JSON only: {"tool": "<name or none>", "args": {<argument name>: <plain value>, ...}}. '
        'Argument values are plain strings or numbers, not objects. '
        'Use {"tool": "none", "args": {}} if no tool applies.\n'
        f'Example: {{"tool": "{first}", "args": {{"symbol": "XYZ"}}}}\n\n'
        f"User message: {question}"
    )


# --------------------------------------------------------------------------


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True)
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-template", type=int, default=8)
    ap.add_argument("--output", default="results/real_app_financial_chat.json")
    a = ap.parse_args()

    _install_shims(Path(a.repo_path))
    from app.tools.risk_management import (  # type: ignore
        calculate_position_size,
        calculate_r_multiples,
        calculate_technical_stops,
    )
    from aicache.llm.ollama import OllamaLLM
    from aicache.matching.task_identification import TaskIdentifier

    real = {"r_multiples": calculate_r_multiples, "position_size": calculate_position_size}
    router_tools = {
        "calculate_technical_stops": calculate_technical_stops,
        "calculate_r_multiples": calculate_r_multiples,
        "calculate_position_size": calculate_position_size,
    }
    name_to_kind = {"calculate_r_multiples": "r_multiples", "calculate_position_size": "position_size"}

    def call_real(kind: str, args: dict) -> str:
        return real[kind].invoke(args)

    calib, test = build_cases(a.seed, a.per_template)
    expected: dict[int, str] = {}
    for c in calib + test:
        if c.kind != "negative":
            expected[id(c)] = call_real(c.kind, c.args)  # type: ignore[arg-type]

    ident = TaskIdentifier(registry=REGISTRY, threshold=0.0)  # we apply the threshold ourselves
    ident.identify("warm up")

    def identify(text: str) -> tuple[str, float]:
        r = ident.identify(text)
        return r.task_type, r.similarity  # type: ignore[return-value]

    # ---- calibrate on TRAIN-template positives + calibration negatives ONLY
    cal_pos_cases = [c for c in calib if c.kind != "negative"]
    cal_pos = [identify(c.text) for c in cal_pos_cases]
    cal_neg = [identify(c.text)[1] for c in calib if c.kind == "negative"]
    emb_only_thr = max(cal_neg) + 1e-6  # what an embedding-only gate would need (zero calib false triggers)
    cand = [t / 100 for t in range(10, 90)]
    # ranking threshold: highest t that still admits >=95% of calibration positives
    thr = max(t for t in cand if sum(1 for (tk, sim), c in zip(cal_pos, cal_pos_cases) if sim >= t and tk == c.kind) / len(cal_pos) >= 0.95)
    sweep = []
    for t in [round(x * 0.02, 2) for x in range(10, 40)]:
        sweep.append({
            "threshold": t,
            "calib_positive_accepted_correctly": sum(1 for (tk, sim), c in zip(cal_pos, cal_pos_cases) if sim >= t and tk == c.kind) / len(cal_pos),
            "calib_negative_false_trigger": sum(1 for sim in cal_neg if sim >= t) / len(cal_neg),
        })
    print(f"calibrated identification threshold: {thr:.3f}")

    llm = OllamaLLM(model_name=a.model, options={"temperature": 0, "seed": 0}, response_format="json")
    llm.complete(router_prompt(router_tools, "What is the position size for AAPL?"))

    def run_cache(c: Case, extractor) -> tuple[str, str, float, str | None, float]:
        t0 = time.perf_counter()
        task, sim = identify(c.text)
        status, reason, out = "rejected", "below_threshold", None
        if sim >= thr:
            ex = extractor(task, c.text)
            if ex.ok:
                out = call_real(task, ex.args)  # type: ignore[arg-type]
                status, reason = "executed", ""
            else:
                reason = ex.reason
        return status, reason, (time.perf_counter() - t0) * 1000, out, sim

    rows = []
    for c in test:
        row = {"kind": c.kind, "split": c.split, "template": c.template, "text": c.text}
        exp = expected.get(id(c))
        neg = c.kind == "negative"

        def judge(status: str, out: str | None) -> tuple[bool, bool]:
            """(correct, wrong_execution)"""
            if neg:
                return (status != "executed", status == "executed")
            return (status == "executed" and out == exp, status == "executed" and out != exp)

        for tag, extractor in (("a1", extract_args), ("a2", extract_args_v2)):
            st, rs, ms, out, sim = run_cache(c, extractor)
            row[f"{tag}_status"], row[f"{tag}_reason"], row[f"{tag}_ms"], row[f"{tag}_sim"] = st, rs, ms, sim
            row[f"{tag}_correct"], row[f"{tag}_wrong_exec"] = judge(st, out)
            row[f"{tag}_out"] = out

        resp = llm.complete(router_prompt(router_tools, c.text))
        b_status, b_out, b_reason = "no_tool", None, ""
        try:
            d = json.loads(resp.text)
            tname = d.get("tool", "none")
            if tname in router_tools:
                if tname in name_to_kind:
                    b_out = router_tools[tname].invoke(d["args"])
                    b_status = "executed"
                else:
                    b_status = "executed_other_tool"
        except Exception as exc:
            b_status, b_reason = "error", type(exc).__name__
        row.update(b_status=b_status, b_reason=b_reason, b_ms=resp.latency_ms)
        if neg:
            row["b_correct"], row["b_wrong_exec"] = b_status in ("no_tool",), b_status.startswith("executed")
        else:
            row["b_correct"] = b_status == "executed" and b_out == exp
            row["b_wrong_exec"] = b_status.startswith("executed") and b_out != exp
        # hybrid = v2 cache first, router only when the cache rejects
        if row["a2_status"] == "executed":
            row["h_correct"], row["h_wrong_exec"], row["h_ms"] = row["a2_correct"], row["a2_wrong_exec"], row["a2_ms"]
        else:
            row["h_correct"], row["h_wrong_exec"] = row["b_correct"], row["b_wrong_exec"]
            row["h_ms"] = row["a2_ms"] + resp.latency_ms
        rows.append(row)

    def agg(rs, p):
        n = len(rs)
        lat = [r[f"{p}_ms"] for r in rs]
        return {
            "n": n, "correct": sum(r[f"{p}_correct"] for r in rs),
            "wrong_executions": sum(r[f"{p}_wrong_exec"] for r in rs),
            "executed": sum(1 for r in rs if r.get(f"{p}_status", "executed") == "executed") if p in ("a1", "a2") else None,
            "p50_ms": statistics.median(lat), "p95_ms": pct(lat, 0.95), "mean_ms": statistics.fmean(lat),
        }

    def block(rs):
        return {p: agg(rs, p) for p in ("a1", "a2", "b", "h")}

    pos = [r for r in rows if r["kind"] != "negative"]
    summary = {
        "model": a.model, "seed": a.seed, "temperature": 0, "identification_threshold": thr,
        "embedding_only_threshold_for_zero_calib_false_triggers": emb_only_thr,
        "n_calibration": len(calib),
        "dev_positives": block([r for r in pos if r["split"] == "dev"]),
        "holdout_positives": block([r for r in pos if r["split"] == "holdout"]),
        "all_positives": block(pos),
        "negatives": block([r for r in rows if r["kind"] == "negative"]),
        "legend": {"a1": "cache, extractor v1 (first pass)", "a2": "cache, extractor v2 (nearest-keyword)",
                   "b": "LLM router (llama3 over the real tool schemas)", "h": "hybrid: cache v2, router on reject"},
        "a2_reject_reasons": {k: sum(1 for r in rows if r["a2_reason"] == k) for k in sorted({r["a2_reason"] for r in rows if r["a2_reason"]})},
        "b_status": {k: sum(1 for r in rows if r["b_status"] == k) for k in sorted({r["b_status"] for r in rows})},
        "calibration_sweep": sweep,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration_sweep"}, indent=2, default=str))


if __name__ == "__main__":
    main()
