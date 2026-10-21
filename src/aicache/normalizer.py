"""Step 2: a small LLM identifies the action behind a request.

Removing the variable values is done by code, because a model asked to copy a
sentence while abstracting its numbers often rewrites it. So the request is
first turned into a skeleton by masking:

    request   What is 15% of 240?
    skeleton  What is {n1}% of {n2}?

and the model is shown the request, the answer the main LLM gave, and the
skeleton, and asked to identify the operation:

    {"action": "percent_of",
     "names": {"n1": "pct", "n2": "base"},
     "expression": "base * pct / 100",
     "response": "{pct}% of {base} is {value}"}

Nothing it says is trusted. ``verify`` checks, deterministically, that

* the names are valid and every one is used by a safe expression,
* running the expression on the request's own values reproduces the answer the
  main LLM gave,
* the template, which code built from the request, aligns with it exactly.

A proposal that fails any check is dropped and the reason is recorded. One that
passes is still only a candidate: it must agree with the LLM on further requests
before it may answer (see ``learner``).
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .binding.extractor import extract_numbers
from .router import render_format
from .safe_expr import ExprError, compile_expr, free_names
from .text import NUMBER, align, check_template

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ACTION_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
MAX_PARAMS = 5

SYSTEM = (
    "You identify the operation behind a request that contains numbers. Reply with a single JSON object and "
    "nothing else. Keys: action (a short snake_case name for the operation), names (give each placeholder "
    "such as n1 a short snake_case role name), expression (an arithmetic expression over those role names that "
    "gives the answer; use + - * / ** and the functions abs, round, min, max, sqrt), response (a short sentence "
    "using {value} for the result and the role names)."
)

EXAMPLES = [
    ("What is 15% of 240?", "36", "What is {n1}% of {n2}?",
     {"action": "percent_of", "names": {"n1": "pct", "n2": "base"},
      "expression": "base * pct / 100", "response": "{pct}% of {base} is {value}"}),
    ("If I invest $1000 at 5 percent for 3 years, what is the final amount?", "1157.63",
     "If I invest ${n1} at {n2} percent for {n3} years, what is the final amount?",
     {"action": "compound_interest", "names": {"n1": "principal", "n2": "rate", "n3": "years"},
      "expression": "principal * (1 + rate / 100) ** years", "response": "It grows to {value}"}),
    ("Subtract 4 from 19.", "15", "Subtract {n1} from {n2}.",
     {"action": "subtract", "names": {"n1": "y", "n2": "x"}, "expression": "x - y", "response": "{value}"}),
]


def skeleton(request: str) -> tuple[str, list[float]]:
    """The request with each number replaced by {n1}, {n2}, ... and the values in order."""
    values: list[float] = []

    def sub(m: re.Match) -> str:
        values.append(float(m.group(0).lstrip("$").replace(",", "")))
        return ("$" if m.group(0).startswith("$") else "") + "{n%d}" % len(values)

    return re.sub(NUMBER, sub, request), values


@dataclass
class Known:
    """An existing action, shown to the model so it can reuse the name."""
    name: str
    template: str
    expression: str
    params: tuple[str, ...] = ()


@dataclass
class Proposal:
    action: str
    names: dict[str, str]
    expression: str
    response: str
    raw: str = ""


@dataclass
class Verified:
    action: str
    template: str
    expression: str
    response: str
    values: dict[str, float]

    @property
    def params(self) -> list[str]:
        return list(self.values)


def build_prompt(request: str, answer: str, known: Sequence[Known]) -> list[dict[str, str]]:
    shots = [f"Request: {r}\nAnswer: {a}\nSkeleton: {sk}\nJSON: {json.dumps(o)}" for r, a, sk, o in EXAMPLES]
    hint = ""
    if known:
        lines = "\n".join(f"- {k.name}: {k.template}  =>  {k.expression}" for k in known)
        hint = ("Operations already known. If the request is the same operation as one of these, use its action "
                f"name and the same role names:\n{lines}\n\n")
    sk, _ = skeleton(request)
    user = "\n\n".join(shots) + f"\n\n{hint}Request: {request}\nAnswer: {answer[:300].strip()}\nSkeleton: {sk}\nJSON:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_proposal(text: str) -> Proposal | str:
    """The proposal in ``text`` (which may wrap the JSON in prose), or a reason."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return "no_json"
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return "bad_json"
    if not isinstance(d, dict):
        return "bad_json"
    for k in ("action", "expression"):
        if not isinstance(d.get(k), str) or not d[k].strip():
            return f"missing_{k}"
    names = d.get("names")
    if not isinstance(names, dict) or not names or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()):
        return "missing_names"
    resp = d.get("response")
    return Proposal(d["action"].strip(), {k: v.strip() for k, v in names.items()}, d["expression"].strip(),
                    resp.strip() if isinstance(resp, str) and resp.strip() else "{value}", text)


class Normalizer:
    """Calls any OpenAI-compatible chat endpoint (Ollama, vLLM, OpenAI, ...).

    ``complete`` can replace the network call, for tests: it receives the chat
    messages and returns the model's text.
    """

    def __init__(self, base_url: str = "", model: str = "", api_key: str | None = None,
                 timeout_s: float = 60.0, json_mode: bool = True, transport: Any = None,
                 complete: Callable[[list[dict[str, str]]], str] | None = None) -> None:
        self.model = model
        self.json_mode = json_mode
        self._complete = complete
        self._client = None
        if complete is None:
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                        timeout=timeout_s, transport=transport)

    def propose(self, request: str, answer: str, known: Sequence[Known] = ()) -> Proposal | str:
        messages = build_prompt(request, answer, known)
        try:
            text = self._complete(messages) if self._complete else self._call(messages)
        except Exception as exc:  # network, timeout, bad status
            return f"normalizer_error:{type(exc).__name__}"
        return parse_proposal(text)

    def _call(self, messages: list[dict[str, str]]) -> str:
        assert self._client is not None
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 300}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


# ---- verification ------------------------------------------------------------

def _close(got: float, want: float) -> bool:
    return abs(got - want) <= max(0.011, 1e-3 * abs(want))


def answer_number(answer: str) -> float | None:
    nums = extract_numbers(answer)
    return nums[-1] if nums else None


_PROBES = [(2.0, 3.0, 5.0, 7.0, 11.0), (11.0, 4.0, 9.0, 6.0, 2.0), (1.5, 8.0, 2.5, 12.0, 3.0), (30.0, 7.0, 3.0, 20.0, 9.0)]


def equivalent_under_renaming(new_expr: str, new_names: Sequence[str], old_expr: str,
                              old_names: Sequence[str]) -> dict[str, str] | None:
    """If the two expressions compute the same function once the new parameter
    names are matched to the old ones in some order, return that matching
    (new name -> old name), else None. This is how a new phrasing of a known
    operation is recognised even when the model names its roles differently."""
    if len(new_names) != len(old_names) or len(new_names) > MAX_PARAMS:
        return None
    try:
        fn_new, fn_old = compile_expr(new_expr, new_names), compile_expr(old_expr, old_names)
    except ExprError:
        return None
    best: tuple[int, dict[str, str]] | None = None
    for perm in itertools.permutations(range(len(old_names))):
        ok = True
        for probe in _PROBES:
            old_vals = {n: probe[i] for i, n in enumerate(old_names)}
            new_vals = {new_names[i]: old_vals[old_names[perm[i]]] for i in range(len(new_names))}
            try:
                if not _close(float(fn_new(**new_vals)), float(fn_old(**old_vals))):
                    ok = False
                    break
            except (ArithmeticError, ValueError, TypeError):
                ok = False
                break
        if ok:
            mapping = {new_names[i]: old_names[perm[i]] for i in range(len(new_names))}
            same = sum(1 for n, o in mapping.items() if n == o)  # prefer keeping names that already agree
            if best is None or same > best[0]:
                best = (same, mapping)
    return best[1] if best else None


def verify(request: str, answer: str, p: Proposal) -> Verified | str:
    """Accept the proposal or return the reason it was rejected."""
    if not ACTION_RE.match(p.action):
        return "bad_action_name"
    sk, vals = skeleton(request)
    if not vals:
        return "no_numbers"
    if len(vals) > MAX_PARAMS:
        return "too_many_numbers"
    placeholders = [f"n{i + 1}" for i in range(len(vals))]
    if set(p.names) != set(placeholders):
        return "names_do_not_cover_numbers"
    new = [p.names[k] for k in placeholders]
    if len(set(new)) != len(new) or not all(NAME_RE.match(n) for n in new):
        return "bad_names"
    template = sk
    for k in placeholders:  # rename n1 -> pct, etc., in the skeleton
        template = template.replace("{" + k + "}", "{" + p.names[k] + "}")
    bad = check_template(template)
    if bad:
        return bad
    values = align(template, request)
    if values is None:  # pragma: no cover - the skeleton is built from the request
        return "template_misaligned"
    try:
        used = free_names(p.expression)
    except SyntaxError:
        return "expression_syntax"
    if not used <= set(new):
        return "expression_unknown_name"
    if used != set(new):
        return "parameter_unused"
    try:
        result = float(compile_expr(p.expression, new)(**values))
    except (ExprError, ArithmeticError, ValueError, TypeError):
        return "expression_error"
    if not math.isfinite(result):
        return "expression_error"
    want = answer_number(answer)
    if want is None:
        return "answer_has_no_number"
    if not _close(result, want):
        return "expression_disagrees_with_answer"
    response = p.response
    try:
        render_format(response, result, values)
    except (ValueError, KeyError, TypeError):
        response = "{value}"
    return Verified(p.action, template, p.expression, response, values)
