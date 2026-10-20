"""ComputationRouter: put your own deterministic tools in front of an LLM.

You register a pure function, a few example phrasings, and a description of
where each parameter appears in a request. For an incoming request the router
(1) finds the closest registered tool by embedding similarity, (2) binds every
number in the text to exactly one parameter, and (3) calls your function. If
any step is unsure it refuses, and your ``fallback`` (usually an LLM call)
handles the request instead. It never guesses.

What this does and does not guarantee, from the benchmarks in
docs/experiment-log.md (EXP-015 to EXP-017):

* A request is executed only if every number in it is claimed by exactly one
  slot and no slot is left empty. Out-of-order phrasings are handled.
* Embedding similarity alone is NOT a safe gate: scores for real and unrelated
  requests overlapped in the finance test. The structural binding above did the
  real rejecting. Still, calibrate ``min_similarity`` on your own traffic.
* A task is only as safe as its slots. Measure wrong executions on your own
  requests before trusting it. Nothing here transfers automatically.

Experimental: the API may change.

Example::

    router = ComputationRouter()

    @router.task(
        examples=["What is X percent of Y?", "Calculate X% of Y."],
        slots=[Slot("pct", after=r"%|percent"), Slot("base", before=r"of")],
    )
    def percent_of(pct, base):
        return base * pct / 100

    result = router.run("What is 15% of 240?", fallback=ask_llm)
    result.source   # "computation", "fallback" or "refused"
    result.value    # 36.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .matching.embedder import Embedder

MAX_GAP = 25  # characters allowed between a keyword and the number it labels
_TOKEN_RE = re.compile(r"(?P<ratio>\d+)\s*:\s*1\b|\$?(?P<num>\d[\d,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class Slot:
    """A numeric parameter.

    ``before`` is a regex for words that usually sit just before the number
    ("entry", "stop at"). ``after`` is a regex for what usually follows it
    ("%", "percent", "shares"). At least one should be given. ``transform``
    converts the raw number (for example dividing a percent by 100) and
    ``validate`` rejects out-of-range values.
    """

    name: str
    before: str = ""
    after: str = ""
    transform: Callable[[float], Any] | None = None
    validate: Callable[[float], bool] | None = None


@dataclass(frozen=True)
class TextSlot:
    """A non-numeric parameter, found by regex.

    ``pattern`` must match exactly one distinct value in the request. If it
    has a capture group, group 1 is the value; otherwise the whole match.
    """

    name: str
    pattern: str


@dataclass
class Result:
    source: str  # "computation", "fallback" or "refused"
    value: Any = None
    task: str | None = None
    similarity: float = -1.0
    args: dict | None = None
    reason: str = ""  # why the request was not computed, when it was not


@dataclass
class BindResult:
    ok: bool
    values: dict = field(default_factory=dict)
    reason: str = ""


def numeric_tokens(text: str) -> list[tuple[tuple[int, int], float]]:
    toks = []
    for m in _TOKEN_RE.finditer(text):
        if m.group("ratio") is not None:  # "3:1" reads as the single number 3
            toks.append((m.span("ratio"), float(m.group("ratio"))))
        else:
            toks.append((m.span("num"), float(m.group("num").replace(",", ""))))
    return toks


def bind_slots(text: str, slots: Sequence[Slot], text_slots: Sequence[TextSlot] = ()) -> BindResult:
    """Bind every number in ``text`` to exactly one slot, or refuse."""
    tokens = numeric_tokens(text)
    # candidate (distance, token index, slot index); a keyword right after the
    # number (distance -1) beats any keyword before it.
    cands: list[tuple[int, int, int]] = []
    for ti, (sp, _v) in enumerate(tokens):
        for si, slot in enumerate(slots):
            if slot.after and re.match(r"\s*(?:" + slot.after + r")", text[sp[1]:], flags=re.I):
                cands.append((-1, ti, si))
            if slot.before:
                for m in re.finditer(slot.before, text[: sp[0]], flags=re.I):
                    gap = text[m.end(): sp[0]]
                    core = gap.strip("$ ")
                    if len(core) <= MAX_GAP and not re.search(r"[\d.;?!]", gap.replace("$", "")):
                        cands.append((len(core), ti, si))
    cands.sort()
    tok_for: dict[int, int] = {}
    dist_for: dict[int, int] = {}
    used: set[int] = set()
    for d, ti, si in cands:
        if si in tok_for or ti in used:
            continue
        tok_for[si], dist_for[si] = ti, d
        used.add(ti)
    for si, slot in enumerate(slots):
        if si not in tok_for:
            return BindResult(False, reason=f"missing:{slot.name}")
    for d, ti, si in cands:  # an unclaimed number as close as a claimed one: ambiguous
        if ti not in used and si in dist_for and d <= dist_for[si]:
            return BindResult(False, reason=f"ambiguous:{slots[si].name}")
    if len(used) != len(tokens):
        return BindResult(False, reason="unclaimed_number")

    values: dict = {}
    for si, slot in enumerate(slots):
        raw = tokens[tok_for[si]][1]
        if slot.validate is not None and not slot.validate(raw):
            return BindResult(False, reason=f"invalid:{slot.name}")
        values[slot.name] = slot.transform(raw) if slot.transform else raw
    for ts in text_slots:
        found = set()
        for m in re.finditer(ts.pattern, text):
            found.add(m.group(1) if m.re.groups else m.group(0))
        if len(found) != 1:
            return BindResult(False, reason=f"text_slot:{ts.name}")
        values[ts.name] = next(iter(found))
    return BindResult(True, values)


@dataclass
class _Task:
    name: str
    fn: Callable[..., Any]
    examples: list[str]
    slots: list[Slot]
    text_slots: list[TextSlot]
    cues: list[str]


class ComputationRouter:
    def __init__(self, embedder: Embedder | None = None, min_similarity: float = 0.5) -> None:
        self.embedder = embedder or Embedder()
        self.min_similarity = min_similarity
        self._tasks: dict[str, _Task] = {}
        self._index: np.ndarray | None = None
        self._index_owner: list[str] = []

    # ---- registration -------------------------------------------------
    def add_task(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        examples: Sequence[str],
        slots: Sequence[Slot],
        text_slots: Sequence[TextSlot] = (),
        cues: Sequence[str] = (),
    ) -> None:
        """``cues`` are optional regexes; if given, a request must match at
        least one. Use them when two tools take the same parameters and differ
        only by operation (add / subtract), where embeddings can confuse them."""
        if not examples:
            raise ValueError("give at least one example phrasing")
        self._tasks[name] = _Task(name, fn, list(examples), list(slots), list(text_slots), list(cues))
        self._index = None

    def task(self, *, examples: Sequence[str], slots: Sequence[Slot],
             text_slots: Sequence[TextSlot] = (), cues: Sequence[str] = (), name: str | None = None):
        def deco(fn):
            self.add_task(name or fn.__name__, fn, examples=examples, slots=slots,
                          text_slots=text_slots, cues=cues)
            return fn
        return deco

    # ---- identification -----------------------------------------------
    def _build(self) -> None:
        texts, owners = [], []
        for t in self._tasks.values():
            for ex in t.examples:
                texts.append(ex)
                owners.append(t.name)
        self._index = self.embedder.encode(texts)
        self._index_owner = owners

    def identify(self, text: str) -> tuple[str | None, float]:
        """Closest task and its similarity, ignoring ``min_similarity``. Use
        this to calibrate the threshold on your own positive and negative requests."""
        if not self._tasks:
            return None, -1.0
        if self._index is None:
            self._build()
        sims = self._index @ self.embedder.encode_one(text)  # type: ignore[operator]
        i = int(np.argmax(sims))
        return self._index_owner[i], float(sims[i])

    # ---- execution ----------------------------------------------------
    def run(self, text: str, fallback: Callable[[str], Any] | None = None) -> Result:
        name, sim = self.identify(text)
        reason, args = "", None
        if name is None:
            reason = "no_tasks"
        elif sim < self.min_similarity:
            reason = "below_similarity"
        else:
            task = self._tasks[name]
            if task.cues and not any(re.search(c, text, flags=re.I) for c in task.cues):
                reason = "cue_missing"
            else:
                bound = bind_slots(text, task.slots, task.text_slots)
                if not bound.ok:
                    reason = bound.reason
                else:
                    args = bound.values
                    try:
                        return Result("computation", task.fn(**args), name, sim, args)
                    except Exception as exc:  # the tool itself refused, for example divide by zero
                        reason = f"tool_error:{type(exc).__name__}"
        if fallback is not None:
            return Result("fallback", fallback(text), name, sim, args, reason)
        return Result("refused", None, name, sim, args, reason)
