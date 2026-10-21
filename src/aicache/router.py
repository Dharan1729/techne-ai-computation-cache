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

Rollout modes per task: ``active`` computes and answers, ``shadow`` computes
but still lets the LLM answer and records whether they agree, ``off`` ignores
the task.

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

import asyncio
import inspect
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

import numpy as np

from .binding.extractor import extract_numbers
from .embedders import EmbedderLike, SentenceTransformerEmbedder
from .telemetry import Telemetry
from .text import mask_numbers

MAX_GAP = 25  # characters allowed between a keyword and the number it labels
MODES = ("active", "shadow", "off")
_TOKEN_RE = re.compile(r"(?P<ratio>\d+)\s*:\s*1\b|\$?(?P<num>\d[\d,]*(?:\.\d+)?)")
_FIELD_RE = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<spec>[^{}]*))?\}")


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
    text: str | None = None  # the computed value rendered with the task's format
    request_id: str = ""
    latency_ms: float = 0.0
    shadow_value: Any = None  # shadow mode: what the tool computed
    shadow_agrees: bool | None = None  # shadow mode: did the LLM answer match?


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


def render_format(fmt: str, value: Any, args: dict) -> str:
    """Fill ``{value}``, ``{value:.2f}`` and ``{slot_name}`` in a template. Only
    these simple fields are supported, never attribute access, so a template
    cannot reach into objects."""
    scope = {"value": value, **args}

    def sub(m: re.Match) -> str:
        name = m.group("name")
        if name not in scope:
            raise ValueError(f"unknown field {{{name}}} in format {fmt!r}")
        return format(scope[name], m.group("spec") or "")

    return _FIELD_RE.sub(sub, fmt)


@dataclass
class _Task:
    name: str
    fn: Callable[..., Any]
    examples: list[str]
    slots: list[Slot]
    text_slots: list[TextSlot]
    cues: list[str]
    description: str = ""
    mode: str = "active"
    min_similarity: float | None = None
    format: str = "{value}"
    # Learned actions are matched in masked form: ``examples`` already hold masked
    # templates ("What is #% of #?") and the request is masked the same way.
    masked: bool = False
    example_vecs: np.ndarray | None = None  # stored embeddings of ``examples``, if known
    # A custom binder replaces slot matching: it maps request text to BindResult.
    # Learned templates use one that requires the request to BE the template.
    binder: Callable[[str], "BindResult"] | None = None


class ComputationRouter:
    def __init__(self, embedder: EmbedderLike | None = None, min_similarity: float = 0.5,
                 telemetry: Telemetry | None = None, ambiguity_margin: float = 0.03) -> None:
        self.embedder: EmbedderLike = embedder or SentenceTransformerEmbedder()
        self.min_similarity = min_similarity
        # If two tools both bind within this similarity of each other and give
        # different answers, refuse instead of picking one.
        self.ambiguity_margin = ambiguity_margin
        self.telemetry = telemetry or Telemetry()
        self._tasks: dict[str, _Task] = {}
        self._index: dict[bool, np.ndarray] | None = None  # masked? -> example vectors
        self._index_owner: dict[bool, list[str]] = {False: [], True: []}
        self._lock = threading.RLock()

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
        description: str = "",
        mode: str = "active",
        min_similarity: float | None = None,
        format: str = "{value}",
        masked: bool = False,
        example_vecs: np.ndarray | None = None,
        binder: Callable[[str], "BindResult"] | None = None,
        param_names: Sequence[str] = (),
    ) -> None:
        """``cues`` are optional regexes; if given, a request must match at
        least one. Use them when two tools take the same parameters and differ
        only by operation (add / subtract), where embeddings can confuse them.
        ``masked`` and ``example_vecs`` are used for learned actions."""
        if not examples:
            raise ValueError("give at least one example phrasing")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        names = [s.name for s in slots] + [t.name for t in text_slots] + list(param_names)
        if len(set(names)) != len(names):
            raise ValueError(f"task {name!r} has duplicate parameter names")
        for fld in _FIELD_RE.finditer(format):
            if fld.group("name") != "value" and fld.group("name") not in names:
                raise ValueError(f"format refers to unknown field {{{fld.group('name')}}}")
        with self._lock:
            self._tasks[name] = _Task(name, fn, list(examples), list(slots), list(text_slots),
                                      list(cues), description, mode, min_similarity, format,
                                      masked, example_vecs, binder)
            self._index = None

    def remove_task(self, name: str) -> None:
        with self._lock:
            self._tasks.pop(name, None)
            self._index = None

    def task(self, *, examples: Sequence[str], slots: Sequence[Slot],
             text_slots: Sequence[TextSlot] = (), cues: Sequence[str] = (),
             name: str | None = None, **kwargs: Any):
        def deco(fn):
            self.add_task(name or fn.__name__, fn, examples=examples, slots=slots,
                          text_slots=text_slots, cues=cues, **kwargs)
            return fn
        return deco

    def set_mode(self, name: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self._tasks[name].mode = mode
            self._index = None

    @property
    def tasks(self) -> dict[str, _Task]:
        return dict(self._tasks)

    # ---- identification -----------------------------------------------
    def _build(self) -> None:
        """Embed every example of every non-off task, once. Raw and masked tasks
        keep separate indexes because a request is compared to each in its own form."""
        texts: dict[bool, list[str]] = {False: [], True: []}
        owners: dict[bool, list[str]] = {False: [], True: []}
        vecs: dict[bool, list[np.ndarray | None]] = {False: [], True: []}
        for t in self._tasks.values():
            if t.mode == "off":
                continue
            have = t.example_vecs is not None and len(t.example_vecs) == len(t.examples)
            for i, ex in enumerate(t.examples):
                texts[t.masked].append(ex)
                owners[t.masked].append(t.name)
                vecs[t.masked].append(t.example_vecs[i] if have else None)  # type: ignore[index]
        index: dict[bool, np.ndarray] = {}
        for k in (False, True):
            if not texts[k]:
                index[k] = np.zeros((0, 1))
                continue
            missing = [i for i, v in enumerate(vecs[k]) if v is None]
            if missing:
                fresh = self.embedder.encode([texts[k][i] for i in missing])
                for j, i in enumerate(missing):
                    vecs[k][i] = fresh[j]
            index[k] = np.vstack(vecs[k])  # type: ignore[arg-type]
        self._index = index
        self._index_owner = owners

    def rank(self, text: str) -> list[tuple[str, float]]:
        """Every active task with its best example similarity, closest first."""
        with self._lock:
            if self._index is None:
                self._build()
            index, owners = self._index, self._index_owner
        assert index is not None
        best: dict[str, float] = {}
        for masked in (False, True):
            if index[masked].shape[0] == 0:
                continue
            q = self.embedder.encode_one(mask_numbers(text) if masked else text)
            for owner, s in zip(owners[masked], index[masked] @ q):
                if s > best.get(owner, -2.0):
                    best[owner] = float(s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    def identify(self, text: str) -> tuple[str | None, float]:
        """Closest task and its similarity, ignoring ``min_similarity``. Use
        this to calibrate the threshold on your own positive and negative requests."""
        ranked = self.rank(text)
        return ranked[0] if ranked else (None, -1.0)

    def _attempt(self, task: _Task, text: str) -> tuple[bool, dict | str]:
        """Cues and slot binding for one candidate: (True, args) or (False, reason)."""
        if task.cues and not any(re.search(c, text, flags=re.I) for c in task.cues):
            return False, "cue_missing"
        bound = task.binder(text) if task.binder else bind_slots(text, task.slots, task.text_slots)
        return (True, bound.values) if bound.ok else (False, bound.reason)

    # ---- execution ----------------------------------------------------
    def route(self, text: str) -> Result:
        """Decide what to do with ``text`` without calling any fallback. Returns
        source ``"computation"`` (value is ready) or ``"fallback"`` (the caller
        must answer it, with ``reason`` saying why). Every call is recorded."""
        t0 = time.perf_counter()
        ranked = self.rank(text)
        name, sim = ranked[0] if ranked else (None, -1.0)
        reason, args, shadow, mode = "", None, None, "active"
        out: Result | None = None
        if name is None:
            reason = "no_tasks"
        else:
            # Embedding similarity only proposes candidates. Slot binding decides.
            # Try every candidate above its threshold, closest first.
            bound_ok: list[tuple[str, float, dict]] = []
            first_failure = "below_similarity"
            for cand, csim in ranked:
                task = self._tasks.get(cand)
                if task is None:  # removed by a concurrent update (a learned action was reinstalled)
                    continue
                threshold = task.min_similarity if task.min_similarity is not None else self.min_similarity
                if csim < threshold:
                    continue
                ok, res = self._attempt(task, text)
                if ok:
                    bound_ok.append((cand, csim, res))  # type: ignore[arg-type]
                elif first_failure == "below_similarity":
                    first_failure = res  # type: ignore[assignment]
            top = self._tasks.get(name)
            if not bound_ok:
                reason = first_failure
                mode = top.mode if top else "active"
            else:
                name, sim, args = bound_ok[0]
                task = self._tasks.get(name)
                if task is None:
                    reason, mode = "task_removed", "active"
                else:
                    mode = task.mode
                    try:
                        value = task.fn(**args)
                        rival = None
                        for cand, csim, cargs in bound_ok[1:]:
                            if sim - csim >= self.ambiguity_margin:
                                break
                            try:
                                other = self._tasks[cand].fn(**cargs)
                            except Exception:
                                continue
                            if other != value:
                                rival = cand
                                break
                        if rival is not None:
                            reason = f"ambiguous_task:{name}|{rival}"
                        else:
                            rendered = render_format(task.format, value, args)
                            if task.mode == "shadow":
                                shadow, reason = value, "shadow"
                            else:
                                out = Result("computation", value, name, sim, args, text=rendered)
                    except Exception as exc:  # the tool itself refused, e.g. divide by zero
                        reason = f"tool_error:{type(exc).__name__}"
        if out is None:
            out = Result("fallback", None, name, sim, args, reason, shadow_value=shadow)
        out.latency_ms = (time.perf_counter() - t0) * 1000
        out.request_id = self.telemetry.record(
            source=out.source, task=out.task, similarity=sim, reason=out.reason,
            latency_ms=out.latency_ms, text=text, mode=mode, shadow_value=shadow)
        return out

    def note_shadow(self, result: Result, llm_answer: Any) -> bool | None:
        """Shadow mode: compare the tool's value with the LLM's answer (the last
        number in it) and record agreement. Returns None if it cannot compare."""
        if result.shadow_value is None or result.task is None:
            return None
        try:
            expected = float(result.shadow_value)
        except (TypeError, ValueError):
            return None
        nums = extract_numbers(llm_answer if isinstance(llm_answer, str) else str(llm_answer))
        if not nums:
            result.shadow_agrees = False
        else:
            got = nums[-1]
            result.shadow_agrees = abs(got - expected) <= max(1e-3, 1e-3 * abs(expected))
        self.telemetry.record_shadow(result.task, result.shadow_agrees)
        return result.shadow_agrees

    def run(self, text: str, fallback: Callable[[str], Any] | None = None) -> Result:
        r = self.route(text)
        if r.source == "computation":
            return r
        if fallback is None:
            r.source = "refused"
            return r
        r.value = fallback(text)
        self.note_shadow(r, r.value)
        return r

    async def arun(self, text: str,
                   fallback: Callable[[str], Any | Awaitable[Any]] | None = None) -> Result:
        """Async variant. Routing runs in a worker thread so embedding does not
        block the event loop. ``fallback`` may be sync or async."""
        r = await asyncio.to_thread(self.route, text)
        if r.source == "computation":
            return r
        if fallback is None:
            r.source = "refused"
            return r
        out = fallback(text)
        r.value = await out if inspect.isawaitable(out) else out
        self.note_shadow(r, r.value)
        return r
