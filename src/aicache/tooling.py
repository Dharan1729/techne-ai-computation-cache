"""Evaluation and calibration against your own labeled requests.

Labeled file: JSON Lines, one object per line::

    {"text": "What is 15% of 240?", "expect": 36}
    {"text": "What does percent mean?", "expect": "refuse"}

``expect`` is the number the tool should produce, or ``"refuse"`` for requests
that must go to the LLM. The point is to measure *wrong executions* (a tool
answered, and the answer was wrong) on traffic that looks like yours, before
trusting the router.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .router import ComputationRouter
from .telemetry import Telemetry


@dataclass
class Labeled:
    text: str
    expect: float | str  # number or "refuse"


def load_labeled(path: str | Path) -> list[Labeled]:
    out: list[Labeled] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            obj = json.loads(line)
            exp = obj["expect"]
            if exp != "refuse":
                exp = float(exp)
            out.append(Labeled(str(obj["text"]), exp))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: expected {{'text': ..., 'expect': number|'refuse'}} ({exc})") from exc
    return out


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper end of the 95% Wilson interval for a rate of k out of n."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    return min(1.0, (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


@dataclass
class EvalReport:
    n: int = 0
    positives: int = 0
    negatives: int = 0
    computed_correct: int = 0
    wrong_executions: int = 0       # a tool answered and the answer was wrong
    false_triggers: int = 0         # expected refuse, but a tool answered
    missed: int = 0                 # expected a number, but it was refused
    correct_refusals: int = 0
    miss_reasons: dict[str, int] = field(default_factory=dict)
    wrong_examples: list[dict] = field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    @property
    def coverage(self) -> float:
        return self.computed_correct / self.positives if self.positives else 0.0

    @property
    def bad_executions(self) -> int:
        return self.wrong_executions + self.false_triggers

    @property
    def bad_execution_upper_95(self) -> float:
        return wilson_upper(self.bad_executions, self.n)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "positives": self.positives, "negatives": self.negatives,
            "coverage": round(self.coverage, 4), "computed_correct": self.computed_correct,
            "wrong_executions": self.wrong_executions, "false_triggers": self.false_triggers,
            "missed": self.missed, "correct_refusals": self.correct_refusals,
            "bad_execution_rate_upper_95": round(self.bad_execution_upper_95, 4),
            "miss_reasons": self.miss_reasons, "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2), "wrong_examples": self.wrong_examples,
        }


def _close(got: float, want: float) -> bool:
    return abs(got - want) <= max(1e-6, 1e-6 * abs(want))


def evaluate(router: ComputationRouter, items: list[Labeled]) -> EvalReport:
    """Route every item as if all tools were active. Telemetry is not touched."""
    saved_tele, saved_modes = router.telemetry, {n: t.mode for n, t in router.tasks.items()}
    router.telemetry = Telemetry(ring_size=1)
    for n in saved_modes:
        if saved_modes[n] != "off":
            router.set_mode(n, "active")
    rep, lat = EvalReport(), []
    try:
        for it in items:
            r = router.route(it.text)
            lat.append(r.latency_ms)
            rep.n += 1
            computed = r.source == "computation"
            if it.expect == "refuse":
                rep.negatives += 1
                if computed:
                    rep.false_triggers += 1
                    rep.wrong_examples.append({"text": it.text, "kind": "false_trigger", "task": r.task, "value": r.value})
                else:
                    rep.correct_refusals += 1
            else:
                rep.positives += 1
                if computed and _close(float(r.value), float(it.expect)):
                    rep.computed_correct += 1
                elif computed:
                    rep.wrong_executions += 1
                    rep.wrong_examples.append({"text": it.text, "kind": "wrong_value", "task": r.task,
                                               "got": r.value, "expected": it.expect})
                else:
                    rep.missed += 1
                    key = r.reason or "unknown"
                    rep.miss_reasons[key] = rep.miss_reasons.get(key, 0) + 1
    finally:
        router.telemetry = saved_tele
        for n, m in saved_modes.items():
            router.set_mode(n, m)
    if lat:
        s = sorted(lat)
        rep.latency_p50_ms = statistics.median(s)
        rep.latency_p95_ms = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    rep.miss_reasons = dict(sorted(rep.miss_reasons.items(), key=lambda kv: -kv[1]))
    return rep


@dataclass
class CalibrationPoint:
    threshold: float
    coverage: float
    wrong_executions: int
    false_triggers: int


@dataclass
class Calibration:
    points: list[CalibrationPoint]
    lowest_safe: float | None      # lowest threshold with zero wrong executions and zero false triggers
    recommended: float | None      # highest safe threshold that keeps >= 98% of the best coverage


def calibrate(router: ComputationRouter, items: list[Labeled],
              thresholds: list[float] | None = None) -> Calibration:
    """Sweep the global similarity threshold on labeled requests.

    ``lowest_safe`` is where similarity stops mattering on this data. The
    recommendation is the HIGHEST safe threshold that still keeps almost all
    coverage: a higher threshold filters more unrelated traffic you did not
    label, at little cost. Both come from your sample only."""
    thresholds = thresholds or [round(t * 0.05, 2) for t in range(0, 20)]
    cache: dict[str, list] = {}
    real_rank = router.rank

    def memo_rank(text: str):
        if text not in cache:
            cache[text] = real_rank(text)
        return cache[text]

    saved_global = router.min_similarity
    saved_task = {n: t.min_similarity for n, t in router.tasks.items()}
    router.rank = memo_rank  # type: ignore[method-assign]
    for t in router._tasks.values():  # per-task overrides would hide the sweep
        t.min_similarity = None
    points: list[CalibrationPoint] = []
    try:
        for th in thresholds:
            router.min_similarity = th
            rep = evaluate(router, items)
            points.append(CalibrationPoint(th, rep.coverage, rep.wrong_executions, rep.false_triggers))
    finally:
        router.rank = real_rank  # type: ignore[method-assign]
        router.min_similarity = saved_global
        for n, v in saved_task.items():
            router._tasks[n].min_similarity = v
    # "Safe" must still compute something. A threshold that refuses every request has
    # zero wrong answers too, and recommending it would silently disable the tools.
    safe = [p for p in points if p.wrong_executions == 0 and p.false_triggers == 0 and p.coverage > 0]
    if not safe:
        return Calibration(points, None, None)
    best = max(p.coverage for p in safe)
    keep = [p for p in safe if p.coverage >= 0.98 * best]
    return Calibration(points, min(p.threshold for p in safe), max(p.threshold for p in keep))


def summarize_log(path: str | Path) -> dict[str, Any]:
    """Summarise a decision log written by ``telemetry.log_file``."""
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not recs:
        return {"requests": 0}
    by_source: dict[str, int] = {}
    reasons: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for r in recs:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        if r["source"] != "computation" and r.get("reason"):
            k = r["reason"].split(":")[0] if r["reason"].startswith(("tool_error", "ambiguous_task")) else r["reason"]
            reasons[k] = reasons.get(k, 0) + 1
        if r["source"] == "computation" and r.get("task"):
            tasks[r["task"]] = tasks.get(r["task"], 0) + 1
    lat = sorted(r["latency_ms"] for r in recs)
    computed = by_source.get("computation", 0)
    return {
        "requests": len(recs), "by_source": by_source,
        "computed_share": round(computed / len(recs), 4), "llm_calls_avoided": computed,
        "fallback_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "computed_by_task": dict(sorted(tasks.items(), key=lambda kv: -kv[1])),
        "route_latency_ms_p50": round(statistics.median(lat), 2),
        "route_latency_ms_p95": round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))], 2),
        "first_ts": recs[0]["ts"], "last_ts": recs[-1]["ts"],
    }
