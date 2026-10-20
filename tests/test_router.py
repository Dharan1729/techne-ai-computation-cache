"""ComputationRouter: slot binding and routing. Uses a tiny deterministic fake
embedder so these tests need no model download."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aicache import ComputationRouter, Slot, TextSlot
from aicache.router import bind_slots


class FakeEmbedder:
    """Bag-of-words hashing into 64 dimensions, L2-normalised."""

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in t.lower().replace("?", " ").replace(".", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)

    def encode_one(self, t):
        return self.encode([t])[0]


PCT_SLOTS = [
    Slot("pct", after=r"%|percent"),
    Slot("base", before=r"of"),
]


def _router(**kw):
    r = ComputationRouter(embedder=FakeEmbedder(), **kw)
    r.add_task(
        "percent_of", lambda pct, base: base * pct / 100,
        examples=["What is X percent of Y?", "Calculate X% of Y."], slots=PCT_SLOTS,
    )
    return r


# ---- bind_slots -------------------------------------------------------------

def test_binds_numbers_in_either_order():
    a = bind_slots("What is 15% of 240?", PCT_SLOTS)
    b = bind_slots("Of 240, what is 15 percent?", PCT_SLOTS)
    assert a.ok and a.values == {"pct": 15.0, "base": 240.0}
    assert b.ok and b.values == {"pct": 15.0, "base": 240.0}


def test_refuses_an_unclaimed_number():
    r = bind_slots("What is 15% of 240 in 2024?", PCT_SLOTS)
    assert not r.ok and r.reason in ("unclaimed_number", "ambiguous:base")


def test_refuses_a_missing_slot():
    r = bind_slots("What is 15% of that?", PCT_SLOTS)
    assert not r.ok and r.reason == "missing:base"


def test_ratio_notation_is_one_number():
    slots = [Slot("m", after=r"r\b|:\s*1")]
    assert bind_slots("a 4:1 target", slots).values == {"m": 4.0}


def test_validate_and_transform():
    slots = [Slot("rate", after=r"%", transform=lambda v: v / 100, validate=lambda v: 0 < v <= 100)]
    assert bind_slots("fee is 5%", slots).values == {"rate": 0.05}
    assert bind_slots("fee is 500%", slots).reason == "invalid:rate"


def test_text_slot_needs_exactly_one_match():
    ts = [TextSlot("sym", r"\b([A-Z]{2,5})\b")]
    assert bind_slots("buy 5 now", [Slot("n", before=r"buy")], ts).reason == "text_slot:sym"
    assert bind_slots("buy 5 AAPL", [Slot("n", before=r"buy")], ts).values == {"n": 5.0, "sym": "AAPL"}
    assert not bind_slots("buy 5 AAPL or MSFT", [Slot("n", before=r"buy")], ts).ok


# ---- routing ----------------------------------------------------------------

def test_computes_when_everything_binds():
    r = _router().run("What is 15% of 240?", fallback=lambda t: "LLM")
    assert r.source == "computation" and r.value == 36.0 and r.task == "percent_of"


def test_falls_back_when_binding_fails():
    r = _router().run("What is 15% of that?", fallback=lambda t: "LLM")
    assert r.source == "fallback" and r.value == "LLM" and r.reason == "missing:base"


def test_refuses_without_a_fallback():
    r = _router().run("What is 15% of that?")
    assert r.source == "refused" and r.value is None


def test_falls_back_below_similarity():
    r = _router(min_similarity=0.99).run("What is 15% of 240?", fallback=lambda t: "LLM")
    assert r.source == "fallback" and r.reason == "below_similarity"


def test_cues_stop_a_wrong_operation():
    router = ComputationRouter(embedder=FakeEmbedder(), min_similarity=0.0)
    two = [Slot("x", before=r"add|plus|minus|subtract|and|from"), Slot("y", before=r"and|plus|minus|from|\bx\b")]
    router.add_task("add", lambda x, y: x + y, examples=["add x and y"], slots=two, cues=[r"\bplus\b|\badd\b|\+"])
    r = router.run("What is 5 minus 3?", fallback=lambda t: "LLM")
    assert r.source == "fallback" and r.reason == "cue_missing"


def test_tool_errors_fall_back_instead_of_raising():
    router = ComputationRouter(embedder=FakeEmbedder(), min_similarity=0.0)
    router.add_task("div", lambda a, b: a / b, examples=["divide a by b"],
                    slots=[Slot("a", before=r"divide"), Slot("b", before=r"by")])
    r = router.run("divide 4 by 0", fallback=lambda t: "LLM")
    assert r.source == "fallback" and r.reason == "tool_error:ZeroDivisionError"


def test_decorator_registers_and_returns_the_function():
    router = ComputationRouter(embedder=FakeEmbedder(), min_similarity=0.0)

    @router.task(examples=["double x"], slots=[Slot("x", before=r"double")])
    def double(x):
        return 2 * x

    assert double(3) == 6
    assert router.run("double 21").value == 42


def test_identify_reports_similarity_for_calibration():
    name, sim = _router().identify("What is 15% of 240?")
    assert name == "percent_of" and 0 < sim <= 1.0


# ---- the shipped example, with the real embedding model -----------------------

def test_shipped_example_computes_in_any_order_and_refuses_when_incomplete(embedder):
    spec = importlib.util.spec_from_file_location(
        "position_size_router", Path(__file__).resolve().parents[1] / "examples" / "position_size_router.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.router.embedder = embedder
    mod.router._index = None
    r = mod.router.run("Size my TSLA trade: stop 436.81, entry 476.92, risking 0.5 percent of $50000.",
                       fallback=lambda t: "LLM")
    assert r.source == "computation" and r.args["risk_percent"] == 0.5  # stays a percent
    assert r.value == round(50000 * 0.5 / 100 / (476.92 - 436.81), 2)
    r2 = mod.router.run("How many AAPL shares can I buy with entry 150 and stop 140?", fallback=lambda t: "LLM")
    assert r2.source == "fallback"
