"""Product core: safe expressions, embedders, telemetry, router modes."""

import asyncio
import json
import math

import numpy as np
import pytest

from aicache import ComputationRouter, Slot
from aicache.embedders import HashingEmbedder, OpenAICompatibleEmbedder, make_embedder
from aicache.router import render_format
from aicache.safe_expr import ExprError, compile_expr, free_names
from aicache.telemetry import Telemetry


# ---- safe_expr --------------------------------------------------------------

def test_expr_arithmetic_and_functions():
    f = compile_expr("round(base * pct / 100, 2) + max(1, 2)", ["base", "pct"])
    assert f(base=240, pct=15) == 38.0
    assert compile_expr("sqrt(x) + pi - pi", ["x"])(x=9) == 3.0
    assert compile_expr("x if x > 0 else -x", ["x"])(x=-4) == 4


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "x.__class__",
    "[i for i in range(3)]",
    "lambda: 1",
    "open('f')",
    "x[0]",
    "'a' + 'b'",
    "exec('1')",
    "(1).real",
    "unknown + 1",
    "f(x)",
    "max(key=abs)",
])
def test_expr_rejects_anything_unsafe_at_compile_time(bad):
    with pytest.raises(ExprError):
        compile_expr(bad, ["x"])


def test_expr_limits():
    with pytest.raises(ExprError):
        compile_expr("1+" * 300 + "1", [])
    with pytest.raises(ExprError):
        compile_expr("x ** 100000", ["x"])(x=2)
    with pytest.raises(ZeroDivisionError):
        compile_expr("1 / x", ["x"])(x=0)


def test_expr_free_names():
    assert free_names("a * b + sqrt(c) + pi") == {"a", "b", "c"}


# ---- embedders --------------------------------------------------------------

def test_hashing_embedder_is_normalised_deterministic_and_sees_overlap():
    e = HashingEmbedder()
    v = e.encode(["what is 15 percent of 240", "what is 20 percent of 90", "tell me a joke"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.array_equal(v, e.encode(["what is 15 percent of 240", "what is 20 percent of 90", "tell me a joke"]))
    assert v[0] @ v[1] > v[0] @ v[2]


def test_openai_compatible_embedder_parses_response():
    import httpx

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "m" and request.url.path.endswith("/embeddings")
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 3.0]}, {"index": 0, "embedding": [4.0, 0.0]}]})

    e = OpenAICompatibleEmbedder("http://x/v1", "m", transport=httpx.MockTransport(handler))
    v = e.encode(["a", "b"])
    assert np.allclose(v, [[1.0, 0.0], [0.0, 1.0]])


def test_make_embedder_validates():
    assert isinstance(make_embedder("hashing"), HashingEmbedder)
    with pytest.raises(ValueError):
        make_embedder("nope")
    with pytest.raises(ValueError):
        make_embedder("openai")


# ---- telemetry --------------------------------------------------------------

def test_telemetry_does_not_store_text_by_default(tmp_path):
    t = Telemetry(log_file=str(tmp_path / "d.jsonl"))
    t.record(source="computation", task="a", similarity=0.9, reason="", latency_ms=3.0, text="secret text")
    line = (tmp_path / "d.jsonl").read_text()
    assert "secret" not in line and json.loads(line)["task"] == "a"
    assert "text" not in t.recent()[0]


def test_telemetry_text_logging_is_opt_in():
    t = Telemetry(log_text=True)
    t.record(source="fallback", task=None, similarity=0.1, reason="below_similarity", latency_ms=1, text="hello")
    assert t.recent()[0]["text"] == "hello"


def test_telemetry_counters_and_prometheus():
    t = Telemetry()
    t.record(source="computation", task="a", similarity=1, reason="", latency_ms=2, text="x")
    t.record(source="fallback", task="a", similarity=1, reason="missing:y", latency_ms=700, text="z")
    t.record_shadow("a", True)
    s = t.snapshot()
    assert s["requests_total"] == 2 and s["llm_calls_avoided"] == 1 and s["computed_share"] == 0.5
    assert s["fallback_reasons"] == {"missing:y": 1} and s["shadow"] == {"a:agree": 1}
    p = t.prometheus()
    assert 'aicache_requests_total{source="computation"} 1' in p
    assert 'aicache_route_latency_ms_bucket{le="+Inf"} 2' in p
    assert 'aicache_shadow_total{task="a",result="agree"} 1' in p


def test_telemetry_ring_is_bounded():
    t = Telemetry(ring_size=3)
    for i in range(10):
        t.record(source="fallback", task=None, similarity=0, reason="r", latency_ms=1, text=str(i))
    assert len(t.recent(100)) == 3


# ---- router modes -----------------------------------------------------------

def _router(**kw):
    r = ComputationRouter(embedder=HashingEmbedder(), min_similarity=0.2, **kw)
    r.add_task("percent_of", lambda pct, base: base * pct / 100,
               examples=["What is X percent of Y?", "Calculate X% of Y.", "X percent of Y"],
               slots=[Slot("pct", after=r"%|percent"), Slot("base", before=r"of")],
               format="{value:.2f}")
    return r


def test_route_does_not_call_a_fallback_and_renders_text():
    r = _router().route("What is 15% of 240?")
    assert r.source == "computation" and r.value == 36.0 and r.text == "36.00"
    assert r.request_id and r.latency_ms >= 0


def test_route_marks_fallback_with_reason():
    r = _router().route("What is 15% of that?")
    assert r.source == "fallback" and r.reason == "missing:base" and r.value is None


def test_shadow_mode_computes_but_lets_the_llm_answer_and_records_agreement():
    router = _router()
    router.set_mode("percent_of", "shadow")
    r = router.run("What is 15% of 240?", fallback=lambda t: "It is 36.")
    assert r.source == "fallback" and r.reason == "shadow" and r.shadow_value == 36.0
    assert r.shadow_agrees is True and r.value == "It is 36."
    r2 = router.run("What is 15% of 240?", fallback=lambda t: "I think it's 40.")
    assert r2.shadow_agrees is False
    assert router.telemetry.snapshot()["shadow"] == {"percent_of:agree": 1, "percent_of:disagree": 1}


def test_off_mode_ignores_the_task():
    router = _router()
    router.set_mode("percent_of", "off")
    r = router.route("What is 15% of 240?")
    assert r.source == "fallback" and r.reason == "no_tasks"


def test_per_task_threshold_overrides_the_default():
    router = ComputationRouter(embedder=HashingEmbedder(), min_similarity=0.0)
    router.add_task("t", lambda x: x, examples=["double x"], slots=[Slot("x", before="double")],
                    min_similarity=0.999)
    assert router.route("double 4").reason == "below_similarity"


def test_format_is_validated_at_registration_and_cannot_reach_attributes():
    router = ComputationRouter(embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        router.add_task("t", lambda x: x, examples=["a"], slots=[Slot("x", before="a")], format="{nope}")
    assert render_format("{value:.1f} of {x}", 2.25, {"x": 9}) == "2.2 of 9"
    # An attribute-style field is not a simple field, so it is left as literal
    # text: nothing is looked up or evaluated.
    assert render_format("{value.__class__}", 1, {}) == "{value.__class__}"


def test_duplicate_parameter_names_rejected():
    router = ComputationRouter(embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        router.add_task("t", lambda x: x, examples=["a"], slots=[Slot("x", before="a"), Slot("x", after="b")])


def test_arun_supports_sync_and_async_fallbacks():
    router = _router()

    async def go():
        a = await router.arun("What is 15% of that?", fallback=lambda t: "sync")
        async def afb(t):
            return "async"
        b = await router.arun("What is 15% of that?", fallback=afb)
        c = await router.arun("What is 15% of 240?", fallback=afb)
        return a.value, b.value, c.source

    assert asyncio.run(go()) == ("sync", "async", "computation")


def test_router_is_safe_under_concurrent_use():
    import concurrent.futures as cf
    router = _router()
    with cf.ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(router.route, ["What is %d%% of 200?" % i for i in range(1, 40)]))
    assert all(r.source == "computation" for r in rs)
    assert [r.value for r in rs] == [i * 2.0 for i in range(1, 40)]
    assert router.telemetry.snapshot()["requests_total"] == 39
