"""Declarative config, golden tests, linting, and multi-candidate routing."""

from pathlib import Path

import pytest

from aicache import ComputationRouter, Slot
from aicache.config import (ConfigError, build_router, lint_config, load_config, run_golden_tests)
from aicache.embedders import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "tools": [{
        "name": "percent_of",
        "examples": ["What is X percent of Y?", "Calculate X% of Y.", "X percent of Y"],
        "slots": [{"name": "pct", "after": "%|percent", "min": 0, "max": 100},
                  {"name": "base", "before": "of"}],
        "compute": "base * pct / 100",
        "format": "{value:.2f}",
        "tests": [{"text": "What is 15% of 240?", "expect": 36},
                  {"text": "What is 15% of that?", "expect": "refuse"}],
    }]
}


def _cfg(**tool_over):
    import copy
    d = copy.deepcopy(BASE)
    d["tools"][0].update(tool_over)
    return d


def test_valid_config_builds_and_passes_its_own_tests():
    cfg = load_config(BASE)
    router = build_router(cfg, embedder=HashingEmbedder())
    assert [o.passed for o in run_golden_tests(cfg, router)] == [True, True]


def test_load_from_yaml_string_and_path(tmp_path):
    import yaml
    text = yaml.safe_dump(BASE)
    assert load_config(text).tools[0].name == "percent_of"
    p = tmp_path / "c.yaml"
    p.write_text(text)
    assert load_config(p).tools[0].name == "percent_of"
    assert load_config(str(p)).tools[0].name == "percent_of"


def test_errors_are_collected_and_readable():
    bad = _cfg(slots=[{"name": "1bad", "before": "of"}, {"name": "b"}], compute="x +")
    with pytest.raises(ConfigError) as e:
        load_config(bad)
    msg = str(e.value)
    assert "tools.0.slots.0.name" in msg and "tools.0.slots.1" in msg


@pytest.mark.parametrize("over,needle", [
    ({"compute": "__import__('os')"}, "compute"),
    ({"compute": "base * unknown"}, "unknown name"),
    ({"python": "a.b:c"}, "exactly one"),
    ({"compute": None, "python": None}, "exactly one"),
    ({"examples": []}, "examples"),
    ({"format": "{nope}"}, "unknown field"),
    ({"cues": ["("]}, "regular expression"),
    ({"mode": "loud"}, "mode"),
    ({"surprise": 1}, "surprise"),
])
def test_bad_tools_are_rejected_at_load_time(over, needle):
    with pytest.raises(ConfigError) as e:
        load_config(_cfg(**over))
    assert needle in str(e.value)


def test_duplicate_tool_names_and_empty_tools_rejected():
    d = _cfg()
    d["tools"].append(d["tools"][0])
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(d)
    with pytest.raises(ConfigError):
        load_config({"tools": []})
    with pytest.raises(ConfigError, match="mapping"):
        load_config("- just\n- a list\n")


def test_slot_min_max_integer_and_scale():
    d = _cfg(
        slots=[{"name": "pct", "after": "%|percent", "min": 0, "max": 100, "scale": 0.01},
               {"name": "base", "before": "of", "integer": True}],
        compute="base * pct")
    d["min_similarity"] = 0.0
    cfg = load_config(d)
    router = build_router(cfg, embedder=HashingEmbedder())
    assert router.route("What is 15% of 240?").value == pytest.approx(36.0)
    assert router.route("What is 150% of 240?").reason == "invalid:pct"
    assert router.route("What is 15% of 240.5?").reason == "invalid:base"


def _hyp(pct, base):
    return (pct ** 2 + base ** 2) ** 0.5


def test_python_tool_is_imported():
    d = _cfg(compute=None, python="tests.test_config:_hyp")
    d["min_similarity"] = 0.0
    router = build_router(load_config(d), embedder=HashingEmbedder())
    assert router.route("What is 3% of 4?").value == 5.0
    bad = load_config(_cfg(compute=None, python="no_such_module:fn"))
    with pytest.raises(ConfigError, match="cannot import"):
        build_router(bad, embedder=HashingEmbedder())


def test_golden_tests_report_failures_and_do_not_touch_telemetry_or_modes():
    d = _cfg(tests=[{"text": "What is 15% of 240?", "expect": 99}, {"text": "What is 15% of 240?", "expect": "refuse"}],
             mode="shadow")
    cfg = load_config(d)
    router = build_router(cfg, embedder=HashingEmbedder())
    out = run_golden_tests(cfg, router)
    assert [o.passed for o in out] == [False, False]
    assert router.telemetry.snapshot()["requests_total"] == 0
    assert router.tasks["percent_of"].mode == "shadow"


def test_lint_flags_missing_tests_few_examples_and_same_signature_tools():
    d = _cfg(examples=["X percent of Y"], tests=[])
    d["tools"].append({"name": "other", "examples": ["a", "b", "c"], "tests": [{"text": "q", "expect": "refuse"}],
                       "slots": [{"name": "a", "before": "x"}, {"name": "b", "before": "y"}], "compute": "a + b"})
    msgs = [f"{i.tool}: {i.message}" for i in lint_config(load_config(d))]
    assert any("fewer than 3" in m for m in msgs)
    assert any("no positive tests" in m for m in msgs)
    assert any("same number of parameters" in m and m.startswith("percent_of") for m in msgs)


def test_shipped_example_config_is_clean_and_every_golden_test_passes(embedder):
    cfg = load_config(ROOT / "examples" / "finance_tools.yaml")
    assert lint_config(cfg) == []
    router = build_router(cfg, embedder=embedder)
    failed = [o for o in run_golden_tests(cfg, router) if not o.passed]
    assert failed == [], [(o.text, o.detail) for o in failed]


# ---- multi-candidate routing ------------------------------------------------

def _two_tools(fn_b):
    r = ComputationRouter(embedder=HashingEmbedder(), min_similarity=0.0, ambiguity_margin=0.05)
    slots = lambda: [Slot("x", before="combine"), Slot("y", before="and")]
    r.add_task("a", lambda x, y: x + y, examples=["combine x and y"], slots=slots())
    r.add_task("b", fn_b, examples=["combine x and y"], slots=slots())
    return r


def test_validation_chooses_among_candidates():
    r = ComputationRouter(embedder=HashingEmbedder(), min_similarity=0.0)
    r.add_task("near", lambda a, b, c: a + b + c, examples=["total of three amounts a b c"],
               slots=[Slot("a", before="of"), Slot("b", before="and"), Slot("c", before="plus")])
    r.add_task("far", lambda p, q: p * q, examples=["multiply p by q"],
               slots=[Slot("p", before="multiply"), Slot("q", before="by")])
    res = r.route("multiply 6 by 7")
    assert res.source == "computation" and res.task == "far" and res.value == 42


def test_two_tools_that_both_bind_and_disagree_are_refused_as_ambiguous():
    r = _two_tools(lambda x, y: x * y)
    res = r.route("combine 3 and 4")
    assert res.source == "fallback" and res.reason.startswith("ambiguous_task")
    assert r.telemetry.snapshot()["fallback_reasons"] == {"ambiguous_task": 1}


def test_two_tools_that_agree_are_not_ambiguous():
    r = _two_tools(lambda x, y: y + x)
    res = r.route("combine 3 and 4")
    assert res.source == "computation" and res.value == 7
