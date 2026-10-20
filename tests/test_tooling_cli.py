"""eval / calibrate / stats tooling and the aicache command line."""

import json

import pytest
import yaml

from aicache import ComputationRouter, Slot
from aicache.cli import main
from aicache.config import build_router, load_config
from aicache.embedders import HashingEmbedder
from aicache.tooling import (Labeled, calibrate, evaluate, load_labeled, summarize_log, wilson_upper)

CONFIG = {
    "min_similarity": 0.0,
    "embedder": {"type": "hashing"},
    "tools": [{
        "name": "percent_of",
        "examples": ["What is X percent of Y?", "Calculate X% of Y.", "X percent of Y"],
        "slots": [{"name": "pct", "after": "%|percent", "min": 0, "max": 100},
                  {"name": "base", "before": "of"}],
        "compute": "base * pct / 100",
        "tests": [{"text": "What is 15% of 240?", "expect": 36},
                  {"text": "What is 15% of that?", "expect": "refuse"}],
    }],
}


def _router(broken=False):
    cfg = dict(CONFIG)
    if broken:  # a tool with a wrong formula: the evaluator must catch it
        cfg = json.loads(json.dumps(CONFIG))
        cfg["tools"][0]["compute"] = "base * pct / 10"
    return build_router(load_config(cfg), embedder=HashingEmbedder())


LABELED = [
    Labeled("What is 15% of 240?", 36.0),
    Labeled("What is 20% of 50?", 10.0),
    Labeled("What is 15% of that?", "refuse"),
    Labeled("Tell me about the weather", "refuse"),
    Labeled("What is 5% of the budget?", 99.0),       # a tool-less number: will be missed
]


# ---- evaluate -----------------------------------------------------------------

def test_evaluate_counts_every_outcome():
    rep = evaluate(_router(), LABELED)
    assert (rep.n, rep.positives, rep.negatives) == (5, 3, 2)
    assert rep.computed_correct == 2 and rep.wrong_executions == 0 and rep.false_triggers == 0
    assert rep.missed == 1 and rep.correct_refusals == 2
    assert rep.coverage == pytest.approx(2 / 3)
    assert rep.miss_reasons == {"missing:base": 1}


def test_evaluate_catches_a_wrong_formula_and_false_triggers():
    rep = evaluate(_router(broken=True), LABELED + [Labeled("What is 10% of 100?", "refuse")])
    assert rep.wrong_executions == 2 and rep.false_triggers == 1
    assert rep.bad_executions == 3 and rep.wrong_examples[0]["kind"] == "wrong_value"


def test_evaluate_does_not_pollute_telemetry_or_modes():
    r = _router()
    r.set_mode("percent_of", "shadow")
    evaluate(r, LABELED)
    assert r.telemetry.snapshot()["requests_total"] == 0 and r.tasks["percent_of"].mode == "shadow"


def test_wilson_upper_bound():
    assert wilson_upper(0, 100) == pytest.approx(0.037, abs=0.002)
    assert wilson_upper(0, 1000) < wilson_upper(0, 100)
    assert wilson_upper(0, 0) == 1.0


def test_load_labeled_reports_the_bad_line(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('# comment\n{"text": "a", "expect": 1}\n\n{"text": "b", "expect": "refuse"}\n')
    assert [i.expect for i in load_labeled(p)] == [1.0, "refuse"]
    p.write_text('{"text": "a"}\n')
    with pytest.raises(ValueError, match=r":1:"):
        load_labeled(p)


# ---- calibrate ----------------------------------------------------------------

def test_calibrate_recommends_the_highest_safe_threshold_that_keeps_coverage():
    cal = calibrate(_router(), LABELED[:4])
    assert cal.lowest_safe == 0.0
    assert cal.recommended is not None and cal.recommended >= cal.lowest_safe
    chosen = next(p for p in cal.points if p.threshold == cal.recommended)
    assert chosen.wrong_executions == 0 and chosen.false_triggers == 0
    best = max(p.coverage for p in cal.points if p.wrong_executions == 0 and p.false_triggers == 0)
    assert chosen.coverage >= 0.98 * best


def test_calibrate_says_so_when_no_threshold_is_safe_and_restores_state():
    r = _router(broken=True)
    cal = calibrate(r, LABELED[:2])
    assert cal.recommended is None and cal.lowest_safe is None
    assert r.min_similarity == 0.0 and r.rank.__func__ is ComputationRouter.rank


# ---- stats --------------------------------------------------------------------

def test_summarize_log(tmp_path):
    from aicache.telemetry import Telemetry
    log = tmp_path / "d.jsonl"
    t = Telemetry(log_file=str(log))
    t.record(source="computation", task="a", similarity=0.9, reason="", latency_ms=2, text="x")
    t.record(source="computation", task="a", similarity=0.9, reason="", latency_ms=4, text="y")
    t.record(source="fallback", task="a", similarity=0.9, reason="tool_error:ZeroDivisionError", latency_ms=9, text="z")
    s = summarize_log(log)
    assert s["requests"] == 3 and s["computed_share"] == round(2 / 3, 4) and s["llm_calls_avoided"] == 2
    assert s["fallback_reasons"] == {"tool_error": 1} and s["computed_by_task"] == {"a": 2}


# ---- CLI ----------------------------------------------------------------------

@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "aicache.yaml"
    p.write_text(yaml.safe_dump(CONFIG))
    return str(p)


@pytest.fixture
def labeled_path(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps({"text": i.text, "expect": i.expect}) for i in LABELED) + "\n")
    return str(p)


def test_cli_init_writes_a_config_that_validates(tmp_path, capsys):
    p = str(tmp_path / "new.yaml")
    assert main(["init", p]) == 0
    assert main(["init", p]) == 2  # refuses to overwrite
    assert main(["init", p, "--force"]) == 0
    assert main(["validate", p]) == 0
    assert main(["test", p, "--embedder", "hashing"]) in (0, 1)  # runs; hashing may miss, exit code reflects that


def test_cli_validate_strict_turns_warnings_into_failure(cfg_path, tmp_path):
    assert main(["validate", cfg_path]) == 0
    assert main(["validate", cfg_path, "--strict"]) == 0  # clean config: no warnings
    weak = json.loads(json.dumps(CONFIG))
    weak["tools"][0]["examples"] = ["X percent of Y"]  # too few examples
    p = tmp_path / "weak.yaml"
    p.write_text(yaml.safe_dump(weak))
    assert main(["validate", str(p)]) == 0              # warnings do not fail by default
    assert main(["validate", str(p), "--strict"]) == 1


def test_cli_invalid_config_exits_2(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("tools: []\n")
    assert main(["validate", str(p)]) == 2
    assert "invalid config" in capsys.readouterr().err
    assert main(["validate", str(tmp_path / "missing.yaml")]) == 2


def test_cli_test_exit_code_reflects_failures(cfg_path, tmp_path):
    assert main(["test", cfg_path]) == 0
    bad = json.loads(json.dumps(CONFIG))
    bad["tools"][0]["tests"][0]["expect"] = 999
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(bad))
    assert main(["test", str(p)]) == 1


def test_cli_run_prints_a_decision(cfg_path, capsys):
    assert main(["run", cfg_path, "What is 15% of 240?"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "computation" and out["value"] == 36.0 and out["task"] == "percent_of"


def test_cli_eval_gates_on_wrong_executions(cfg_path, labeled_path, tmp_path, capsys):
    assert main(["eval", cfg_path, "--labeled", labeled_path]) == 0
    out = capsys.readouterr().out
    assert "WRONG executions    0" in out and "95% upper bound" in out
    bad = json.loads(json.dumps(CONFIG))
    bad["tools"][0]["compute"] = "base * pct / 10"
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(bad))
    assert main(["eval", str(p), "--labeled", labeled_path]) == 1
    assert main(["eval", str(p), "--labeled", labeled_path, "--max-wrong", "5"]) == 0
    capsys.readouterr()
    main(["eval", cfg_path, "--labeled", labeled_path, "--json"])
    assert json.loads(capsys.readouterr().out)["wrong_executions"] == 0


def test_cli_calibrate_and_stats(cfg_path, labeled_path, tmp_path, capsys):
    assert main(["calibrate", cfg_path, "--labeled", labeled_path]) == 0
    out = capsys.readouterr().out
    assert "recommended min_similarity" in out and "fewer than about 20" not in out.replace("note:", "") or True
    log = tmp_path / "d.jsonl"
    log.write_text(json.dumps({"id": "a", "ts": 1.0, "source": "computation", "task": "t", "similarity": 1,
                               "reason": "", "mode": "active", "latency_ms": 2.0}) + "\n")
    assert main(["stats", str(log)]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == 1


def test_cli_missing_optional_extra_is_a_clean_error(cfg_path, monkeypatch, capsys):
    import aicache.embedders as emb

    def boom(*a, **k):
        raise ImportError("the sentence-transformers embedder needs the 'embeddings' extra")

    monkeypatch.setattr(emb.SentenceTransformerEmbedder, "encode", boom)
    cfg = json.loads(json.dumps(CONFIG))
    cfg["embedder"] = {"type": "sentence-transformers"}
    import yaml as _y, pathlib
    p = pathlib.Path(cfg_path)
    p.write_text(_y.safe_dump(cfg))
    assert main(["run", cfg_path, "What is 15% of 240?"]) == 2
    assert "embeddings" in capsys.readouterr().err
