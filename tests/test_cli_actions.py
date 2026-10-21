"""Review and offline-learning commands."""

import json
import re

import pytest
import yaml

import aicache.engine as engine_mod
from aicache.actions import Action, ActionStore, Template
from aicache.cli import main
from aicache.normalizer import Normalizer

NUM = r"\$?\d[\d,]*(?:\.\d+)?"


def oracle(messages):
    req = re.findall(r"Request: (.*)\nAnswer:", messages[-1]["content"])[-1]
    n = len(re.findall(NUM, req))
    return json.dumps({"action": "percent_of", "names": dict(zip([f"n{i + 1}" for i in range(n)], ["pct", "base"])),
                       "expression": "base * pct / 100", "response": "{value}"})


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "min_similarity": 0.0, "embedder": {"type": "hashing"},
        "learning": {"enabled": True, "store": str(tmp_path / "actions.db"),
                     "normalizer": {"base_url": "http://x/v1", "model": "tiny"}},
    }))
    return str(p), tmp_path / "actions.db"


def seed(db):
    st = ActionStore(db)
    st.add(Action("percent_of", "base * pct / 100", ["pct", "base"],
                  [Template("What is {pct}% of {base}?", "probation"), Template("Calculate {pct} percent of {base}", "probation")]))
    st.close()


def test_actions_lists_and_filters(cfg, capsys):
    path, db = cfg
    seed(db)
    assert main(["actions", path]) == 0
    out = capsys.readouterr().out
    assert "percent_of" in out and "What is {pct}% of {base}?" in out and "1 actions" in out
    assert main(["actions", path, "--status", "active"]) == 0
    assert "0 actions" in capsys.readouterr().out


def test_action_shows_evidence_and_unknown_names_fail(cfg, capsys):
    path, db = cfg
    seed(db)
    assert main(["action", path, "percent_of"]) == 0
    assert "base * pct / 100" in capsys.readouterr().out
    assert main(["action", path, "nope"]) == 2


def test_approve_and_reject_whole_actions_or_one_phrasing(cfg, capsys):
    path, db = cfg
    seed(db)
    assert main(["approve", path, "percent_of", "--phrasing", "What is {pct}% of {base}?"]) == 0
    st = ActionStore(db)
    assert [t.status for t in st.get("percent_of").templates] == ["active", "probation"]
    st.close()
    assert main(["reject", path, "percent_of"]) == 0
    st = ActionStore(db)
    a = st.get("percent_of")
    assert a.status == "rejected" and all(t.status == "rejected" for t in a.templates)
    st.close()
    assert main(["approve", path, "percent_of"]) == 2          # a rejected phrasing is not silently revived
    assert main(["approve", path, "percent_of", "--phrasing", "no such phrasing"]) == 2
    assert main(["reject", path, "nope"]) == 2


def test_export_and_import_round_trip(cfg, tmp_path, capsys):
    path, db = cfg
    seed(db)
    out = tmp_path / "actions.yaml"
    assert main(["export", path, str(out)]) == 0
    data = yaml.safe_load(out.read_text())["actions"][0]
    assert data["name"] == "percent_of" and len(data["templates"]) == 2
    db.unlink()
    assert main(["import", path, str(out)]) == 0
    assert "added 1" in capsys.readouterr().out
    assert main(["import", path, str(out)]) == 0
    assert "skipped 1" in capsys.readouterr().out
    assert main(["import", path, str(out), "--overwrite"]) == 0
    st = ActionStore(db)
    assert st.get("percent_of").compute == "base * pct / 100"
    st.close()


def test_learn_from_request_answer_pairs(cfg, tmp_path, monkeypatch, capsys):
    path, db = cfg
    monkeypatch.setattr(engine_mod, "Normalizer", lambda *a, **k: Normalizer(complete=oracle))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(p) for p in [
        {"text": "What is 15% of 240?", "answer": "36"},
        {"text": "Calculate 10 percent of 50", "answer": "5"},
        {"text": "Tell me a joke", "answer": "Why did the chicken..."},
        {"text": "What is 15% of 240?", "answer": "99"},        # wrong answer: must not be learned
    ]) + "\n")
    assert main(["learn", path, str(pairs), "--embedder", "hashing"]) == 0
    out = capsys.readouterr().out
    assert "learned x1" in out and "merged x1" in out and "skipped x1" in out and "rejected x1" in out
    st = ActionStore(db)
    a = st.get("percent_of")
    assert len(a.templates) == 2 and all(t.status == "probation" for t in a.templates)
    st.close()


def test_learn_needs_learning_enabled(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"embedder": {"type": "hashing"}, "tools": [
        {"name": "t", "examples": ["a"], "slots": [{"name": "x", "before": "a"}], "compute": "x"}]}))
    f = tmp_path / "f.jsonl"
    f.write_text("")
    assert main(["learn", str(p), str(f)]) == 2
    assert "learning is not enabled" in capsys.readouterr().err
