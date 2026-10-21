"""The learning loop: propose, verify, probation, promotion, reuse.

The small LLM is replaced by a deterministic stand-in, so these tests exercise
the machinery and not any particular model.
"""

import json
import re

import pytest

from aicache.actions import ActionStore
from aicache.config import load_config
from aicache.embedders import HashingEmbedder
from aicache.engine import ActionCache
from aicache.learner import LearnPolicy
from aicache.normalizer import Normalizer
from aicache.router import ComputationRouter

NUM = r"\$?\d[\d,]*(?:\.\d+)?"


def _nums(text):
    return [float(n.replace("$", "").replace(",", "")) for n in re.findall(NUM, text)]


def oracle(messages, action_override=None, names_override=None, expr_override=None):
    """Stand-in for the small LLM: identifies the operation and names the roles."""
    user = messages[-1]["content"]
    req = re.findall(r"Request: (.*)\nAnswer:", user)[-1]
    if re.search(r"%|percent", req):
        act, names, expr = "percent_of", ["pct", "base"], "base * pct / 100"
    elif re.search(r"\bplus\b|\badd\b|\bsum\b", req, re.I):
        act, names, expr = "add", ["x", "y"], "x + y"
    else:
        act, names, expr = "mystery", ["a"], "a"
    act = action_override or act
    names = names_override or names
    expr = expr_override or expr
    return json.dumps({"action": act, "names": {f"n{i + 1}": n for i, n in enumerate(names)},
                       "expression": expr, "response": "{value}"})


def llm(text):
    """The main LLM: always right, and counts how often it is called."""
    llm.calls += 1
    n = _nums(text)
    if re.search(r"%|percent", text):
        return f"The answer is {n[0] * n[1] / 100:g}."
    if re.search(r"\bplus\b|\badd\b|\bsum\b", text, re.I):
        return f"{n[0] + n[1]:g}"
    return "I do not know."


@pytest.fixture(autouse=True)
def _reset():
    llm.calls = 0


def make(complete=oracle, store=None, **policy):
    router = ComputationRouter(HashingEmbedder(), min_similarity=0.0)
    store = store or ActionStore(":memory:")
    kw = dict(min_agreements=3, reject_after=2)
    kw.update(policy)
    cache = ActionCache(router, store, Normalizer(complete=complete), LearnPolicy(**kw), synchronous=True)
    return cache


def status(cache, name="percent_of"):
    return cache.store.get(name).status


def test_a_request_is_learned_then_promoted_then_answered_without_the_llm():
    c = make()
    first = c.handle("What is 15% of 240?", llm)
    assert first.source == "llm" and first.learning == "learned:percent_of"
    assert status(c) == "probation"
    assert [t.text for t in c.store.get("percent_of").templates] == ["What is {pct}% of {base}?"]
    for req in ("What is 20% of 50?", "What is 10% of 90?"):
        out = c.handle(req, llm)
        assert out.source == "llm"             # still on probation: the LLM answers, the action only watches
    assert status(c) == "probation"
    c.handle("What is 7% of 300?", llm)        # third distinct agreement
    assert status(c) == "active"
    calls = llm.calls
    out = c.handle("What is 12% of 500?", llm)
    assert out.source == "computation" and out.value == 60.0 and out.text == "60.0"
    assert llm.calls == calls                  # no LLM call
    assert c.store.get("percent_of").used == 1
    s = c.router.telemetry.snapshot()["learning"]
    assert s["learned"] == 1 and s["promoted"] == 1


def test_the_same_values_asked_twice_count_as_one_agreement():
    c = make()
    c.handle("What is 15% of 240?", llm)
    for _ in range(5):
        c.handle("What is 20% of 50?", llm)
    assert c.store.get("percent_of").templates[0].agree == 1 and status(c) == "probation"


def test_a_formula_that_is_right_once_by_coincidence_is_caught_on_probation():
    def bad(messages):
        d = json.loads(oracle(messages))
        if d["action"] == "add":
            d["expression"] = "x * y"          # 2 + 2 == 2 * 2
        return json.dumps(d)

    c = make(bad, reject_after=1)
    first = c.handle("What is 2 plus 2?", llm)
    assert first.learning == "learned:add"      # passed verification: 4 == 4
    c.handle("What is 3 plus 5?", llm)          # the action says 15, the LLM says 8
    assert status(c, "add") == "rejected"
    out = c.handle("What is 4 plus 6?", llm)    # a rejected action never answers
    assert out.source == "llm"
    assert c.router.telemetry.snapshot()["learning"]["demoted"] == 1


def test_proposals_that_fail_verification_are_dropped_and_counted():
    def lies(messages):
        d = json.loads(oracle(messages))
        d["expression"] = "base * pct"          # wrong: does not reproduce the answer
        return json.dumps(d)

    c = make(lies)
    out = c.handle("What is 15% of 240?", llm)
    assert out.learning == "rejected:expression_disagrees_with_answer"
    assert c.store.count() == 0
    assert c.router.telemetry.snapshot()["learning"] == {"rejected:expression_disagrees_with_answer": 1}


def test_a_model_that_returns_garbage_or_fails_does_not_break_requests():
    assert make(lambda m: "I cannot help").handle("What is 15% of 240?", llm).learning == "rejected:no_json"

    def boom(m):
        raise TimeoutError("slow")

    out = make(boom).handle("What is 15% of 240?", llm)
    assert out.source == "llm" and out.learning == "rejected:normalizer_error:TimeoutError"


@pytest.mark.parametrize("req,why", [
    ("My card 4111111111111111 what is 5% of 200?", "sensitive_pattern"),
    ("Email bob@example.com what is 5% of 200?", "sensitive_pattern"),
    ("My password is hunter2, what is 5% of 200?", "sensitive_pattern"),
    ("Tell me a joke", "no_numbers"),
    ("What is 1 plus 2 plus 3 plus 4 plus 5 plus 6 plus 7?", "too_many_numbers"),
    ("What is 5% of 200? " + "padding " * 60, "too_long"),
])
def test_requests_that_look_sensitive_or_unsuitable_are_never_sent_to_the_normalizer(req, why):
    sent = []
    c = make(lambda m: sent.append(m) or oracle(m))
    assert c.handle(req, llm).learning == f"skipped:{why}"
    assert sent == [] and c.store.count() == 0


def test_a_new_phrasing_of_a_known_operation_is_merged_not_duplicated():
    c = make()
    c.handle("What is 15% of 240?", llm)
    out = c.handle("Calculate 15 percent of 240", llm)
    assert out.learning == "merged:percent_of"
    a = c.store.get("percent_of")
    assert len(a.templates) == 2 and c.store.count() == 1


def test_a_wrong_formula_under_a_known_name_is_refused():
    c = make()
    c.handle("What is 15% of 240?", llm)

    c.learner.normalizer = Normalizer(complete=lambda m: oracle(m, expr_override="base + pct"))
    out = c.handle("Calculate 10 percent of 50", llm)
    assert out.learning == "rejected:expression_disagrees_with_answer"   # fails verification before any merge
    assert len(c.store.get("percent_of").templates) == 1


def test_learned_actions_survive_a_restart(tmp_path):
    path = tmp_path / "actions.db"
    c = make(store=ActionStore(path))
    for r in ("What is 15% of 240?", "What is 20% of 50?", "What is 10% of 90?", "What is 7% of 300?"):
        c.handle(r, llm)
    assert status(c) == "active"
    c.close()
    c2 = make(store=ActionStore(path))                    # a fresh process: nothing in memory
    calls = llm.calls
    out = c2.handle("What is 30% of 70?", llm)
    assert out.source == "computation" and out.value == 21.0 and llm.calls == calls


def test_manual_promotion_waits_for_a_person():
    c = make(promotion="manual")
    for r in ("What is 15% of 240?", "What is 20% of 50?", "What is 10% of 90?", "What is 7% of 300?"):
        c.handle(r, llm)
    a = c.store.get("percent_of")
    assert a.status == "probation" and "ready for approval" in a.templates[0].note
    assert c.handle("What is 12% of 500?", llm).source == "llm"
    c.approve("percent_of")
    assert c.handle("What is 12% of 500?", llm).source == "computation"
    c.reject("percent_of")
    assert c.handle("What is 12% of 500?", llm).source == "llm"


def test_an_audit_that_disagrees_sends_a_live_action_back_to_probation():
    c = make()
    for r in ("What is 15% of 240?", "What is 20% of 50?", "What is 10% of 90?", "What is 7% of 300?"):
        c.handle(r, llm)
    assert status(c) == "active"
    c.audit_rate = 1.0
    drifted = lambda text: "The answer is 1."          # the LLM now disagrees
    c.handle("What is 12% of 500?", drifted)             # computed locally, then audited in the background
    c.flush()
    assert status(c) == "probation" and c.store.get("percent_of").templates[0].note == "audit disagreement"
    c.close()


def test_requests_that_ask_for_an_explanation_use_the_llm_but_not_for_the_math():
    c = make()
    for r in ("Explain what 15% of 240 is", "Explain what 20% of 50 is", "Explain what 10% of 90 is",
              "Explain what 7% of 300 is"):
        c.handle(r, llm)                                   # learn and promote this exact wording
    assert status(c) == "active"
    seen = []
    out = c.handle("Explain what 12% of 500 is", lambda t: seen.append(t) or "Twelve percent of 500 is 60, because...")
    assert out.source == "computation+llm" and "60" in out.text
    assert "60.0" in seen[0] and "do not recompute" in seen[0]      # the LLM is handed the result, not asked to compute it


def test_the_store_has_a_size_limit():
    c = make(max_actions=1)
    c.handle("What is 15% of 240?", llm)
    assert c.handle("What is 2 plus 3?", llm).learning == "skipped:store_full"


def test_background_learning_does_not_block_and_can_be_flushed():
    router = ComputationRouter(HashingEmbedder(), min_similarity=0.0)
    c = ActionCache(router, ActionStore(":memory:"), Normalizer(complete=oracle), LearnPolicy(), synchronous=False)
    out = c.handle("What is 15% of 240?", llm)
    assert out.learning is None                          # not known yet: it runs on the worker
    c.flush()
    assert c.store.get("percent_of") is not None
    c.close()


def test_a_failing_learner_never_breaks_the_request_path():
    router = ComputationRouter(HashingEmbedder(), min_similarity=0.0)
    c = ActionCache(router, ActionStore(":memory:"), Normalizer(complete=oracle), LearnPolicy(), synchronous=False)

    def broken(*a, **k):
        raise RuntimeError("disk full")

    c.store.add = broken                                  # a failure after verification, in the worker
    assert c.handle("What is 15% of 240?", llm).source == "llm"
    c.flush()
    assert c.router.telemetry.snapshot()["learning"].get("error", 0) == 1
    assert c.handle("What is 20% of 50?", llm).source == "llm"   # and requests keep working
    c.close()


def test_from_config_builds_the_whole_thing(tmp_path):
    cfg = load_config({
        "min_similarity": 0.0, "embedder": {"type": "hashing"},
        "learning": {"enabled": True, "store": str(tmp_path / "a.db"), "synchronous": True, "min_agreements": 2,
                     "normalizer": {"base_url": "http://x/v1", "model": "tiny"}},
        "tools": [{"name": "double", "examples": ["double x", "twice x", "x times two"],
                   "slots": [{"name": "x", "before": "double|twice"}], "compute": "x * 2"}],
    })
    c = ActionCache.from_config(cfg, normalizer=Normalizer(complete=oracle))
    assert c.pinned == {"double"}
    assert c.handle("double 21", llm).value == 42.0       # a pinned tool answers from the start
    c.handle("What is 15% of 240?", llm)
    assert c.store.get("percent_of").status == "probation"


def test_config_requires_a_normalizer_when_learning_is_on_and_some_source_of_actions():
    from aicache.config import ConfigError
    with pytest.raises(ConfigError, match="normalizer"):
        load_config({"learning": {"enabled": True}})
    with pytest.raises(ConfigError, match="at least one tool"):
        load_config({"embedder": {"type": "hashing"}})


def test_the_same_operation_with_different_role_names_is_merged_by_computing_not_by_name():
    c = make()
    c.handle("What is 15% of 240?", llm)                       # percent_of(pct, base)
    c.learner.normalizer = Normalizer(complete=lambda m: oracle(m, "tax_owed", names_override=["rate", "income"],
                                                               expr_override="income * rate / 100"))
    out = c.handle("The tax at 8 percent on an income of 500", llm)
    assert out.learning == "merged:percent_of" and c.store.count() == 1
    a = c.store.get("percent_of")
    assert len(a.templates) == 2 and set(a.templates[1].text.replace("{", " ").replace("}", " ").split()) >= {"pct", "base"}
    assert a.templates[1].status == "probation"


def test_roles_in_a_different_order_are_matched_to_the_known_ones():
    def minus_llm(text):
        n = _nums(text)
        return f"{n[0] - n[1]:g}" if "minus" in text else f"{n[1] - n[0]:g}"   # "Subtract 4 from 19" -> 19 - 4

    c = make(lambda m: oracle(m, "subtract", ["x", "y"], "x - y"))
    c.handle("What is 19 minus 4?", minus_llm)                       # subtract(x, y) = x - y
    # "Subtract 4 from 19": the first number is the amount taken away. Roles differ.
    c.learner.normalizer = Normalizer(complete=lambda m: oracle(m, "subtract", ["amount", "start"], "start - amount"))
    out = c.handle("Subtract 4 from 19", minus_llm)
    assert out.learning == "merged:subtract"
    t = c.store.get("subtract").templates[1].text
    assert t == "Subtract {y} from {x}"                              # amount -> y, start -> x: the only valid matching


def test_a_generic_name_reused_for_a_different_operation_gets_its_own_action():
    c = make()
    c.handle("What is 15% of 240?", llm)
    c.learner.normalizer = Normalizer(complete=lambda m: oracle(m, action_override="percent_of"))
    out = c.handle("What is 3 plus 4?", llm)                    # same name, different function
    assert out.learning == "learned:percent_of_2" and c.store.count() == 2


def test_numbers_are_masked_by_code_so_a_model_cannot_mangle_the_request():
    from aicache.normalizer import skeleton
    assert skeleton("If I invest $1,000.50 at 5% for 3 years") == (
        "If I invest ${n1} at {n2}% for {n3} years", [1000.5, 5.0, 3.0])
    sent = []
    c = make(lambda m: sent.append(m[-1]["content"]) or oracle(m))
    c.handle("What is 15% of 240?", llm)
    assert "Skeleton: What is {n1}% of {n2}?" in sent[0]


def test_equivalent_under_renaming():
    from aicache.normalizer import equivalent_under_renaming as eq
    assert eq("a - b", ["a", "b"], "x - y", ["x", "y"]) == {"a": "x", "b": "y"}
    assert eq("b - a", ["a", "b"], "x - y", ["x", "y"]) == {"a": "y", "b": "x"}        # roles swapped
    assert eq("a + b", ["a", "b"], "x * y", ["x", "y"]) is None
    assert eq("a", ["a"], "x + y", ["x", "y"]) is None                                    # different arity
    assert eq("income * rate / 100", ["income", "rate"], "base * pct / 100", ["pct", "base"]) is not None


def test_a_new_wording_on_a_trusted_action_has_to_earn_trust_by_itself():
    c = make()
    for r in ("What is 15% of 240?", "What is 20% of 50?", "What is 10% of 90?", "What is 7% of 300?"):
        c.handle(r, llm)
    assert c.handle("What is 12% of 500?", llm).source == "computation"          # trusted wording
    assert c.handle("Calculate 15 percent of 240", llm).learning == "merged:percent_of"
    a = c.store.get("percent_of")
    assert [t.status for t in a.templates] == ["active", "probation"]            # the action is trusted, the wording is not
    out = c.handle("Calculate 9 percent of 100", llm)
    assert out.source == "llm"                                                   # the LLM still answers it
    for r in ("Calculate 11 percent of 20", "Calculate 13 percent of 30", "Calculate 5 percent of 70"):
        c.handle(r, llm)
    assert [t.status for t in c.store.get("percent_of").templates] == ["active", "active"]
    assert c.handle("Calculate 2 percent of 1000", llm).source == "computation"


def test_a_request_for_a_different_operation_is_never_answered_by_another_actions_wording():
    """The failure that broke the first design: 'add' bound by 'multiply'."""
    def llm2(text):
        n = _nums(text)
        return f"{n[0] * n[1]:g}" if "multiply" in text else f"{n[0] + n[1]:g}"

    def oracle2(m):
        req = re.findall(r"Request: (.*)\nAnswer:", m[-1]["content"])[-1]
        mul = "multiply" in req
        return json.dumps({"action": "multiply" if mul else "add", "names": {"n1": "a", "n2": "b"},
                           "expression": "a * b" if mul else "a + b", "response": "{value}"})

    c = make(oracle2, min_agreements=2)
    for r in ("What do you get if you multiply 3 and 4?", "What do you get if you multiply 5 and 6?",
              "What do you get if you multiply 7 and 8?"):
        c.handle(r, llm2)
    assert c.store.get("multiply").status == "active"
    out = c.handle("What do you get if you add 508 and 780?", llm2)             # one word different
    assert out.source == "llm" and out.text == "1288"                            # not answered by multiply
    assert c.handle("What do you get if you multiply 9 and 9?", llm2).value == 81.0


def test_phrasings_are_matched_exactly_up_to_case_spacing_and_end_punctuation():
    c = make(min_agreements=1)
    c.handle("What is 15% of 240?", llm)
    c.handle("What is 20% of 50?", llm)
    assert status(c) == "active"
    for ok in ("what is 7% of 300", "What  is 7%  of 300?", "What is 7% of 1,300!"):
        assert c.handle(ok, llm).source == "computation", ok
    for no in ("What's 7% of 300?", "What is 7% of 300 dollars?", "What is 7 % of 300?", "Tell me what is 7% of 300"):
        assert c.handle(no, llm).source == "llm", no


def test_routing_stays_correct_while_actions_are_reinstalled_on_another_thread():
    """The learner reinstalls an action's phrasings in the background while requests
    are being routed. A request must never error, and never be answered wrongly."""
    import threading

    c = make(min_agreements=1)
    for r in ("What is 15% of 240?", "What is 20% of 50?"):
        c.handle(r, llm)
    assert status(c) == "active"
    action = c.store.get("percent_of")
    stop = threading.Event()
    errors, wrong = [], []

    def churn():
        while not stop.is_set():
            c._install(action)               # remove and re-add every phrasing, over and over

    def ask(i):
        for k in range(150):
            pct, base = (i % 9) + 1, (k % 40) + 10
            try:
                out = c.handle(f"What is {pct}% of {base}?", llm)
            except Exception as exc:         # pragma: no cover - this is what the test guards against
                errors.append(repr(exc))
                return
            if out.source == "computation" and abs(out.value - pct * base / 100) > 1e-9:
                wrong.append((pct, base, out.value))

    t = threading.Thread(target=churn)
    workers = [threading.Thread(target=ask, args=(i,)) for i in range(6)]
    t.start()
    [w.start() for w in workers]
    [w.join() for w in workers]
    stop.set()
    t.join()
    assert errors == [] and wrong == []


def test_a_ranking_that_names_a_task_removed_a_moment_ago_does_not_break_routing():
    """Deterministic version of the race: the ranking was computed, then the learner
    removed a task before the router looked it up."""
    c = make(min_agreements=1)
    for r in ("What is 15% of 240?", "What is 20% of 50?"):
        c.handle(r, llm)
    real_rank = c.router.rank
    c.router.rank = lambda text: [("ghost#0", 0.99)] + real_rank(text)    # a task that no longer exists, ranked first
    out = c.handle("What is 7% of 300?", llm)
    assert out.source == "computation" and out.value == 21.0
    c.router.rank = lambda text: [("ghost#0", 0.99)]                      # only ghosts: falls back cleanly
    assert c.handle("What is 7% of 300?", llm).source == "llm"


def test_learning_off_never_creates_a_database_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config({"min_similarity": 0.0, "embedder": {"type": "hashing"},
                       "tools": [{"name": "double", "examples": ["double x"],
                                  "slots": [{"name": "x", "before": "double"}], "compute": "x * 2"}]})
    c = ActionCache.from_config(cfg, embedder=HashingEmbedder())
    assert c.handle("double 4", llm).value == 8.0
    c.close()
    assert list(tmp_path.iterdir()) == []                  # no aicache.db, no -wal, no -shm


# ---- the integration interface ------------------------------------------------

def test_lookup_and_record_are_all_a_framework_needs():
    c = make(min_agreements=1)
    text = "What is 15% of 240?"
    r = c.lookup(text)
    assert r.source == "fallback"                                  # nothing known yet
    answer = llm(text)
    assert c.record(text, r, answer) == "learned:percent_of"        # the framework hands the answer back
    for t in ("What is 20% of 50?",):
        r = c.lookup(t)
        c.record(t, r, llm(t))                                      # a phrasing on probation: compared, LLM still answers
    assert status(c) == "active"
    r = c.lookup("What is 12% of 500?")
    assert r.source == "computation" and r.text == "60.0"


def test_wrap_turns_any_text_function_into_one_with_the_cache_in_front():
    c = make(min_agreements=1)
    ask = c.wrap(llm)
    assert ask("What is 15% of 240?") == "The answer is 36."          # the LLM answered, and the cache learned
    ask("What is 20% of 50?")
    calls = llm.calls
    assert ask("What is 12% of 500?") == "60.0" and llm.calls == calls


def test_async_handle_with_sync_and_async_models():
    import asyncio

    async def go():
        c = make(min_agreements=1)

        async def allm(t):
            return llm(t)

        a = await c.ahandle("What is 15% of 240?", allm)           # async model
        b = await c.ahandle("What is 20% of 50?", llm)             # sync model
        d = await c.ahandle("What is 12% of 500?", llm)
        return a.source, b.source, d.source, d.text

    assert asyncio.run(go()) == ("llm", "llm", "computation", "60.0")
