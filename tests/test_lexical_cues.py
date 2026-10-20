"""The lexical-cue guard (EXP-016): a request the embedding matches to the
WRONG same-signature task must be rejected, not executed.

Uses a stub identifier that always returns the wrong task, so the test pins
down the guard itself and does not depend on embedding behaviour.
"""

from aicache.cache.computation import ComputationCache
from aicache.computation.tasks import TASK_REGISTRY
from aicache.matching.task_identification import IdentificationResult


class _AlwaysIdentifies:
    def __init__(self, task_type: str) -> None:
        self.task_type = task_type

    def identify(self, text: str) -> IdentificationResult:
        return IdentificationResult(self.task_type, TASK_REGISTRY[self.task_type], 0.55, "stub")


def test_wrong_operator_is_rejected_not_executed():
    # Observed live (EXP-016): "508 + 780 = ?" matched to subtract at 0.55.
    cache = ComputationCache(identifier=_AlwaysIdentifies("subtract"))
    r = cache.lookup("508 + 780 = ?")
    assert not r.hit
    assert r.reason == "lexical_cue_missing"


def test_matching_operator_still_hits():
    cache = ComputationCache(identifier=_AlwaysIdentifies("add"))
    r = cache.lookup("508 + 780 = ?")
    assert r.hit and r.answer == 1288.0


def test_guard_can_be_disabled_for_ablation():
    cache = ComputationCache(identifier=_AlwaysIdentifies("subtract"), use_lexical_cues=False)
    r = cache.lookup("508 + 780 = ?")
    assert r.hit and r.answer == -272.0  # the silent wrong answer the guard exists to stop


def test_cues_cover_the_natural_phrasings_of_each_operator():
    cases = {
        "add": ["What is 3 plus 4?", "Calculate the sum of 3 and 4.", "Can you add 3 and 4 together?"],
        "subtract": ["What is 9 minus 4?", "Starting from 9, subtract 4.", "Calculate the difference between 9 and 4."],
        "multiply": ["What is 3 times 4?", "Multiply 3 by 4.", "Calculate the product of 3 and 4."],
    }
    for task, texts in cases.items():
        cache = ComputationCache(identifier=_AlwaysIdentifies(task))
        for t in texts:
            assert cache.lookup(t).hit, (task, t)
