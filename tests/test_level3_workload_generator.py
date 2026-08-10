from aicache.binding.entity_extractor import extract_entities_and_numbers
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from workloads.level3.generator import generate_level3_workload
from workloads.level3.templates import LEVEL3_TEMPLATES


def test_generates_reproducible_workload_for_fixed_seed():
    _, a = generate_level3_workload(combinations_per_task=2, paraphrases_per_combination=2, seed=7)
    _, b = generate_level3_workload(combinations_per_task=2, paraphrases_per_combination=2, seed=7)
    assert [i.text for i in a] == [i.text for i in b]
    assert [i.expected_result for i in a] == [i.expected_result for i in b]


def test_different_seeds_produce_different_workloads():
    _, a = generate_level3_workload(combinations_per_task=2, paraphrases_per_combination=2, seed=1)
    _, b = generate_level3_workload(combinations_per_task=2, paraphrases_per_combination=2, seed=2)
    assert [i.text for i in a] != [i.text for i in b]


def test_expected_result_matches_independent_recomputation():
    """Recompute each item's result directly against the DB snapshot the
    generator returns (not via the generator's own bookkeeping) to make
    sure the workload's ground truth is actually ground truth."""
    db, items = generate_level3_workload(combinations_per_task=3, paraphrases_per_combination=2, seed=3)
    for item in items:
        spec = WORKFLOW_TASK_REGISTRY[item.task_type]
        outcome = spec.fetch(item.entities, item.numbers, db)
        assert outcome.values is not None
        recomputed = spec.executor(**item.entities, **item.numbers, **outcome.values)
        assert recomputed == item.expected_result


def test_generated_text_extracts_back_to_the_same_entities_and_numbers():
    db, items = generate_level3_workload(combinations_per_task=3, paraphrases_per_combination=3, seed=11)
    for item in items:
        spec = WORKFLOW_TASK_REGISTRY[item.task_type]
        extraction = extract_entities_and_numbers(item.text, spec.entity_slots, spec.number_slots)
        assert not extraction.missing_slots, (item.text, extraction.missing_slots)
        assert extraction.entities == item.entities
        for name, value in item.numbers.items():
            assert abs(extraction.numbers[name] - value) < 1e-6


def test_every_registered_task_has_templates():
    for task_type in WORKFLOW_TASK_REGISTRY:
        assert task_type in LEVEL3_TEMPLATES and LEVEL3_TEMPLATES[task_type]
