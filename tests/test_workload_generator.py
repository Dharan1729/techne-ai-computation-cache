from aicache.binding.extractor import extract_numbers
from aicache.computation.tasks import TASK_REGISTRY
from workloads.generators import generate_workload


def test_generates_reproducible_workload_for_fixed_seed():
    a = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=7)
    b = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=7)
    assert [i.text for i in a] == [i.text for i in b]
    assert [i.expected_answer for i in a] == [i.expected_answer for i in b]


def test_different_seeds_produce_different_workloads():
    a = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=1)
    b = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=2, seed=2)
    assert [i.text for i in a] != [i.text for i in b]


def test_expected_answer_matches_independent_recomputation():
    """Recompute each item's answer directly from TASK_REGISTRY (not via
    the generator's own bookkeeping) to make sure the workload's ground
    truth is actually ground truth."""
    items = generate_workload(param_sets_per_task=3, paraphrases_per_param_set=2, seed=3)
    for item in items:
        spec = TASK_REGISTRY[item.task_type]
        recomputed = spec.executor(**item.bound_params)
        assert abs(recomputed - item.expected_answer) < 1e-9


def test_text_numbers_extract_in_declared_param_order():
    """This is the property the harness depends on for the computation
    cache to bind correctly: numbers appear in the text in the same
    order as the task's declared parameters."""
    items = generate_workload(param_sets_per_task=2, paraphrases_per_param_set=3, seed=5)
    for item in items:
        extracted = extract_numbers(item.text)[: len(item.raw_numbers)]
        assert extracted == item.raw_numbers, (
            f"template produced out-of-order numbers for {item.task_type}: "
            f"{item.text!r} -> {extracted} (expected {item.raw_numbers})"
        )


def test_every_registered_task_has_templates():
    from workloads.templates import TEMPLATES

    for task_type in TASK_REGISTRY:
        assert task_type in TEMPLATES and TEMPLATES[task_type]
