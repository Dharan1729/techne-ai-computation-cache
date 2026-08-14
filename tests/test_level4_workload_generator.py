from aicache.binding.entity_extractor import extract_entities_and_numbers
from aicache.computation.graph_tasks import GRAPH_TASK_REGISTRY
from workloads.level4.generator import BRANCHES, generate_level4_workload


def test_generates_reproducible_workload_for_fixed_seed():
    _, a = generate_level4_workload(combinations_per_branch=2, paraphrases_per_combination=2, seed=7)
    _, b = generate_level4_workload(combinations_per_branch=2, paraphrases_per_combination=2, seed=7)
    assert [i.text for i in a] == [i.text for i in b]
    assert [i.expected_result for i in a] == [i.expected_result for i in b]


def test_every_branch_is_represented_and_balanced():
    _, items = generate_level4_workload(combinations_per_branch=5, paraphrases_per_combination=2, seed=99)
    counts = {b: sum(1 for i in items if i.branch == b) for b in BRANCHES}
    for branch in BRANCHES:
        assert counts[branch] == 10, counts  # 5 combos x 2 paraphrases, none skipped


def test_expected_result_matches_independent_recomputation():
    db, items = generate_level4_workload(combinations_per_branch=3, paraphrases_per_combination=2, seed=3)
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    for item in items:
        outcome = spec.fetch(item.entities, item.numbers, db)
        assert outcome.values is not None
        recomputed = spec.executor(**item.entities, **item.numbers, **outcome.values)
        assert recomputed == item.expected_result


def test_generated_text_extracts_back_to_the_same_entities():
    db, items = generate_level4_workload(combinations_per_branch=3, paraphrases_per_combination=3, seed=11)
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    for item in items:
        extraction = extract_entities_and_numbers(item.text, spec.entity_slots, spec.number_slots)
        assert not extraction.missing_slots, item.text
        assert extraction.entities == item.entities


def test_templates_actually_substitute_entities_not_literal_placeholders():
    """Regression test for the real bug found during development: the
    workload templates were accidentally the static, unparameterized
    example_phrasings (no {0}/{1} placeholders), so every generated
    item's text was identical regardless of which entities were drawn."""
    _, items = generate_level4_workload(combinations_per_branch=3, paraphrases_per_combination=3, seed=11)
    texts = {i.text for i in items}
    assert len(texts) == len(items)
    for item in items:
        assert item.entities["customer_id"] in item.text
        assert item.entities["transaction_id"] in item.text
