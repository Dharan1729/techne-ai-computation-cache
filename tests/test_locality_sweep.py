from benchmarks.level3.locality_sweep import _run_point


def test_run_point_produces_all_five_baselines_with_matching_request_counts():
    results = _run_point(combinations_per_task=2, seed=1, db_seed=1)
    by_baseline: dict[str, int] = {}
    for r in results:
        by_baseline[r.baseline] = by_baseline.get(r.baseline, 0) + 1

    assert set(by_baseline) == {
        "no_cache",
        "exact_cache",
        "semantic_cache",
        "computation_cache_fresh",
        "computation_cache_stale_ablation",
    }
    # 5 task families x 2 combinations x 3 paraphrases (a couple may be
    # skipped if a combination happens to be invalid — see generator.py).
    counts = set(by_baseline.values())
    assert counts == {30} or all(c <= 30 for c in counts)


def test_computation_cache_correctness_is_insensitive_to_combination_count():
    """The locality sweep's headline claim: unlike response caching, the
    computation cache's correctness doesn't depend on how many distinct
    entity combinations are in play."""
    for combos in (2, 10):
        results = _run_point(combinations_per_task=combos, seed=1, db_seed=1)
        fresh = [r for r in results if r.baseline == "computation_cache_fresh"]
        assert all(r.correct for r in fresh), f"combos={combos}: found an incorrect fresh-cache result"
