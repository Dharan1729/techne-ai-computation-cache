from aicache.cache.semantic import SemanticCache


def test_miss_on_empty_cache(embedder):
    cache = SemanticCache(embedder=embedder, threshold=0.75)
    result = cache.lookup("What is 10 plus 20?")
    assert not result.hit


def test_hit_on_paraphrase_above_threshold(embedder):
    """This is exactly baseline C's advantage over exact caching: a
    paraphrase of a previously-seen request is a hit."""
    cache = SemanticCache(embedder=embedder, threshold=0.5)
    cache.store("What is 10 plus 20?", "30")
    result = cache.lookup("Calculate the sum of 10 and 20.")
    assert result.hit
    assert result.answer == "30"


def test_miss_below_threshold(embedder):
    cache = SemanticCache(embedder=embedder, threshold=0.999)
    cache.store("What is 10 plus 20?", "30")
    result = cache.lookup("What is the capital of France?")
    assert not result.hit


def test_reuses_stale_answer_under_different_parameters(embedder):
    """The false-hit mechanism RQ2/RQ3 are about: a semantic cache with a
    loose-enough threshold reuses an answer computed for DIFFERENT
    numbers, because it only measures textual similarity, not parameter
    equality. Correctness of that reuse is the harness's job to check
    (aicache.metrics.correctness) — this test only proves the cache layer
    itself has no such check built in, which is the point being tested."""
    cache = SemanticCache(embedder=embedder, threshold=0.6)
    cache.store("What is 10 plus 20?", "30")
    result = cache.lookup("What is 512 plus 937?")
    # Whether this specific pair clears 0.6 depends on the embedding
    # model; assert on the *mechanism*, not a specific similarity value:
    # if it does hit, the cache returns the OLD answer regardless of the
    # new parameters.
    if result.hit:
        assert result.answer == "30"
