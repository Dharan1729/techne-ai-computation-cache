from aicache.cache.exact import ExactCache


def test_miss_on_empty_cache():
    cache = ExactCache()
    assert cache.lookup("What is 10 plus 20?") is None


def test_hit_on_byte_identical_request():
    cache = ExactCache()
    cache.store("What is 10 plus 20?", "30")
    assert cache.lookup("What is 10 plus 20?") == "30"


def test_hit_is_case_and_whitespace_insensitive():
    cache = ExactCache()
    cache.store("What is 10 plus 20?", "30")
    assert cache.lookup("  WHAT IS 10   plus 20?  ") == "30"


def test_miss_on_paraphrase():
    """The whole point of exact caching's limitation (RQ1): a paraphrase
    of the same computation is NOT a hit."""
    cache = ExactCache()
    cache.store("What is 10 plus 20?", "30")
    assert cache.lookup("Calculate the sum of 10 and 20.") is None


def test_miss_on_different_parameters():
    cache = ExactCache()
    cache.store("What is 10 plus 20?", "30")
    assert cache.lookup("What is 51 plus 93?") is None
