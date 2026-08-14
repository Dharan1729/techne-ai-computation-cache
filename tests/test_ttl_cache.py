from aicache.cache.ttl_cache import TTLCache


def test_cache_miss_computes_and_stores():
    cache = TTLCache(ttl_seconds=10.0, now_fn=lambda: 0.0)
    calls = []

    def compute():
        calls.append(1)
        return "value"

    value, was_hit = cache.get_or_compute("k", compute)
    assert value == "value"
    assert was_hit is False
    assert len(calls) == 1


def test_repeat_call_within_ttl_is_a_hit_and_does_not_recompute():
    clock = [0.0]
    cache = TTLCache(ttl_seconds=10.0, now_fn=lambda: clock[0])
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    v1, hit1 = cache.get_or_compute("k", compute)
    clock[0] = 5.0
    v2, hit2 = cache.get_or_compute("k", compute)

    assert v1 == v2 == 1
    assert hit1 is False
    assert hit2 is True
    assert len(calls) == 1


def test_call_after_ttl_expires_recomputes():
    clock = [0.0]
    cache = TTLCache(ttl_seconds=10.0, now_fn=lambda: clock[0])
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    cache.get_or_compute("k", compute)
    clock[0] = 10.1
    v2, hit2 = cache.get_or_compute("k", compute)

    assert v2 == 2
    assert hit2 is False
    assert len(calls) == 2


def test_ttl_none_memoizes_forever():
    clock = [0.0]
    cache = TTLCache(ttl_seconds=None, now_fn=lambda: clock[0])
    calls = []

    def compute():
        calls.append(1)
        return "same"

    cache.get_or_compute("k", compute)
    clock[0] = 10_000_000.0
    v2, hit2 = cache.get_or_compute("k", compute)

    assert hit2 is True
    assert len(calls) == 1


def test_different_keys_are_independent():
    cache = TTLCache(ttl_seconds=10.0, now_fn=lambda: 0.0)
    v1, hit1 = cache.get_or_compute("a", lambda: "A")
    v2, hit2 = cache.get_or_compute("b", lambda: "B")
    assert (v1, hit1) == ("A", False)
    assert (v2, hit2) == ("B", False)
