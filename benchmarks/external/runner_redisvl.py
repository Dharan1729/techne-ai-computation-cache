"""Runs INSIDE the dedicated RedisVL venv — must not import anything
from this repo's own `aicache` package (see runner_gptcache.py's
docstring for why out-of-process isolation is deliberate here).

Usage: python3 runner_redisvl.py <input.json> <output.json> <distance_threshold>

Requires a reachable Redis instance with the search module (e.g.
`redis/redis-stack`) at redis://localhost:6379 — this repo does not
start one; it expects the user's existing container (or any reachable
redis-stack-compatible instance) to already be running.

Note the threshold direction: RedisVL's `SemanticCache` takes a
*distance* threshold (lower = stricter/closer match required), the
inverse of this repo's own semantic cache and of GPTCache's
*similarity* threshold (higher = stricter). The caller (see
../redisvl_adapter.py) converts similarity -> distance via
`distance = 1 - similarity` before invoking this script, so the sweep
stays aligned across all three systems in the final comparison.
"""

from __future__ import annotations

import json
import sys
import time

from redisvl.extensions.cache.llm import SemanticCache


def main() -> None:
    input_path, output_path, distance_str = sys.argv[1], sys.argv[2], sys.argv[3]
    distance_threshold = float(distance_str)

    with open(input_path) as f:
        items = json.load(f)

    cache = SemanticCache(
        name=f"aicc-thresh-{distance_threshold}",
        redis_url="redis://localhost:6379",
        distance_threshold=distance_threshold,
        overwrite=True,
    )
    cache.clear()

    vectorizer_name = type(cache._vectorizer).__name__
    vectorizer_model = getattr(cache._vectorizer, "model", None)

    results = []
    for item in items:
        t0 = time.perf_counter()
        hits = cache.check(prompt=item["text"], num_results=1)
        lookup_latency_ms = (time.perf_counter() - t0) * 1000

        hit = len(hits) > 0
        result = dict(item)
        result["hit"] = hit
        result["served_text"] = hits[0]["response"] if hit else None
        result["lookup_latency_ms"] = lookup_latency_ms
        result["similarity_or_distance"] = hits[0]["vector_distance"] if hit else None
        results.append(result)

        if not hit:
            cache.store(prompt=item["text"], response=item["llm_answer_text"])

    cache.clear()

    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "meta": {
                    "system": "redisvl",
                    "distance_threshold": distance_threshold,
                    "vectorizer": vectorizer_name,
                    "embedding_model": vectorizer_model,
                },
            },
            f,
        )


if __name__ == "__main__":
    main()
