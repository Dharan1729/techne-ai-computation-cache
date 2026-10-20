"""Runs INSIDE the dedicated GPTCache venv (see scripts/setup_external_baselines.sh) — must not import anything from this repo's own `aicache`
package, since the whole point of running as a subprocess is to keep
GPTCache's dependency tree (torch, onnxruntime, faiss-cpu) isolated from
this project's environment.

Usage: python3 runner_gptcache.py <input.json> <output.json> <threshold>

Protocol: see ../protocol.py in the main process. Reads a JSON list of
precomputed workload items, replays them through GPTCache's local
(SBERT + FAISS + SQLite) stack in order — check, and on miss store the
precomputed LLM answer — writes results back as JSON.

GPTCache's own default embedding here (SBERT `all-MiniLM-L6-v2`) is
deliberately left as GPTCache's own out-of-the-box default rather than
forced to match this repo's own `redis/langcache-embed-v1` choice — the
point of this comparison is partly to show how much the embedding model
choice itself matters (see docs/prior-art.md), not just the caching algorithm.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Must happen before gptcache/faiss/torch touch OpenMP — faiss-cpu and
# torch each bundle their own libomp on macOS, which segfaults under
# multi-threaded use unless faiss is pinned to a single thread first.
import faiss  # noqa: E402

faiss.omp_set_num_threads(1)

from gptcache import Cache  # noqa: E402
from gptcache.adapter.api import get, put  # noqa: E402
from gptcache.embedding import SBERT  # noqa: E402
from gptcache.manager import CacheBase, VectorBase, get_data_manager  # noqa: E402
from gptcache.processor.pre import get_prompt  # noqa: E402
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation  # noqa: E402


def build_cache(workdir: str) -> Cache:
    # GPTCache's FAISS/sqlite backends default to fixed filenames and
    # silently reload a stale index across runs (a documented gotcha
    # from this project's own earlier GPTCache spike) unless cleared.
    faiss_path = os.path.join(workdir, "faiss.index")
    sqlite_path = os.path.join(workdir, "sqlite.db")
    for stale in (faiss_path, sqlite_path):
        if os.path.exists(stale):
            os.remove(stale)

    cache = Cache()
    embedder = SBERT("all-MiniLM-L6-v2")
    data_manager = get_data_manager(
        CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
        VectorBase("faiss", dimension=embedder.dimension, index_path=faiss_path),
    )
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=embedder.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    return cache, embedder.dimension


def main() -> None:
    input_path, output_path, threshold_str = sys.argv[1], sys.argv[2], sys.argv[3]
    threshold = float(threshold_str)
    workdir = os.path.dirname(os.path.abspath(output_path))

    with open(input_path) as f:
        items = json.load(f)

    cache, embed_dim = build_cache(workdir)
    cache.config.similarity_threshold = threshold

    results = []
    for item in items:
        t0 = time.perf_counter()
        answer = get(item["text"], cache_obj=cache)
        lookup_latency_ms = (time.perf_counter() - t0) * 1000

        hit = answer is not None
        result = dict(item)
        result["hit"] = hit
        result["served_text"] = answer if hit else None
        result["lookup_latency_ms"] = lookup_latency_ms
        result["similarity_or_distance"] = None  # GPTCache's api.get() doesn't surface the score
        results.append(result)

        if not hit:
            put(item["text"], item["llm_answer_text"], cache_obj=cache)

    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "meta": {
                    "system": "gptcache",
                    "threshold": threshold,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": embed_dim,
                },
            },
            f,
        )


if __name__ == "__main__":
    main()
