"""Main-process side of the GPTCache comparison baseline.

Requires the dedicated GPTCache venv from the sibling `semantic
caching/docker-tests/` project (or any venv with `gptcache faiss-cpu
sentence-transformers` installed) — see that project's README for setup.
This is an *optional* baseline: this repo's own four baselines
(no_cache/exact_cache/semantic_cache/computation_cache) never depend on
it, and `benchmarks/run_experiment.py` skips it gracefully (with a
printed reason) if the venv isn't found, rather than failing the whole
run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from aicache.llm.simulated import SimulatedLLM
from aicache.metrics.types import RequestResult

from workloads.types import WorkloadItem

from .protocol import build_export, results_from_runner_output

_DEFAULT_VENVS_ROOT = (
    Path.home() / "Desktop" / "caching" / "semantic caching" / "docker-tests"
)
VENVS_ROOT = Path(os.environ.get("AICC_EXTERNAL_VENVS_ROOT", str(_DEFAULT_VENVS_ROOT)))
PYTHON_BIN = VENVS_ROOT / ".venv-gptcache" / "bin" / "python3"
RUNNER = Path(__file__).resolve().parent / "runner_gptcache.py"


class GPTCacheUnavailable(RuntimeError):
    pass


def is_available() -> tuple[bool, str]:
    if not PYTHON_BIN.exists():
        return False, f"no venv python at {PYTHON_BIN}"
    return True, ""


def run_gptcache(
    workload: list[WorkloadItem], llm: SimulatedLLM, threshold: float = 0.75
) -> list[RequestResult]:
    ok, reason = is_available()
    if not ok:
        raise GPTCacheUnavailable(reason)

    export = build_export(workload, llm)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.json"
        out_path = Path(tmp) / "output.json"
        in_path.write_text(json.dumps(export))

        # faiss-cpu and torch each bundle their own OpenMP runtime; both
        # env vars are needed to avoid a hard segfault on macOS (see
        # runner_gptcache.py's docstring and the sibling docker-tests
        # project's README for the same requirement there).
        env = {**os.environ, "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
        proc = subprocess.run(
            [str(PYTHON_BIN), str(RUNNER), str(in_path), str(out_path), str(threshold)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if proc.returncode != 0:
            raise GPTCacheUnavailable(
                f"gptcache runner failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
            )

        output = json.loads(out_path.read_text())

    return results_from_runner_output(output["results"], baseline_name="gptcache")
