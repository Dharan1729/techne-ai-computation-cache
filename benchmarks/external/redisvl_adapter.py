"""Main-process side of the RedisVL comparison baseline.

Requires (a) a dedicated virtual environment with `redisvl
sentence-transformers` installed (`bash scripts/setup_external_baselines.sh`) and (b) a reachable Redis instance
with the search module at redis://localhost:6379 (e.g. `docker run -d
--name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest`).
Optional baseline — see gptcache_adapter.py's docstring for why this
repo's own baselines never depend on it, and how a missing dependency
here degrades to a skipped baseline rather than a failed run.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path

from aicache.llm.simulated import SimulatedLLM
from aicache.metrics.types import RequestResult

from workloads.types import WorkloadItem

from .protocol import build_export, results_from_runner_output

# Set up with `bash scripts/setup_external_baselines.sh`; override the location with AICC_EXTERNAL_VENVS_ROOT.
_DEFAULT_VENVS_ROOT = Path(__file__).resolve().parents[2] / ".venvs-external"
VENVS_ROOT = Path(os.environ.get("AICC_EXTERNAL_VENVS_ROOT", str(_DEFAULT_VENVS_ROOT)))
PYTHON_BIN = VENVS_ROOT / ".venv-redisvl" / "bin" / "python3"
RUNNER = Path(__file__).resolve().parent / "runner_redisvl.py"

REDIS_HOST = os.environ.get("AICC_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("AICC_REDIS_PORT", "6379"))


class RedisVLUnavailable(RuntimeError):
    pass


def is_available() -> tuple[bool, str]:
    if not PYTHON_BIN.exists():
        return False, f"no venv python at {PYTHON_BIN} (run: bash scripts/setup_external_baselines.sh)"
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=1.0):
            pass
    except OSError as exc:
        return False, f"redis not reachable at {REDIS_HOST}:{REDIS_PORT} ({exc})"
    return True, ""


def run_redisvl(
    workload: list[WorkloadItem], llm: SimulatedLLM, similarity_threshold: float = 0.75
) -> list[RequestResult]:
    """`similarity_threshold` uses the same convention as this repo's
    own semantic cache and GPTCache (higher = stricter). Converted to
    RedisVL's distance convention (lower = stricter) via
    ``distance = 1 - similarity`` before invoking the runner — see
    runner_redisvl.py's docstring."""
    ok, reason = is_available()
    if not ok:
        raise RedisVLUnavailable(reason)

    distance_threshold = 1.0 - similarity_threshold
    export = build_export(workload, llm)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.json"
        out_path = Path(tmp) / "output.json"
        in_path.write_text(json.dumps(export))

        proc = subprocess.run(
            [str(PYTHON_BIN), str(RUNNER), str(in_path), str(out_path), str(distance_threshold)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RedisVLUnavailable(
                f"redisvl runner failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
            )

        output = json.loads(out_path.read_text())

    return results_from_runner_output(output["results"], baseline_name="redisvl")
