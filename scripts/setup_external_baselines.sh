#!/usr/bin/env bash
# Set up the optional GPTCache and RedisVL comparison baselines (EXP-006).
# Creates two isolated virtual environments under .venvs-external/ (override with
# AICC_EXTERNAL_VENVS_ROOT) so their torch/faiss dependencies never touch your own
# environment. Versions match the ones used in EXP-006.
#
# RedisVL also needs a Redis with the search module on localhost:6379. If Docker is
# available this script starts one (container name: aicache-redis). Remove it with:
#   docker rm -f aicache-redis
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${AICC_EXTERNAL_VENVS_ROOT:-$(pwd)/.venvs-external}"
mkdir -p "$ROOT"

python3 -m venv "$ROOT/.venv-gptcache"
# sqlalchemy is imported by gptcache 0.1.44 but not declared by it
"$ROOT/.venv-gptcache/bin/pip" install -q "gptcache==0.1.44" faiss-cpu sentence-transformers sqlalchemy

python3 -m venv "$ROOT/.venv-redisvl"
"$ROOT/.venv-redisvl/bin/pip" install -q "redisvl==0.25.1" sentence-transformers

PORT="${AICC_REDIS_PORT:-6379}"
if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); s.connect(('localhost', $PORT))" 2>/dev/null; then
  echo "something already listens on localhost:$PORT; using it as the Redis for RedisVL"
elif command -v docker >/dev/null 2>&1; then
  docker rm -f aicache-redis >/dev/null 2>&1 || true
  docker run -d --name aicache-redis -p "$PORT:6379" redis/redis-stack-server:latest >/dev/null
  echo "started redis-stack-server as aicache-redis on :$PORT"
else
  echo "Docker not found: start any redis-stack-compatible server on localhost:$PORT yourself."
fi
echo "environments in $ROOT"
echo "now run: AICC_REDIS_PORT=$PORT python -m benchmarks.run_experiment --skip-threshold-sweep --output-dir out/external"
