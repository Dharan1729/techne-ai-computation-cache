# aicache gateway. Light image: no torch. Uses a remote or dependency-free embedder.
#   docker build -t aicache .
#   docker run -p 8080:8080 -e GATEWAY_KEY=... -v $PWD/aicache.yaml:/config/aicache.yaml aicache
# The container binds 0.0.0.0, so the config must set server.api_key_env (here GATEWAY_KEY)
# or `aicache serve` refuses to start.
# For the local embedding model, build with:  --build-arg EXTRAS=serve,embeddings   (adds torch, large)
FROM python:3.11-slim
ARG EXTRAS=serve
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[${EXTRAS}]" && useradd --create-home aicache
USER aicache
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1
ENTRYPOINT ["aicache", "serve", "/config/aicache.yaml", "--host", "0.0.0.0", "--port", "8080"]
