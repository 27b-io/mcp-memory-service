# MCP Memory Service - Multi-platform container
# Supports: linux/amd64, linux/arm64
# Build args:
#   CUDA_ENABLED=false (default) - CPU-only build (~1.2GB)
#   CUDA_ENABLED=true            - CUDA-enabled build (~5GB)

FROM python:3.12-slim AS builder

ARG EMBEDDING_MODEL=intfloat/e5-small-v2
ARG CUDA_ENABLED=false

WORKDIR /app

# Build tools - no git (all deps are from PyPI), no curl (not needed in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin uv version for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# Dependencies first (cache layer)
COPY pyproject.toml uv.lock README.md ./

# Install deps. CPU builds use --prune torch to strip torch + all nvidia-*/triton
# transitive deps from the lockfile export (~3GB avoided). CPU torch installed first
# so sentence-transformers' torch requirement is already satisfied. --no-deps on the
# bulk install is safe because uv export produces a fully-resolved flat list.
RUN uv venv && \
    if [ "$CUDA_ENABLED" = "false" ]; then \
        uv export --frozen --no-dev --no-emit-project --no-hashes --prune torch \
            -o /tmp/requirements.txt && \
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && \
        uv pip install --no-deps -r /tmp/requirements.txt && \
        rm /tmp/requirements.txt; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# Source code (no scripts/ - they are legacy maintenance tools, not needed at runtime)
COPY src/ ./src/
RUN uv pip install --no-deps -e .

# Pre-download spaCy model, embedding model, and clean up in the same layer
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
    .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    find .venv -name "*.pyc" -delete 2>/dev/null || true && \
    find .venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Runtime stage - minimal
FROM python:3.12-slim

ARG EMBEDDING_MODEL=intfloat/e5-small-v2

WORKDIR /app

# Copy venv, source, and model cache - no apt packages needed
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_MEMORY_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/root/.cache/huggingface

RUN mkdir -p /data/sqlite

# Healthcheck using Python - no curl dependency needed
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "mcp_memory_service.unified_server"]
