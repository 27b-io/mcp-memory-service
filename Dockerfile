# MCP Memory Service - Multi-stage container
# Supports: linux/amd64, linux/arm64
#
# Targets:
#   full (default) - Backward-compatible, includes torch + sentence-transformers (~1.2GB)
#   api            - Thin image, no torch/sentence-transformers (~200MB), uses HTTP embedding provider
#
# Build args:
#   CUDA_ENABLED=false (default) - CPU-only build for full target
#   CUDA_ENABLED=true            - CUDA-enabled build for full target (~5GB)
#   EMBEDDING_MODEL              - HuggingFace model ID (full target only)

# =============================================================================
# Stage 1: builder-base (shared) — all deps EXCEPT torch + sentence-transformers
# =============================================================================
FROM python:3.12-slim AS builder-base

WORKDIR /app

# Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin uv version for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# Dependencies first (cache layer)
COPY pyproject.toml uv.lock README.md ./

# Install deps with torch AND sentence-transformers pruned.
# This gives us the full dependency tree minus the heavy ML libs (~3GB avoided).
# --no-deps is safe because uv export produces a fully-resolved flat list.
RUN uv venv && \
    uv export --frozen --no-dev --no-emit-project --no-hashes \
        --prune torch --prune sentence-transformers \
        -o /tmp/requirements.txt && \
    uv pip install --no-deps -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Source code (no scripts/ - legacy maintenance tools, not needed at runtime)
COPY src/ ./src/
RUN uv pip install --no-deps -e .

# Pre-download spaCy model (GitHub releases - can be flaky, retry up to 3 times).
# spaCy is optional: if this fails the runtime falls back to FallbackAnalyzer for
# query intent analysis. We still want it for better multi-concept fan-out.
RUN for attempt in 1 2 3; do \
        uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
        && break \
        || if [ "$attempt" -lt 3 ]; then echo "spaCy download attempt $attempt failed, retrying in 5s..."; sleep 5; \
           else echo "spaCy download failed after 3 attempts, skipping (optional)"; fi; \
    done

# Clean up caches to reduce layer size
RUN rm -rf /root/.cache/pip /root/.cache/uv && \
    find .venv -name "*.pyc" -delete 2>/dev/null || true && \
    find .venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Stage 2: builder-full — extends builder-base with torch + sentence-transformers
# =============================================================================
FROM builder-base AS builder-full

ARG EMBEDDING_MODEL=intfloat/e5-small-v2
ARG CUDA_ENABLED=false

# Install torch + sentence-transformers on top of the base venv.
# Most sentence-transformers transitive deps (huggingface-hub, transformers, numpy,
# tokenizers, tqdm) are already installed from builder-base. However scikit-learn,
# scipy, and pillow are ONLY reachable through sentence-transformers (pruned in base),
# so they must be installed explicitly before --no-deps.
# CPU builds: install CPU torch from pytorch index first.
# CUDA builds: install CUDA torch from default PyPI.
RUN if [ "$CUDA_ENABLED" = "false" ]; then \
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        uv pip install torch; \
    fi && \
    uv pip install scikit-learn scipy pillow && \
    uv pip install --no-deps sentence-transformers

# Pre-download embedding model (HuggingFace - reliable)
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Clean up caches again after ML deps
RUN rm -rf /root/.cache/pip /root/.cache/uv && \
    find .venv -name "*.pyc" -delete 2>/dev/null || true && \
    find .venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Stage 3: api — thin runtime, no torch/sentence-transformers
# =============================================================================
FROM python:3.12-slim AS api

WORKDIR /app

COPY --from=builder-base /app/.venv /app/.venv
COPY --from=builder-base /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_EMBEDDING_PROVIDER=openai_compat

RUN mkdir -p /data/sqlite

# Shorter start-period: no model to load
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "mcp_memory_service.unified_server"]

# =============================================================================
# Stage 4: full — backward-compatible default (MUST be last stage)
# =============================================================================
FROM python:3.12-slim AS full

ARG EMBEDDING_MODEL=intfloat/e5-small-v2

WORKDIR /app

COPY --from=builder-full /app/.venv /app/.venv
COPY --from=builder-full /app/src /app/src
COPY --from=builder-full /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_MEMORY_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/root/.cache/huggingface

RUN mkdir -p /data/sqlite

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "mcp_memory_service.unified_server"]
