# Horizontal Scale-Out: Embedding Provider Extraction

**Date**: 2026-03-13
**Status**: Draft
**Branch**: `feat/embedding-provider-protocol`

## Problem

mcp-memory-service bundles SentenceTransformer embedding inference in-process alongside the API layer. The embedding model is the bottleneck (~50-100 embeds/sec on CPU). The API layer is trivially stateless. They cannot scale independently.

**Target**: Handle burst to 10k concurrent requests with scale-to-zero baseline. Container-first, deploy anywhere.

## Solution

Extract embedding generation into a pluggable `EmbeddingProvider` protocol (ports & adapters). The API service becomes a thin, stateless container (~200MB, <2s cold start) that delegates embedding to an external service. Off-the-shelf embedding servers (HuggingFace TEI, vLLM, Ollama) serve as the embedding backend — we do not build a custom embedding server.

## Architecture

```
                    +--------------------------------------+
                    |          Load Balancer               |
                    +----------------+---------------------+
                                     |
                    +----------------v---------------------+
                    |       API Service (N replicas)       |
                    |  FastAPI + MCP streamable HTTP       |
                    |  Stateless, CPU-only, scale-to-zero  |
                    |                                      |
                    |  CachedEmbeddingProvider             |
                    |    -> L1 (in-process) -> L2 (Redis)  |
                    |    -> EmbeddingProvider on miss       |
                    +-------+-------------------+----------+
                            |                   |
              +-------------v----+    +---------v-----------+
              |  Redis           |    |  Embedding Service   |
              |  CacheKit L2     |    |  (off-the-shelf)     |
              |  Hebbian queue   |    |  TEI / vLLM / Ollama |
              +-------------+---+    |  or managed API      |
                            |        +---------+-----------+
              +-------------v------------------v-----------+
              |           Data Plane (external)             |
              |  Qdrant (vectors)  +  FalkorDB (graph)      |
              +--------------------------------------------+
```

## Core Abstraction: EmbeddingProvider Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...
```

### Adapters

| Adapter | Backends Covered | Transport |
|---------|-----------------|-----------|
| `LocalProvider` | In-process SentenceTransformer | Direct call |
| `OpenAICompatAdapter` | TEI, vLLM, Ollama, OpenAI | HTTP `POST /v1/embeddings` |

Two adapters cover all current backends. The OpenAI embeddings API is the de facto standard — TEI, vLLM, and Ollama all support it. No custom wire protocol.

**Future adapters** (not built now): `VertexAIAdapter` (Google SDK), other managed providers. Added when needed.

### prompt_name Translation

Instruction-tuned models use different prefixes. The adapter owns translation:

| Model Family | query | passage |
|-------------|-------|---------|
| nomic-embed | `search_query` | `search_document` |
| E5 | `query:` | `passage:` |
| OpenAI | (ignored) | (ignored) |

Mapping lives in adapter config, not in the protocol. Default mapping for `nomic-embed` is built-in. Custom mappings via `EmbeddingSettings.prompt_name_map: dict[str, dict[str, str]]` for other model families.

## Caching: CachedEmbeddingProvider Wrapper

A decorator that wraps any `EmbeddingProvider` with CacheKit L1/L2:

```python
class CachedEmbeddingProvider:
    """Wraps an EmbeddingProvider with L1 (in-process) + L2 (Redis) caching."""

    def __init__(self, inner: EmbeddingProvider): ...

    async def embed_batch(self, texts, prompt_name="query"):
        # Per-text: delegate to a CacheKit-decorated async helper
        # with (text, prompt_name) as explicit parameters so CacheKit
        # auto-includes both in the Blake2b key hash.
        # Namespace: mcp_memory_embed_{model_name}_{dimensions}
        # CacheKit key: ns:{namespace}:func:{qualname}:args:{blake2b(text, prompt_name)}
        #
        # CRITICAL: prompt_name must be a function parameter, not hardcoded.
        # "query" and "passage" produce different vectors for the same text
        # (instruction-tuned models prepend different prefixes).
        # The current _cached_embed(text) hardcodes prompt_name inside the
        # function body — CacheKit can't see it, so query/passage share keys.
        #
        # Per-text L1 check -> L2 check -> inner.embed_batch(misses)
        # Populate L1 + L2 on miss
```

**CacheKit key generation**: Keys are auto-generated as `ns:{namespace}:func:{qualname}:args:{blake2b(args)}`. All function parameters are hashed. The namespace handles model/dimension isolation; the args hash handles text/prompt_name isolation. No manual key construction needed.

**Known limitation**: `ainvalidate_cache()` no-ops on parameterized functions (cachekit-io/cachekit-py#59). No other open issues affect key generation or async correctness.

- Consolidates caching from `memory_service.py._get_embeddings()` into the provider layer
- Cache namespace includes `model_name` and `dimensions` (e.g., `mcp_memory_embed_nomic_embed_text_v1.5_768`) — model or dimension change = new namespace = automatic invalidation
- TTL: 86400s (24h). Embeddings are deterministic for a given model+text. Redis `maxmemory-policy allkeys-lru` handles eviction.
- Existing CacheKit wiring for tags, corpus count, and keywords stays in `memory_service.py` (unchanged)

### Multi-Instance Cache Behavior

- L1 (in-process): per-instance, warm after first request
- L2 (Redis): shared across all instances, survives restarts
- Cold start: empty L1, warm L2 = first request costs one Redis RTT (~1ms), not embedding inference (~15ms)
- Known limitation: `ainvalidate_cache()` no-ops on parameterized functions (cachekit-py#59). TTL convergence is the workaround. Acceptable for this use case.
- Model change: new namespace means old L1 entries are orphaned but harmless — they stop being hit and evict on TTL or LRU. L2 (Redis) keys are namespaced, so old-model entries age out naturally.

## Configuration

All via pydantic-settings (`EmbeddingSettings` class in `config.py`):

```bash
# Provider selection (default: local = backward compat)
MCP_EMBEDDING_PROVIDER=local          # local|openai_compat

# Connection (for openai_compat provider)
MCP_EMBEDDING_URL=http://tei:8080     # base URL — validated as AnyHttpUrl, WARNING logged for http://
MCP_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5  # reuses existing var
MCP_EMBEDDING_TIMEOUT=30              # request timeout seconds
MCP_EMBEDDING_MAX_BATCH=64            # max texts per request
MCP_EMBEDDING_TLS_VERIFY=true         # TLS certificate verification (disable only for local dev)

# For managed providers (future)
MCP_EMBEDDING_API_KEY=                # SecretStr
```

`MCP_EMBEDDING_PROVIDER=local` is the default. Existing deployments require zero config changes.

### URL Validation

`MCP_EMBEDDING_URL` is validated at startup via pydantic `AnyHttpUrl`:
- Must be `http://` or `https://` scheme
- Cloud metadata IPs (169.254.x.x) and link-local ranges blocked by default
- `http://` scheme logs a WARNING: "Embedding service URL uses plaintext HTTP. Use HTTPS in production."
- RFC 1918 ranges allowed (common for internal service mesh) but documented as requiring network isolation

## Provider Injection

A single `CachedEmbeddingProvider` instance is created by the factory and shared by both `MemoryService` and `QdrantStorage`. This ensures all embedding calls — regardless of origin — hit the cache.

- **Factory** (`embedding/factory.py`): Reads `MCP_EMBEDDING_PROVIDER`, instantiates the correct adapter, wraps in `CachedEmbeddingProvider`. Returns one instance.
- **QdrantStorage**: Receives the `CachedEmbeddingProvider` via constructor. All internal embedding call sites are replaced:
  - `store()` (line ~834): content embedding via `self._generate_embedding()` → `provider.embed_batch()`
  - `retrieve()` (line ~944): query embedding via `self._generate_embedding()` → `provider.embed_batch()`
  - `count_semantic_search()`: delegates to `retrieve()`, covered transitively
  - `recall_memory()` (line ~1903): same pattern as `retrieve()`
  - `index_new_tags()`: tag embedding generation → `provider.embed_batch()`
  - `generate_embeddings_batch()`: removed entirely (was the public API for this)
  - `_ensure_model_loaded()`, `_generate_embedding()`, `_embedding_model_instance`: deleted. `LocalProvider` owns model lifecycle.
- **MemoryService**: Also receives the same `CachedEmbeddingProvider`. `_get_embeddings()` simplifies to `self.embedding_provider.embed_batch()` — caching is transparent inside the provider, no cache logic in the service layer.

**Key decision**: Both layers share one `CachedEmbeddingProvider` instance. No double-caching. No cache bypass.

### Startup Checks

- Provider factory queries `provider.dimensions` at startup
- Compares against Qdrant collection dimensions
- Fail-fast with clear error on mismatch (e.g., 768-dim provider vs 1024-dim collection)

### Readiness Probe

API service separates liveness from readiness:
- **Liveness** (`/health`): local-only, never calls external services. Returns `{"status": "healthy"}` and status code. No diagnostic info.
- **Readiness** (`/health/ready`): checks embedding provider reachability. Result cached 10s to prevent fan-out amplification.
  - `LocalProvider`: model loaded = ready
  - `OpenAICompatAdapter`: `GET {base_url}/health` returns 200 = ready
- Detailed diagnostics (`/health/detailed`): behind authentication, includes version/uptime/backend info.

### Error Handling

Adapters classify upstream errors:
- **Retryable** (503, 429, timeout): exponential backoff, 3 attempts
- **Fatal** (400, 401, model not found): raise immediately

Cold start cascade (both API and embedding service scaling from zero): adapter retries absorb the embedding service startup time. Docs recommend `min_replicas=1` for the embedding service in production.

## Dockerfile: Single File, Multi-Stage

```dockerfile
FROM python:3.12-slim AS base
# Common: uv, source, spaCy model

FROM base AS full
# torch + sentence-transformers + HF model download
# Default target. Backward compat. Single-node.

FROM base AS api
# No torch, no sentence-transformers
# Thin image for scale-out with external embedding service.
```

```bash
docker build --target api  -t mcp-memory:api  .   # ~200MB, <2s start
docker build --target full -t mcp-memory:full .   # ~2.5GB, ~15s start
docker build .                                     # defaults to full (last stage)
```

Target selection uses `--target`. No build args needed. The `full` stage is last, making it the default when `--target` is omitted.

CI builds both, tags: `ghcr.io/27b-io/mcp-memory-service:latest` (full), `ghcr.io/27b-io/mcp-memory-service:api` (thin).

## Removed Components

| Component | Reason | Replacement |
|-----------|--------|-------------|
| SSE (`web/sse.py`, SSE routes, `SSEManager`) | Stateful (in-process connection dict), unused by MCP consumers. Breaks: events router, SSE test page, dashboard live stats. | None. Clients use request/response. Dashboard simplified to poll-based or removed. |
| asyncio WriteQueue (`web/write_queue.py`) | In-process FIFO serialization, blocks horizontal scaling | `asyncio.Semaphore(20)` for concurrent write backpressure. Qdrant server mode handles write concurrency natively. `WriteQueue.get_stats()` replaced with semaphore counter in health endpoint. |
| Diagnostic deques (`_search_logs`, `_audit_logs` in `MemoryService`) | Per-instance in-process state, unreliable under horizontal scaling | Gate behind `MCP_MEMORY_EXPOSE_DEBUG_TOOLS=true` with per-instance warning. Analytics endpoints become debug-only. |
| `BaseStorage.generate_embeddings_batch()` | Coupling storage + embedding | `EmbeddingProvider` protocol |
| torch/CUDA/device detection in storage | Moves to `LocalProvider` | `LocalProvider` owns model loading |

## Deployment Example

Minimal two-service deployment with Docker Compose (**local development only — production requires TLS and Redis auth**):

```yaml
services:
  api:
    image: ghcr.io/27b-io/mcp-memory-service:api
    environment:
      MCP_EMBEDDING_PROVIDER: openai_compat
      MCP_EMBEDDING_URL: http://tei:8080
      MCP_EMBEDDING_MODEL: nomic-ai/nomic-embed-text-v1.5
      MCP_QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://:${REDIS_PASSWORD:-changeme}@redis:6379
    ports:
      - "8000:8000"

  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest
    command: --model-id nomic-ai/nomic-embed-text-v1.5 --port 8080
    # No host port binding — internal only

  qdrant:
    image: qdrant/qdrant:latest
    # No host port binding — internal only

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD:-changeme}
    # No host port binding — internal only
```

**Security notes**: Only the API service port is exposed to the host. Backend services (TEI, Qdrant, Redis) communicate via Docker's internal network. Redis requires a password. For production: use `https://` URLs, TLS Redis (`rediss://`), and network policies.

## Migration Phases

Each phase is a separate PR. All tests pass at every phase boundary.

### Phase 1: Extract EmbeddingProvider Protocol + LocalProvider

- Define `EmbeddingProvider` protocol in `embedding/protocol.py`
- Move `_ensure_model_loaded()` + `model.encode()` from `QdrantStorage` into `LocalProvider`
- Inject provider into `QdrantStorage` constructor — replace ALL internal embedding call sites with `provider.embed_batch()` delegation:
  - `store()`: content embedding
  - `retrieve()`: query embedding
  - `count_semantic_search()`: transitive via `retrieve()`
  - `recall_memory()`: query embedding
  - `index_new_tags()`: tag embedding
  - `generate_embeddings_batch()`: removed
  - `_ensure_model_loaded()`, `_generate_embedding()`, `_embedding_model_instance`: deleted (lifecycle moves to `LocalProvider`)
- Inject provider into `MemoryService` — `_get_embeddings()` calls provider
- Factory default: `LocalProvider`
- `BaseStorage.generate_embeddings_batch()` deprecated (soft removal)
- **All existing tests pass unchanged. Zero behavior change.**

### Phase 2: OpenAICompatAdapter

- `OpenAICompatAdapter` in `embedding/http.py`
- Speaks `POST /v1/embeddings` (OpenAI schema)
- Handles prompt_name translation per model family
- Retry logic: 3 attempts, exponential backoff on 503/429/timeout
- Integration test: spin up TEI in Docker, embed through adapter
- Factory reads `MCP_EMBEDDING_PROVIDER` env var

### Phase 3: CachedEmbeddingProvider Wrapper

- Decorator wrapping any `EmbeddingProvider` with L1/L2 cache
- Migrates `_cached_embed()` logic from `memory_service.py` into wrapper
- `memory_service.py._get_embeddings()` simplified to `self.embedding_provider.embed_batch()` (caching is transparent)
- TTL bump: 300s -> 86400s
- Startup dimension check: provider dimensions vs Qdrant collection

### Phase 4: Multi-Stage Dockerfile

- Refactor `Dockerfile` into `base`, `full`, `api` stages
- `api` target excludes torch, sentence-transformers
- CI builds both targets
- Readiness probe includes embedding provider health

### Phase 5: Remove Stateful Components

- Delete `web/sse.py`, SSE routes, `SSEManager` (verify access logs for consumers first; removal breaks events router, SSE test page, dashboard live stats)
- Replace asyncio `WriteQueue` (`web/write_queue.py`) with `asyncio.Semaphore(20)`. Qdrant server mode handles concurrent writes natively. Replace `WriteQueue.get_stats()` with semaphore counter in health endpoint.
- Gate `_search_logs` and `_audit_logs` deques behind `MCP_MEMORY_EXPOSE_DEBUG_TOOLS=true` with per-instance warning in analytics response
- **Note**: `HebbianWriteQueue` (graph layer, Redis-backed) is NOT removed — it's already stateless via Redis CQRS.

### Phase 6: Managed Provider Adapters (Future)

- `VertexAIAdapter` (Google SDK) — optional dependency
- Other managed providers as needed
- Not part of initial implementation

## Out of Scope

- **Pluggable storage protocol**: `BaseStorage` cleanup into a lean `StorageProvider` protocol is the right long-term direction but is a separate project. The embedding extraction will reveal where the storage boundary should be.
- **gRPC transport**: Deferred. OpenAI-compat HTTP covers all current backends. gRPC adapter can be added later for latency-sensitive self-hosted deployments.
- **Custom embedding server**: We use off-the-shelf servers (TEI, vLLM, Ollama). No custom embedding service Dockerfile.
- **Rust rewrite / Cloudflare Workers**: The embedding extraction makes the thin API service a stateless proxy — ideal for Rust/WASM on Cloudflare Workers (128MB limit, no filesystem, pure IO). Ship the Python extraction first, prove the architecture, then evaluate a Rust port as a follow-up project. The `EmbeddingProvider` protocol and service boundaries designed here are forward-compatible with a Rust trait-based implementation.

## Success Criteria

1. `MCP_EMBEDDING_PROVIDER=local` behaves identically to current single-process deployment
2. `MCP_EMBEDDING_PROVIDER=openai_compat` with TEI produces embeddings within cosine similarity >0.999 of local SentenceTransformer with same model (floating-point determinism across runtimes is not guaranteed)
3. API service image <300MB, cold start <3s
4. All existing tests pass at every migration phase
5. Docker Compose example boots and serves requests end-to-end
6. CacheKit L2 hit rate >80% for repeat queries under burst load
