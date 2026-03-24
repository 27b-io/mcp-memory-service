# Configuration Reference

All configuration via environment variables. Pydantic-settings under the hood — type-safe, validated, with sensible defaults. Supports `.env` files.

## Quick Reference

| Prefix | Feature | Default State |
|--------|---------|---------------|
| `MCP_MEMORY_` | Core (paths, storage, embedding, debug) | Enabled |
| `MCP_QDRANT_` | Qdrant vector database | Embedded mode |
| `MCP_FALKORDB_` | FalkorDB knowledge graph | Disabled |
| `MCP_SALIENCE_` | Emotional tagging + salience scoring | Enabled |
| `MCP_INTERFERENCE_` | Contradiction detection | Enabled |
| `MCP_SPACED_REPETITION_` | Access-pattern retrieval boost | Enabled |
| `MCP_ENCODING_CONTEXT_` | Context-dependent retrieval | Enabled |
| `MCP_THREE_TIER_` | Sensory/working memory tiers | Enabled (tools hidden) |
| `MCP_INTENT_` | Query intent analysis (spaCy) | Enabled |
| `MCP_SEMANTIC_TAG_` | k-NN semantic tag matching | Enabled |
| `MCP_CROSS_REF_` | Automatic cross-referencing | Enabled |
| `MCP_CONSOLIDATION_` | Graph pruning + maintenance | Always runs |
| `MCP_SUMMARY_` | Auto-summarization | Extractive |
| `MCP_OAUTH_` | OAuth authentication | Disabled |
| `MCP_` (transport) | HTTP server + content limits + TOON | HTTP disabled |

---

## Core Settings

### Paths (`MCP_MEMORY_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_BASE_DIR` | Platform-specific (see below) | Base directory for all MCP memory data |
| `MCP_MEMORY_BACKUPS_PATH` | `<base_dir>/backups` | Path for database backups |

**Platform defaults for `MCP_MEMORY_BASE_DIR`:**

| OS | Default Path |
|----|-------------|
| Linux | `~/.local/share/mcp-memory` (XDG_DATA_HOME) |
| macOS | `~/Library/Application Support/mcp-memory` |
| Windows | `C:\Users\<user>\AppData\Local\mcp-memory` |

Both paths are created with write-permission validation on startup.

### Storage (`MCP_MEMORY_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Sentence-transformers model for embedding generation |
| `MCP_MEMORY_USE_ONNX` | `false` | Use ONNX runtime for embeddings (PyTorch-free) |

**Available embedding models (local provider):**

| Model | Dims | Notes |
|-------|------|-------|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Default — best balance, 8K context, ~62 MTEB avg |
| `intfloat/e5-base-v2` | 768 | Alternative, shorter context |
| `intfloat/e5-small-v2` | 384 | Speed over accuracy |
| `intfloat/e5-large-v2` | 1024 | Best quality |

### Embedding Provider (`MCP_EMBEDDING_` prefix)

Controls how embeddings are generated. Default is `local` (in-process via sentence-transformers). Set to `openai_compat` for external providers like TEI, vLLM, or any OpenAI-compatible `/v1/embeddings` endpoint.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_EMBEDDING_PROVIDER` | `local` | Provider type: `local` (in-process) or `openai_compat` (external HTTP) |
| `MCP_EMBEDDING_URL` | `null` | Base URL for HTTP provider (e.g., `http://tei:80`). Required when provider is `openai_compat`. |
| `MCP_EMBEDDING_DIMENSIONS` | auto-detected | Vector dimensions. Auto-detected for local provider; must be set explicitly for `openai_compat`. |
| `MCP_EMBEDDING_TIMEOUT` | `30` | Request timeout in seconds (1–300) |
| `MCP_EMBEDDING_MAX_BATCH` | `64` | Max texts per batch request (1–1024) |
| `MCP_EMBEDDING_TLS_VERIFY` | `true` | TLS certificate verification for HTTP provider |
| `MCP_EMBEDDING_API_KEY` | `null` | **Sensitive.** API key for managed embedding providers |

**Example: TEI (HuggingFace Text Embeddings Inference):**

```bash
MCP_EMBEDDING_PROVIDER=openai_compat
MCP_EMBEDDING_URL=http://tei:80
MCP_EMBEDDING_DIMENSIONS=1024
MCP_MEMORY_EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0
```

See the [Kubernetes deployment guide](deployment.md#tei-text-embeddings-inference) for TEI configuration and tuning.

### Content Limits (`MCP_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLE_AUTO_SPLIT` | `true` | Automatically split content that exceeds limits |
| `MCP_CONTENT_SPLIT_OVERLAP` | `50` | Character overlap between split chunks (0–500) |
| `MCP_CONTENT_PRESERVE_BOUNDARIES` | `true` | Respect sentence/paragraph boundaries when splitting |

### Security (`MCP_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_API_KEY` | `null` | **Sensitive.** API key for HTTP authentication. No auth if unset. |
| `MCP_CORS_ORIGINS` | `[]` | Allowed CORS origins. Comma-separated string or JSON list. Empty = no cross-origin access. |

---

## Transport & HTTP

### HTTP Server (`MCP_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HTTP_ENABLED` | `false` | Enable the HTTP/REST server |
| `MCP_HTTP_HOST` | `0.0.0.0` | Bind address for HTTP server |
| `MCP_HTTP_PORT` | `8000` | HTTP port (1024–65535) |
| `MCP_HTTP_WORKERS` | `1` | Uvicorn worker processes (1–32) |
| `MCP_HTTPS_ENABLED` | `false` | Enable HTTPS |
| `MCP_SSL_CERT_FILE` | `null` | Path to TLS certificate file |
| `MCP_SSL_KEY_FILE` | `null` | Path to TLS private key file |
| `SSE_HEARTBEAT_INTERVAL` | `30` | SSE keepalive interval in seconds (5–300) |

### OAuth (`MCP_OAUTH_` prefix)

OAuth is disabled by default. The `authlib`/`python-jose` dependencies were removed due to critical CVEs — enabling OAuth requires reinstalling them.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_OAUTH_ENABLED` | `false` | Enable OAuth 2.1 authentication |
| `MCP_OAUTH_PRIVATE_KEY` | `null` | **Sensitive.** PEM-encoded RSA private key for RS256. Auto-generated if OAuth enabled and not set. |
| `MCP_OAUTH_PUBLIC_KEY` | `null` | PEM-encoded RSA public key for RS256. Auto-generated alongside private key. |
| `MCP_OAUTH_SECRET_KEY` | `null` | **Sensitive.** Symmetric key for HS256 fallback (used when RSA unavailable). |
| `MCP_OAUTH_ISSUER` | Auto-derived | JWT issuer URL. Auto-set from HTTP host/port if not provided. |
| `MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token lifetime in minutes (1–1440) |
| `MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES` | `10` | Auth code lifetime in minutes (1–60) |
| `MCP_ALLOW_ANONYMOUS_ACCESS` | `false` | Allow unauthenticated requests when OAuth is enabled |

**Key selection logic:** RS256 (RSA key pair) takes precedence over HS256 (symmetric). If neither is provided at startup and OAuth is enabled, RSA keys are auto-generated (ephemeral — not persisted across restarts).

---

## Storage Backends

### Qdrant (`MCP_QDRANT_` prefix)

Qdrant is the sole storage backend. It runs embedded by default — no external service required.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_QDRANT_URL` | `null` | Qdrant server URL (e.g., `http://localhost:6333`). Set to use network mode instead of embedded. |
| `MCP_QDRANT_STORAGE_PATH` | `<base_dir>/qdrant` | Embedded mode storage directory. Created with `0o700` permissions. Ignored in network mode. |
| `MCP_QDRANT_COLLECTION_NAME` | `memories` | Qdrant collection name. Change when using a different embedding model/dimension to avoid conflicts. |
| `MCP_QDRANT_QUANTIZATION_ENABLED` | `false` | Enable scalar quantization (~32x memory savings, ~10% slower retrieval) |

**Internal constants (not configurable):**

| Constant | Value | Notes |
|----------|-------|-------|
| Distance metric | `Cosine` | |
| HNSW M | `16` | Edges per node — balanced quality/speed |
| HNSW ef_construct | `100` | Build-time quality |
| HNSW ef | `128` | Search-time recall |
| Full-scan threshold | `10000` | Brute force below this count |
| On-disk payload | `false` | Payload kept in memory |
| Indexing threshold | `20000` | HNSW index built after this many vectors |

### FalkorDB (`MCP_FALKORDB_` prefix)

Adds a graph layer over Qdrant for Hebbian learning and spreading activation. Disabled by default.

**Connection:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_FALKORDB_ENABLED` | `false` | Enable FalkorDB graph layer |
| `MCP_FALKORDB_HOST` | `localhost` | FalkorDB host (Redis protocol) |
| `MCP_FALKORDB_PORT` | `6379` | FalkorDB port (1–65535) |
| `MCP_FALKORDB_PASSWORD` | `null` | **Sensitive.** FalkorDB/Redis password |
| `MCP_FALKORDB_GRAPH_NAME` | `memory_graph` | Graph name within FalkorDB |
| `MCP_FALKORDB_MAX_CONNECTIONS` | `16` | Redis connection pool size (1–128) |

**Write queue (CQRS — Redis LPUSH/BRPOP):**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_FALKORDB_WRITE_QUEUE_KEY` | `mcp:graph:write_queue` | Redis key for async Hebbian write queue |
| `MCP_FALKORDB_WRITE_QUEUE_BATCH_SIZE` | `50` | Max edges processed per consumer tick (1–500) |
| `MCP_FALKORDB_WRITE_QUEUE_POLL_INTERVAL` | `0.5` | Seconds between BRPOP polls when queue empty (0.1–10.0) |

**Hebbian learning:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_FALKORDB_HEBBIAN_INITIAL_WEIGHT` | `0.1` | Initial weight for new co-access edges (0.01–1.0) |
| `MCP_FALKORDB_HEBBIAN_STRENGTHEN_RATE` | `0.15` | Multiplicative strengthen rate per co-access: `w *= (1 + rate)` (0.01–1.0) |
| `MCP_FALKORDB_HEBBIAN_MAX_WEIGHT` | `1.0` | Maximum Hebbian edge weight (0.1–10.0) |
| `MCP_FALKORDB_HEBBIAN_BOOST` | `0.15` | Max boost from Hebbian co-access edges on search scores. `0` = disabled (0.0–1.0) |

**Spreading activation:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_FALKORDB_SPREADING_ACTIVATION_MAX_HOPS` | `2` | BFS depth for spreading activation (1–3) |
| `MCP_FALKORDB_SPREADING_ACTIVATION_DECAY` | `0.5` | Per-hop exponential decay: `activation *= decay^hops` (0.01–1.0) |
| `MCP_FALKORDB_SPREADING_ACTIVATION_BOOST` | `0.2` | Weight of graph activation boost on vector scores (0.0–1.0) |
| `MCP_FALKORDB_SPREADING_ACTIVATION_MIN_ACTIVATION` | `0.01` | Minimum activation to consider a graph neighbor (0.0–1.0) |

---

## Cognitive Features

### Salience Scoring (`MCP_SALIENCE_` prefix)

Computes a 0–1 salience score per memory from emotional magnitude, access frequency, and explicit importance. Applied as a multiplicative retrieval boost (max +15% by default).

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SALIENCE_ENABLED` | `true` | Enable emotional tagging and salience scoring |
| `MCP_SALIENCE_EMOTIONAL_WEIGHT` | `0.3` | Emotional magnitude contribution to salience score (0.0–1.0) |
| `MCP_SALIENCE_FREQUENCY_WEIGHT` | `0.3` | Access frequency contribution to salience score (0.0–1.0) |
| `MCP_SALIENCE_IMPORTANCE_WEIGHT` | `0.4` | Explicit importance metadata contribution to salience score (0.0–1.0) |
| `MCP_SALIENCE_BOOST_WEIGHT` | `0.15` | Maximum salience boost applied to retrieval scores (0.0–1.0, i.e., up to +15%) |

Weights do not need to sum to 1.0, but that's recommended for interpretability.

### Interference Detection (`MCP_INTERFERENCE_` prefix)

Checks incoming memories at store time for potential contradictions with existing memories. Contradicting pairs are linked with `CONTRADICTS` edges in the graph layer.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_INTERFERENCE_ENABLED` | `true` | Enable contradiction detection at store time |
| `MCP_INTERFERENCE_SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity to flag as potentially contradictory (0.5–0.95) |
| `MCP_INTERFERENCE_MIN_CONFIDENCE` | `0.3` | Minimum confidence for a contradiction signal to be reported (0.1–0.9) |
| `MCP_INTERFERENCE_MAX_CANDIDATES` | `5` | Maximum similar memories checked per store operation (1–20) |

### Spaced Repetition (`MCP_SPACED_REPETITION_` prefix)

Implements the spacing effect (Ebbinghaus, 1885): memories accessed at increasing intervals receive stronger retrieval boosts than those accessed in rapid bursts.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SPACED_REPETITION_ENABLED` | `true` | Enable spaced repetition and adaptive LTP |
| `MCP_SPACED_REPETITION_BOOST_WEIGHT` | `0.1` | Maximum spacing quality boost on retrieval scores (0.0–1.0, i.e., up to +10%) |
| `MCP_SPACED_REPETITION_MAX_TIMESTAMPS` | `20` | Maximum access timestamps retained per memory in ring buffer (5–100) |

### Encoding Context (`MCP_ENCODING_CONTEXT_` prefix)

Implements encoding specificity (Tulving & Thomson, 1973): memories are retrieved more effectively when retrieval context matches encoding context.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENCODING_CONTEXT_ENABLED` | `true` | Enable encoding context capture and context-dependent retrieval |
| `MCP_ENCODING_CONTEXT_BOOST_WEIGHT` | `0.1` | Maximum context similarity boost on retrieval scores (0.0–1.0, i.e., up to +10%) |

### Cross-Referencing (`MCP_CROSS_REF_` prefix)

At store time, finds semantically related memories and creates `RELATES_TO` edges. Complements interference detection: interference creates `CONTRADICTS` edges; cross-referencing creates `RELATES_TO` edges.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_CROSS_REF_ENABLED` | `true` | Enable automatic `RELATES_TO` edge creation at store time |
| `MCP_CROSS_REF_SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity to create a cross-reference (0.3–0.9) |
| `MCP_CROSS_REF_MAX_CANDIDATES` | `5` | Maximum similar memories checked per store operation (1–20) |

---

## Search & Retrieval

### Hybrid Search (`MCP_MEMORY_` prefix)

Controls blending of vector similarity and tag-based search, plus temporal decay.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_HYBRID_ALPHA` | `null` | Vector vs tag weight: `0.0`=tags only, `1.0`=vector only, `null`=adaptive based on corpus size |
| `MCP_MEMORY_RECENCY_DECAY` | `0.01` | Exponential decay rate for recency boost. `0`=disabled, `0.01`=~70-day half-life |
| `MCP_MEMORY_TEMPORAL_DECAY_LAMBDA` | `0.0` | Temporal decay rate applied to scores. `0`=disabled, `0.01`=~69-day half-life |
| `MCP_MEMORY_TEMPORAL_DECAY_BASE` | `0.7` | Minimum relevance floor after temporal decay (0.0–1.0, i.e., 70% retention) |
| `MCP_MEMORY_ADAPTIVE_THRESHOLD_SMALL` | `500` | Corpus size below which `alpha=0.5` (balanced vector+tag) |
| `MCP_MEMORY_ADAPTIVE_THRESHOLD_LARGE` | `5000` | Corpus size above which `alpha=0.8` (strong semantic) |

**Adaptive alpha logic:** When `MCP_MEMORY_HYBRID_ALPHA=null`, alpha scales linearly from 0.5 → 0.8 between the two thresholds.

### Query Intent (`MCP_INTENT_` prefix)

Extracts concepts from queries using spaCy, then fans out to multiple sub-queries for broader recall. Also injects graph neighbors into the candidate pool.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_INTENT_ENABLED` | `true` | Enable query intent inference and fan-out |
| `MCP_INTENT_SPACY_MODEL` | `en_core_web_sm` | spaCy model for concept extraction |
| `MCP_INTENT_MAX_SUB_QUERIES` | `4` | Maximum sub-queries generated from concept extraction (1–8) |
| `MCP_INTENT_MIN_QUERY_TOKENS` | `3` | Minimum tokens to trigger fan-out; shorter queries use single-vector search |
| `MCP_INTENT_GRAPH_INJECT` | `true` | Inject graph neighbors into candidate pool |
| `MCP_INTENT_GRAPH_INJECT_LIMIT` | `10` | Maximum graph-injected neighbors (1–50) |
| `MCP_INTENT_GRAPH_INJECT_MIN_ACTIVATION` | `0.05` | Minimum spreading activation score for graph injection (0.0–1.0) |
| `MCP_INTENT_LLM_RERANK` | `false` | Enable LLM-based re-ranking of results |
| `MCP_INTENT_LLM_PROVIDER` | `anthropic` | LLM provider for re-ranking |
| `MCP_INTENT_LLM_MODEL` | `claude-haiku-4-5-20251001` | LLM model for re-ranking |
| `MCP_INTENT_LLM_TIMEOUT_MS` | `2000` | LLM re-ranking timeout in milliseconds (500–10000) |

### Semantic Tags (`MCP_SEMANTIC_TAG_` prefix)

Expands tag-based queries using embedding k-NN to find semantically similar tags, not just exact matches.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SEMANTIC_TAG_ENABLED` | `true` | Enable semantic tag matching via embedding k-NN |
| `MCP_SEMANTIC_TAG_SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity for a tag match (0.0–1.0) |
| `MCP_SEMANTIC_TAG_MAX_TAGS` | `10` | Maximum semantically matched tags to fan out (1–50) |
| `MCP_SEMANTIC_TAG_CACHE_TTL` | `3600` | Tag embedding cache TTL in seconds (60+) |

---

## Optional Features

### Three-Tier Memory (`MCP_THREE_TIER_` prefix)

In-process, session-scoped sensory buffer and working memory on top of Qdrant long-term storage. Both tiers are not persisted across restarts.

> **Note:** These are client-side concerns. The 1-second sensory TTL is shorter than a single MCP round-trip, making it largely academic for LLM consumers. Expose tools only for autonomous agent rigs that process continuously.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_THREE_TIER_ENABLED` | `true` | Enable three-tier memory model |
| `MCP_THREE_TIER_EXPOSE_TOOLS` | `false` | Expose sensory/working memory as MCP tools |
| `MCP_THREE_TIER_SENSORY_CAPACITY` | `7` | Sensory buffer item capacity — Miller's magic number (1–50) |
| `MCP_THREE_TIER_SENSORY_DECAY_MS` | `1000` | Sensory buffer item TTL in milliseconds (100–30000) |
| `MCP_THREE_TIER_WORKING_CAPACITY` | `4` | Working memory capacity — Cowan's limit (1–20) |
| `MCP_THREE_TIER_WORKING_DECAY_MINUTES` | `30.0` | Working memory item decay in minutes (1.0–1440.0) |
| `MCP_THREE_TIER_AUTO_CONSOLIDATE` | `true` | Auto-consolidate working memory items to LTM on access threshold |

### Summary Generation (`MCP_SUMMARY_` prefix)

Controls how the ~50-token memory summaries are generated. Defaults to extractive (no LLM calls, no API cost).

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SUMMARY_MODE` | `null` | Summary mode: `extractive`, `llm`, or `null` (auto-detect) |
| `MCP_SUMMARY_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `gemini` |
| `MCP_SUMMARY_MAX_TOKENS` | `50` | Maximum output tokens for LLM-generated summaries (10–200) |
| `MCP_SUMMARY_TIMEOUT_SECONDS` | `5.0` | HTTP timeout for LLM API calls (1.0–30.0) |

**Auto-detect logic (`mode=null`):** Uses LLM mode if Anthropic API key is set or a non-default base URL is configured (proxy), or if Gemini API key is set. Otherwise, falls back to extractive.

**Anthropic provider:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SUMMARY_ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API base URL. Override for proxy/load balancer. |
| `MCP_SUMMARY_ANTHROPIC_API_KEY` | `null` | **Sensitive.** Anthropic API key. Optional when using a proxy that injects auth. |
| `MCP_SUMMARY_ANTHROPIC_MODEL_SMALL` | `claude-3-5-haiku-20241022` | Model for short memories (< threshold chars) |
| `MCP_SUMMARY_ANTHROPIC_MODEL_LARGE` | `claude-3-5-sonnet-20241022` | Model for long memories (>= threshold chars) |
| `MCP_SUMMARY_ANTHROPIC_SIZE_THRESHOLD` | `500` | Character count threshold for small/large model switch (100–2000) |

**Gemini provider (legacy):**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SUMMARY_MODEL` | `gemini-2.5-flash` | Gemini model identifier |
| `MCP_SUMMARY_API_KEY` | `null` | **Sensitive.** Gemini API key. Required for Gemini provider. |

### Consolidation (`MCP_CONSOLIDATION_` prefix)

Periodic graph maintenance: decay stale Hebbian edges, prune weak ones, and merge near-duplicate memories.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_CONSOLIDATION_DECAY_FACTOR` | `0.9` | Global edge weight decay per run (synaptic homeostasis) (0.01–0.99) |
| `MCP_CONSOLIDATION_PRUNE_THRESHOLD` | `0.05` | Delete edges with weight below this after decay (0.001–0.5) |
| `MCP_CONSOLIDATION_STALE_EDGE_DAYS` | `30` | Edges not co-accessed within this many days receive extra decay (1–365) |
| `MCP_CONSOLIDATION_STALE_DECAY_FACTOR` | `0.5` | Additional decay multiplier for stale edges, applied on top of global decay (0.01–0.99) |
| `MCP_CONSOLIDATION_MAX_EDGES_PER_RUN` | `10000` | Safety cap on edges processed per consolidation run (100–100000) |
| `MCP_CONSOLIDATION_DUPLICATE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity above which memories are considered duplicates (0.8–1.0) |
| `MCP_CONSOLIDATION_MAX_DUPLICATES_PER_RUN` | `100` | Maximum duplicate pairs merged per run (1–1000) |

---

## Output & Debug

### TOON Format (`MCP_` prefix)

TOON is the pipe-delimited output format that reduces token consumption in tool responses.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLE_TOON_FORMAT` | `true` | Enable TOON format encoding. Set `false` to revert to JSON (emergency kill switch). |
| `MCP_LOG_TOKEN_SAVINGS` | `false` | Log token savings metrics during TOON encoding |

### Debug (`MCP_MEMORY_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_EXPOSE_DEBUG_TOOLS` | `false` | Expose internal debug MCP tools |
| `MCP_MEMORY_INCLUDE_HOSTNAME` | `false` | Include hostname in tool responses |
| `MCP_MEMORY_LATENCY_METRICS` | `false` | Include `latency_ms` field in tool responses |

---

## .env File Support

All variables can be placed in a `.env` file in the working directory. Environment variables take precedence over `.env` values. Both are case-insensitive.

```bash
# Example .env
MCP_MEMORY_BASE_DIR=/data/mcp-memory
MCP_HTTP_ENABLED=true
MCP_HTTP_PORT=8080
MCP_API_KEY=your-secret-key
MCP_QDRANT_URL=http://qdrant:6333
MCP_FALKORDB_ENABLED=true
MCP_FALKORDB_HOST=falkordb
MCP_SUMMARY_MODE=llm
MCP_SUMMARY_ANTHROPIC_API_KEY=sk-ant-...
```
