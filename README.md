# MCP Memory Service

Semantic memory with teeth. A Model Context Protocol server that gives AI assistants persistent, searchable memory backed by vector search, a knowledge graph, and cognitive science you didn't ask for but definitely want.

## Quick Start

### Docker + HTTP (Recommended)

Run the server, point your MCP client at it. Done.

```bash
docker run -d \
  -p 8001:8001 \
  -v memory-data:/app/data \
  -e MCP_TRANSPORT_MODE=http \
  -e MCP_SERVER_PORT=8001 \
  ghcr.io/27b-io/mcp-memory-service:latest
```

Then in your Claude Code or Claude Desktop config:

```json
{
  "mcpServers": {
    "memory": {
      "type": "http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

The server speaks streamable HTTP natively via FastMCP 2.0 — no `mcp-remote` bridge, no `npx` shims, no Node.js. Your MCP client connects directly.

### From Source (stdio)

For local development or if you prefer the MCP stdio transport:

```bash
git clone https://github.com/27b-io/mcp-memory-service.git
cd mcp-memory-service
uv sync
uv run memory server
```

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-memory-service", "memory", "server"]
    }
  }
}
```

## Architecture

```text
                         ┌─────────────────────────────────┐
  Claude Desktop         │        MCP Memory Service       │
  Claude Code     ──MCP──│                                 │
  VS Code / Cursor       │  ┌───────────────────────────┐  │
  Any MCP Client         │  │     9 MCP Tools            │  │
                         │  │  store · search · delete   │  │
                         │  │  health · relation         │  │
                         │  │  supersede · contradictions │  │
                         │  │  find_dupes · merge_dupes  │  │
                         │  └─────────┬─────────────────┘  │
                         │            │                     │
                         │  ┌─────────▼─────────────────┐  │
                         │  │    Memory Service          │  │
                         │  │                            │  │
                         │  │  Hybrid RRF Search         │  │
                         │  │  Emotional Analysis        │  │
                         │  │  Interference Detection    │  │
                         │  │  Salience Scoring          │  │
                         │  │  Spaced Repetition         │  │
                         │  │  Three-Tier Memory         │  │
                         │  └──┬──────────────────┬─────┘  │
                         │     │                  │         │
                         │  ┌──▼───────┐  ┌──────▼──────┐  │
                         │  │  Qdrant  │  │  FalkorDB   │  │
                         │  │ (vector) │  │  (graph)    │  │
                         │  └──────────┘  └─────────────┘  │
                         └─────────────────────────────────┘
```

Every memory stored gets vectorized, emotionally tagged, checked for contradictions against existing memories, and scored for salience. Retrieval uses Reciprocal Rank Fusion to blend semantic similarity with tag matching, then applies recency decay, Hebbian co-access boosting, and spaced repetition signals. It's a lot happening behind a simple `store_memory` / `search` interface.

## MCP Tools

Nine tools. Down from 16 in earlier versions — because your LLM's context window isn't free real estate.

| Tool | What it does |
|------|-------------|
| **store_memory** | Persist content with tags, metadata, and automatic cognitive processing (emotional analysis, contradiction detection, salience scoring). Auto-splits large content. |
| **search** | Unified retrieval across 5 modes: `hybrid`, `scan`, `similar`, `tag`, `recent`. One tool, multiple strategies. |
| **delete_memory** | Permanently remove a memory by content hash. |
| **check_database_health** | Backend status, memory count, storage stats. |
| **relation** | Create, query, or delete typed edges in the knowledge graph (RELATES_TO, PRECEDES, CONTRADICTS). |
| **memory_supersede** | Mark a memory as superseded by a newer one. Old memory excluded from search, audit trail preserved. |
| **memory_contradictions** | List unresolved contradiction pairs for review. |
| **find_duplicates** | Scan memories for near-duplicates using embedding cosine similarity. Configurable threshold and canonical selection strategy. |
| **merge_duplicates** | Supersede duplicate memories in favour of a canonical one. Non-destructive — originals kept for audit. |

### Search Modes

The `search` tool's `mode` parameter selects the retrieval strategy:

| Mode | When to use it | Requires query? |
|------|---------------|-----------------|
| **hybrid** (default) | General-purpose. Semantic search + tag boosting with adaptive corpus-aware weighting. | Yes |
| **scan** | Token-budget triage. Returns ~50-token summaries instead of full content. | Yes |
| **similar** | Duplicate detection. Pure k-NN vector similarity, no tag boosting. | Yes |
| **tag** | Category lookup. Exact tag matching with AND/OR logic. | No (needs `tags`) |
| **recent** | Timeline browsing. Newest first, optional tag/type filter. | No |

Results come back in TOON format (pipe-delimited, ~83% smaller than JSON) for `hybrid`, `tag`, and `recent` modes. `scan` and `similar` return structured dicts.

### Three-Tier Memory (Optional)

Behind `MCP_THREE_TIER_EXPOSE_TOOLS=true`, five additional tools expose a Cowan's embedded-processes memory model:

| Tier | Capacity | TTL | Purpose |
|------|----------|-----|---------|
| Sensory Buffer | ~7 items | 1 second | Raw input capture (ring buffer) |
| Working Memory | ~4 chunks | 30 minutes | Active task context (LRU eviction) |
| Long-Term | Unlimited | Permanent | Qdrant persistent storage |

Items accessed 2+ times in working memory auto-consolidate to long-term storage. These tiers are ephemeral and session-scoped — they don't touch the database until consolidation fires.

## Cognitive Features

These run automatically on every `store_memory` and `search` call. No configuration needed (but you can tune or disable each one).

**Emotional Analysis** — Keyword-based sentiment detection across 8 categories (joy, frustration, urgency, curiosity, concern, excitement, sadness, confidence). Handles negation and intensifiers. No ML model needed.

**Salience Scoring** — Weighted combination of emotional magnitude, access frequency, and explicit importance. High-salience memories get a retrieval boost (up to +15%).

**Interference Detection** — At store time, new memories are checked against existing ones for contradictions via four signal types: negation asymmetry, antonym pairs, temporal supersession ("no longer X, now Y"), and sentiment flips. Contradictions are flagged (not blocked) and recorded as CONTRADICTS edges in the knowledge graph.

**Spaced Repetition** — Memories accessed at expanding intervals (1h, 1d, 1w) get a retrieval boost. Based on Ebbinghaus spacing effect. Adaptive LTP prevents runaway potentiation (BCM theory).

**Hebbian Learning** — Co-retrieved memories form implicit association edges in the knowledge graph. Weights strengthen on co-access and decay over time. Used for spreading activation during search (BFS up to 2 hops).

**Hybrid Search (RRF)** — Reciprocal Rank Fusion blends vector similarity with tag matching. Alpha adapts to corpus size: balanced at <500 memories, semantic-biased at 5000+. Optional recency decay with configurable half-life (~70 days default).

## Knowledge Graph

Optional FalkorDB layer for typed relationships between memories.

**Explicit edges** (created via `relation` tool):
- `RELATES_TO` — generic semantic link
- `PRECEDES` — temporal/causal ordering
- `CONTRADICTS` — conflicting information (also auto-created by interference detection)

**Implicit edges** (created automatically):
- `HEBBIAN` — co-access associations, weight modulated by adaptive LTP

Graph features: spreading activation search boost, consolidation pruning, async write queue (CQRS pattern to avoid blocking reads).

## TOON Format

Terser Object Notation — a pipe-delimited output format optimized for LLM token budgets.

```text
# page=1 total=42 page_size=10 has_more=true total_pages=5
Meeting notes about auth redesign|planning,auth|{"priority":"high"}|2026-01-15T10:30:00Z|2026-01-15T10:30:00Z|abc123|0.95
Fixed race condition in worker pool|bugfix,concurrency|{}|2026-01-14T09:00:00Z|2026-01-14T09:00:00Z|def456|0.87
```

~83% fewer tokens than equivalent JSON. The format spec is available as an MCP resource at `toon://format/documentation`.

## Docker Image Variants

The Dockerfile produces two targets:

| Target | Size | Embedding | Use case |
|--------|------|-----------|----------|
| **full** (default) | ~1.2 GB | In-process SentenceTransformer | Single-node, `docker compose up` |
| **slim** | ~200 MB | External (TEI/vLLM/Ollama) | Scale-out, separate embedding service |

`docker build .` produces the `full` image (backward compatible). For the slim image:

```bash
# Build locally
docker build --target slim -t mcp-memory:slim .

# Or pull from GHCR
docker pull ghcr.io/27b-io/mcp-memory-service:latest-slim
```

### Split Deployment (API + External Embedding)

For production scale-out, use `docker-compose.prod.yml` which deploys a thin API container with a separate HuggingFace TEI embedding service:

```bash
docker compose -f docker-compose.prod.yml up
```

This gives you:
- **api** — Thin MCP service (~200MB, <2s cold start, no GPU needed)
- **embedding** — HuggingFace TEI serving the embedding model
- **qdrant** — Vector database

Only the API port (8000) is exposed to the host. Backend services communicate on the internal Docker network.

## Configuration

All config via environment variables. Pydantic-settings under the hood — type-safe, validated, with sensible defaults.

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MEMORY_STORAGE_BACKEND` | `qdrant` | Storage backend |
| `MCP_MEMORY_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace model (768-dim, 8K context) |
| `MCP_MEMORY_USE_ONNX` | `false` | ONNX runtime for CPU optimization |
| `MCP_QDRANT_URL` | — | Remote Qdrant server (omit for embedded mode) |
| `MCP_QDRANT_STORAGE_PATH` | Platform default | Local Qdrant data path |

### Embedding Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_EMBEDDING_PROVIDER` | `local` | `local` (in-process) or `openai_compat` (HTTP) |
| `MCP_EMBEDDING_URL` | — | Base URL for HTTP embedding service (required for `openai_compat`) |
| `MCP_EMBEDDING_TIMEOUT` | `30` | Request timeout in seconds (1-300) |
| `MCP_EMBEDDING_MAX_BATCH` | `64` | Max texts per batch request (1-1024) |
| `MCP_EMBEDDING_DIMENSIONS` | auto-detect | Override embedding dimensions |
| `MCP_EMBEDDING_TLS_VERIFY` | `true` | TLS certificate verification for HTTP provider |
| `MCP_EMBEDDING_API_KEY` | — | API key for managed embedding providers |

### Cognitive Features

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SALIENCE_ENABLED` | `true` | Emotional tagging + salience scoring |
| `MCP_INTERFERENCE_ENABLED` | `true` | Contradiction detection at store time |
| `MCP_SPACED_REPETITION_ENABLED` | `true` | Spacing-quality retrieval boost |
| `MCP_ENCODING_CONTEXT_ENABLED` | `true` | Context-dependent retrieval |
| `MCP_MEMORY_HYBRID_ALPHA` | — | Override adaptive alpha (0.0-1.0) |
| `MCP_MEMORY_RECENCY_DECAY` | `0.01` | Temporal decay rate (~70-day half-life) |

### Three-Tier Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_THREE_TIER_ENABLED` | `true` | Enable sensory/working memory tiers |
| `MCP_THREE_TIER_EXPOSE_TOOLS` | `false` | Register as MCP tools |
| `MCP_THREE_TIER_SENSORY_CAPACITY` | `7` | Sensory buffer size |
| `MCP_THREE_TIER_WORKING_CAPACITY` | `4` | Working memory slots |

### Content Handling

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLE_AUTO_SPLIT` | `true` | Auto-chunk large content |
| `MCP_CONTENT_SPLIT_OVERLAP` | `50` | Chunk overlap (chars) |
| `MCP_CONTENT_PRESERVE_BOUNDARIES` | `true` | Respect sentence/paragraph breaks |

### HTTP Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HTTP_ENABLED` | `false` | Enable HTTP/REST interface |
| `MCP_HTTP_PORT` | `8000` | Listen port |
| `MCP_HTTP_HOST` | `0.0.0.0` | Bind address |

### Summary Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SUMMARY_MODE` | auto-detect | `extractive` or `llm` |
| `MCP_SUMMARY_PROVIDER` | `anthropic` | `anthropic` or `gemini` |

## Storage Backends

| Backend | Best for | Notes |
|---------|----------|-------|
| **Qdrant** | All deployments | HNSW index, scalar quantization option, embedded or remote mode. Default and only backend. |

Qdrant runs in embedded mode (zero config) or connects to a remote server via `MCP_QDRANT_URL`.

## Development

```text
src/mcp_memory_service/
├── mcp_server.py              # 9 MCP tools (+ optional three-tier)
├── config.py                  # Pydantic settings
├── memory_tiers.py            # Cowan's three-tier model
├── services/
│   └── memory_service.py      # Core business logic
├── embedding/
│   ├── protocol.py            # EmbeddingProvider protocol
│   ├── local.py               # In-process SentenceTransformer
│   ├── http.py                # OpenAI-compatible HTTP adapter
│   ├── cached.py              # CacheKit L1/L2 wrapper
│   └── factory.py             # Provider factory
├── storage/
│   ├── base.py                # Storage protocol (ABC)
│   ├── qdrant_storage.py      # Qdrant backend
│   └── factory.py             # Backend factory
├── graph/
│   ├── schema.py              # FalkorDB Cypher schema
│   ├── client.py              # Graph operations
│   └── queue.py               # Async write queue
├── formatters/
│   └── toon.py                # TOON encoder
├── utils/
│   ├── emotional_analysis.py  # Sentiment detection
│   ├── interference.py        # Contradiction signals
│   ├── salience.py            # Importance scoring
│   ├── spaced_repetition.py   # Spacing + LTP
│   ├── hybrid_search.py       # RRF + adaptive alpha
│   ├── content_splitter.py    # Auto-chunking
│   └── summariser.py          # LLM/extractive summaries
└── models/
    └── memory.py              # Memory dataclass
```

### Running Tests

```bash
# Full suite
uv run pytest tests/

# Fast (skip slow integration tests)
uv run pytest -x -m "not slow"

# Quality gates
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| **nomic-ai/nomic-embed-text-v1.5** | 768 | Default. 8K context, good balance. |
| intfloat/e5-small-v2 | 384 | Speed over accuracy |
| intfloat/e5-base-v2 | 768 | Previous default |
| intfloat/e5-large-v2 | 1024 | Best quality |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 768 | Alternative |

Switching models requires re-embedding existing memories.

## License

[WTFPL](LICENSE) — Do What The Fuck You Want To Public License.
