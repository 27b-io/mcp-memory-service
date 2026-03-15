# CLAUDE.md

MCP Memory Service - Semantic memory server for Claude with Qdrant vector search and FalkorDB knowledge graph.

## Quick Start

```bash
# Start server
uv run memory server

# Run tests
pytest tests/

# Check status
uv run memory status
```

## Storage Backend

| Backend | Performance | Use Case |
|---------|-------------|----------|
| **Qdrant** | 5ms read | Embedded (default) or remote server |

Remote Qdrant via env: `MCP_QDRANT_URL=http://your-qdrant:6333`

## Architecture

```text
mcp_memory_service/
├── mcp_server.py         # FastMCP 2.0 — 9 MCP tools
├── unified_server.py     # HTTP + MCP dual-mode server
├── services/
│   └── memory_service.py # Core business logic (~3000 lines)
├── storage/
│   ├── base.py           # MemoryStorage ABC
│   ├── qdrant_storage.py # Qdrant backend
│   └── factory.py        # Backend factory
├── graph/
│   ├── client.py         # FalkorDB graph operations
│   ├── schema.py         # Cypher schema
│   └── queue.py          # Async write queue (CQRS)
├── models/
│   └── memory.py         # Pydantic v2 Memory model
├── formatters/
│   └── toon.py           # TOON pipe-delimited encoder
├── utils/
│   ├── emotional_analysis.py
│   ├── interference.py
│   ├── salience.py
│   ├── spaced_repetition.py
│   ├── hybrid_search.py
│   ├── content_splitter.py
│   ├── query_intent.py
│   └── summariser.py
├── config.py             # pydantic-settings (23 settings classes)
└── cli/main.py           # CLI entry point
```

## Embedding Models

Default: `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8K context, ~62 MTEB avg)

| Model | Dims | Use Case |
|-------|------|----------|
| **nomic-ai/nomic-embed-text-v1.5** | 768 | Default (best balance, 8K context) |
| **intfloat/e5-base-v2** | 768 | Alternative (shorter context) |
| **intfloat/e5-small-v2** | 384 | Speed > accuracy |
| **intfloat/e5-large-v2** | 1024 | Best quality |

## Environment Variables

```bash
# Remote Qdrant (omit for embedded mode)
export MCP_QDRANT_URL=http://qdrant:6333

# HTTP server
export MCP_TRANSPORT_MODE=streamable-http  # or stdio
export MCP_SERVER_PORT=8001
export MCP_HTTP_ENABLED=true
export MCP_HTTP_PORT=8000

# Knowledge graph (optional)
export MCP_FALKORDB_ENABLED=true
export MCP_FALKORDB_HOST=localhost
export MCP_FALKORDB_PORT=6379

# Security
export MCP_MEMORY_API_KEY="your-api-key"  # Optional API key auth

# Debug
export MCP_MEMORY_EXPOSE_DEBUG_TOOLS=false
```

## Development

- Storage backends implement `MemoryStorage` ABC
- All features require tests
- Version updates: `__init__.py` → `pyproject.toml` → `uv lock`
- Config uses pydantic-settings with thread-safe lazy loading

## Key Files

- `src/mcp_memory_service/config.py` - All configuration (pydantic-settings)
- `src/mcp_memory_service/mcp_server.py` - MCP tool definitions
- `src/mcp_memory_service/storage/factory.py` - Backend factory

## Documentation

- **Configuration reference**: `docs/configuration.md`
- **Deployment guide**: `docs/deployment.md`
