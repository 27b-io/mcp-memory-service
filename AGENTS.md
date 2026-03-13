# Agent Instructions

MCP Memory Service — semantic memory server with Qdrant vector search and FalkorDB knowledge graph.

## Project

- **Repo**: 27b-io/mcp-memory-service (fork — PRs go here, not upstream doobidoo/)
- **Stack**: Python 3.12, FastMCP 2.0, Qdrant, FalkorDB, pydantic-settings
- **Entry**: `uv run memory server` (stdio) or Docker with `MCP_TRANSPORT_MODE=http`

## Architecture

```
src/mcp_memory_service/
├── mcp_server.py         # 9 MCP tools (FastMCP 2.0)
├── unified_server.py     # HTTP + MCP dual-mode server
├── services/memory_service.py  # Core business logic
├── storage/              # Qdrant backend (base.py ABC + factory)
├── graph/                # FalkorDB knowledge graph (optional)
├── models/memory.py      # Pydantic v2 Memory model
├── formatters/toon.py    # TOON pipe-delimited output
├── utils/                # Cognitive features (emotion, salience, interference, etc.)
├── config.py             # 23 pydantic-settings classes
└── cli/main.py           # CLI
```

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration — pydantic-settings, env vars |
| `mcp_server.py` | MCP tool definitions (store, search, delete, health, relation, supersede, contradictions, find_duplicates, merge_duplicates) |
| `services/memory_service.py` | Business logic — search pipeline, cognitive features, CRUD |
| `storage/qdrant_storage.py` | Vector storage implementation |
| `graph/client.py` | FalkorDB graph operations |

## Code Style

- **Linter**: Ruff, 129 char line length
- **Types**: basedpyright
- **Config**: pydantic-settings with `SecretStr` for secrets
- **Imports**: Absolute only
- **Tests**: pytest, `uv run pytest -x -m "not slow"` for fast feedback

## Quality Gates

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest -x -m "not slow"
```

All three must pass before pushing.

## Conventions

- Guard clauses over nesting
- No manual dependency edits — use `uv add`
- Semantic commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- One feature per PR
