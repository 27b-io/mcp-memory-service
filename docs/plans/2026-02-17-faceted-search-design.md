# Faceted Search Design

**Bead:** mm-2jmnr
**Date:** 2026-02-17
**Status:** Approved
**Epic:** mm-wisp-xlvsn (mol-polecat-work)

## Overview

Add multi-facet search capability to the MCP Memory Service, allowing users to filter memories by multiple dimensions simultaneously: tags, date ranges, memory type, and optional semantic queries.

## Problem Statement

Current search modes are isolated:
- Semantic search (`hybrid` mode) filters by tags and memory_type but not date ranges
- Tag search (`tag` mode) filters by date ranges but lacks semantic ranking
- No unified API for combining multiple facets in a single query

Users need to chain multiple API calls or accept incomplete filtering. This design addresses that gap.

## Requirements

1. **Unified API**: Add `mode="faceted"` to the existing `search()` MCP tool
2. **Multi-facet filtering**: Support tags, date ranges (start_date, end_date), and memory_type
3. **AND logic between facets**: All specified facets must match
4. **AND/OR within tags**: Support `match_all` parameter for tag filtering
5. **Optional semantic query**: If provided, rank by relevance; otherwise, sort by recency
6. **Pagination**: Standard page/page_size with metadata
7. **TOON output format**: Consistent with other search modes

## Design

### API Signature

Extend the existing `search()` tool with new parameters for `mode="faceted"`:

```python
@mcp.tool()
async def search(
    ctx: Context,
    query: str = "",
    mode: str = "hybrid",
    tags: str | list[str] | None = None,
    match_all: bool = False,
    start_date: str | None = None,  # NEW: ISO 8601 (YYYY-MM-DD)
    end_date: str | None = None,    # NEW: ISO 8601 (YYYY-MM-DD)
    memory_type: str | None = None,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float = 0.6,
    # ... existing params
) -> str | dict[str, Any]:
    """Search and retrieve memories with multi-facet filtering."""
```

**New parameters:**
- `start_date`: Filter memories from this date (ISO 8601: YYYY-MM-DD)
- `end_date`: Filter memories until this date (ISO 8601: YYYY-MM-DD)

**When `mode="faceted"`:**
- `query`: Optional semantic query. If omitted, sort by recency.
- `tags`: Optional tag filter with AND/OR logic via `match_all`
- `start_date` / `end_date`: Optional date range filter
- `memory_type`: Optional memory type filter
- All filters combine with AND logic

### Architecture

**Storage Layer (Qdrant):**
- Extend `MemoryStorage.retrieve()` to accept date range parameters
- Build Qdrant filter combining all facets:
  ```python
  {
      "must": [
          {"key": "created_at", "range": {"gte": start_ts, "lte": end_ts}},
          {"key": "memory_type", "match": {"value": memory_type}},
          {"key": "tags", "match": {"any": tags}} if not match_all else {"key": "tags", "match": {"all": tags}}
      ]
  }
  ```

**Service Layer (MemoryService):**
- Add `faceted_search()` method to coordinate facet filtering
- If semantic query provided: `storage.retrieve()` with vector search + filter
- If no semantic query: `storage.scroll()` with filter only, sort by `created_at` descending

**MCP Tool (mcp_server.py):**
- Route `mode="faceted"` to `memory_service.faceted_search()`
- Format results as TOON with pagination metadata

### Data Flow

```
User calls search(mode="faceted", tags=["python"], start_date="2024-01-01", query="async")
    ↓
mcp_server.search() validates params
    ↓
memory_service.faceted_search() builds filter spec
    ↓
storage.retrieve(query="async", tags=["python"], start_timestamp=..., ...)
    ↓
Qdrant performs vector search + compound filter
    ↓
Results formatted as TOON and returned
```

### File Changes

**Files to Modify:**
- `src/mcp_memory_service/storage/base.py` - Add date range params to `retrieve()` signature
- `src/mcp_memory_service/storage/qdrant_storage.py` - Implement Qdrant filter logic
- `src/mcp_memory_service/services/memory_service.py` - Add `faceted_search()` method
- `src/mcp_memory_service/mcp_server.py` - Add `start_date`/`end_date` params, route `mode="faceted"`

**Files NOT to touch:**
- `src/mcp_memory_service/web/api/search.py` - HTTP API (out of scope)
- `src/mcp_memory_service/graph/*` - Graph layer (unchanged)
- `src/mcp_memory_service/formatters/toon.py` - Already supports required format

### Error Handling

- **Invalid date format**: Return clear error message with expected format
- **Invalid date range**: `start_date > end_date` returns empty results (graceful degradation)
- **No facets specified**: Fallback to standard hybrid search behavior
- **Storage backend doesn't support filtering**: Gracefully degrade to in-memory filtering

### Testing Strategy

**Unit Tests:**
- Date parsing and validation
- Filter construction for various facet combinations
- Pagination with filtered results

**Integration Tests:**
- Faceted search with all facets specified
- Faceted search with semantic query + facets
- Faceted search without semantic query (recency sorting)
- Tag AND logic vs OR logic
- Empty result sets

**Manual Testing:**
- Performance comparison: faceted vs chained API calls
- Large corpus behavior (10k+ memories)

## Trade-offs

**Storage-layer filtering (chosen approach):**
- ✅ Most efficient - database does filtering
- ✅ Scales with corpus size
- ✅ Consistent with existing `search_by_tag()` pattern
- ⚠️ Requires interface changes to `storage.base.py`
- ⚠️ Qdrant filter DSL complexity

**Rejected alternatives:**
- Service-layer filtering: Inefficient, high memory usage
- Hybrid filtering: Split responsibility, harder to debug

## Success Criteria

1. Users can filter memories by tags + date range + memory_type in a single call
2. Faceted search performs within 200ms for 10k memory corpus
3. Results correctly combine all facets with AND logic
4. No regression in existing search modes
5. TOON output format matches other search modes

## Open Questions

None - design is complete and ready for implementation.

## References

- Bead: mm-2jmnr
- Epic: mm-wisp-xlvsn
- Related: `search_by_tag()` already uses date filtering (proven pattern)
- Qdrant filtering docs: https://qdrant.tech/documentation/concepts/filtering/
