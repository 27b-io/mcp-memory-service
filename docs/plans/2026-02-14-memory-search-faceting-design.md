# Memory Search Faceting Design

**Date**: 2026-02-14
**Status**: Approved
**Feature**: Faceted search on memory metadata (tags, type, date)

## Overview

Add a dedicated `faceted_search()` MCP tool that enables filtering memories by multiple metadata dimensions simultaneously. This complements existing search tools:
- `retrieve_memory`: Semantic search with tag boosting
- `search_by_tag`: Exact tag matching
- `list_memories`: Chronological browsing with single tag/type filter

The new tool provides **pure filtering** (no semantic ranking) with results ordered chronologically (newest first).

## Requirements

From bead mm-mc4z: "Faceted search on memory metadata. Filter by tags, type, date."

### Supported Facets
1. **Tags**: Filter by multiple tags with AND/OR logic
2. **Memory Type**: Filter by type (note, decision, task, reference)
3. **Date Range**: Filter by creation date with flexible formats

## Design

### Approach: Dedicated `faceted_search()` Tool

**Rationale**: Creating a new tool provides:
- Clear intent through naming
- No breaking changes to existing tools
- Optimized query path for filtering
- Easy extensibility for future facets

### API Signature

```python
@mcp.tool()
async def faceted_search(
    ctx: Context,
    tags: list[str] | None = None,
    tag_match_all: bool = False,  # False = OR (any tag), True = AND (all tags)
    memory_type: str | None = None,
    date_from: str | None = None,  # ISO8601 or relative ("7d", "30d", "1y")
    date_to: str | None = None,    # ISO8601 or relative
    page: int = 1,
    page_size: int = 10
) -> str:  # Returns TOON format
    """
    Filter memories by metadata facets.

    Results ordered chronologically (newest first), not by relevance.
    Use retrieve_memory() for semantic ranking.
    """
```

### Architecture

```
┌─────────────────────────────────────┐
│  MCP Tool Layer                     │
│  faceted_search() → TOON output     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Service Layer                      │
│  MemoryService.faceted_search()     │
│  - Parse filters (relative dates)   │
│  - Validate parameters              │
│  - Call storage layer               │
│  - Format pagination                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Storage Layer                      │
│  QdrantStorage.faceted_search()     │
│  - Build Qdrant filter objects      │
│  - Execute filtered query           │
│  - Map results to Memory objects    │
└─────────────────────────────────────┘
```

## Implementation Details

### 1. Date Parsing (Relative + Absolute)

Support both formats for user convenience:

**Relative Dates**:
- `"7d"` → 7 days ago from now
- `"30d"` → 30 days ago
- `"1y"` → 1 year ago

**Absolute Dates**:
- ISO8601: `"2026-01-15T00:00:00Z"`
- RFC3339 compatible

```python
def parse_date_filter(date_str: str) -> float:
    """Parse relative or absolute date to Unix timestamp."""
    if re.match(r'^\d+[dmy]$', date_str):
        # Relative: "7d", "30d", "1y"
        return parse_relative_date(date_str)
    else:
        # Absolute: ISO8601
        return datetime.fromisoformat(date_str).timestamp()
```

### 2. Qdrant Filter Construction

Use Qdrant's native filtering for efficiency:

**Tag Filtering**:
```python
if tags:
    if tag_match_all:
        # AND logic: all tags must match
        for tag in tags:
            must_conditions.append(
                FieldCondition(key="tags_str", match=MatchText(text=tag))
            )
    else:
        # OR logic: any tag matches
        should_conditions.append(
            Filter(
                should=[
                    FieldCondition(key="tags_str", match=MatchText(text=tag))
                    for tag in tags
                ]
            )
        )
```

**Type Filtering**:
```python
if memory_type:
    must_conditions.append(
        FieldCondition(key="type", match=MatchValue(value=memory_type))
    )
```

**Date Range Filtering**:
```python
if date_from or date_to:
    range_filter = {}
    if date_from:
        range_filter["gte"] = parse_date_filter(date_from)
    if date_to:
        range_filter["lte"] = parse_date_filter(date_to)

    must_conditions.append(
        FieldCondition(key="created_at", range=Range(**range_filter))
    )
```

### 3. Result Ordering

Results ordered by `created_at` descending (newest first):

```python
results = await self.qdrant_client.scroll(
    collection_name=self.collection_name,
    scroll_filter=Filter(must=must_conditions, should=should_conditions),
    order_by="created_at",  # Descending
    limit=page_size,
    offset=(page - 1) * page_size
)
```

### 4. TOON Output Format

Consistent with existing search tools:

```
# page=1 total=42 page_size=10 has_more=true total_pages=5
Memory content here|tag1,tag2|{"key":"val"}|2026-02-14T10:30:00Z|2026-02-14T10:30:00Z|abc123
Another memory|tag3|{}|2026-02-13T15:20:00Z|2026-02-13T15:20:00Z|def456
```

## Error Handling

| Error Condition | Response |
|----------------|----------|
| Invalid date format | `ValueError: Invalid date format. Use ISO8601 or relative (e.g., "7d")` |
| Invalid memory_type | `ValueError: memory_type must be one of: note, decision, task, reference` |
| page < 1 | `ValueError: page must be >= 1` |
| page_size > 100 | `ValueError: page_size must be <= 100` |
| Empty results | Return empty TOON with `total=0, has_more=false` |
| Storage error | Propagate with context: `StorageError: Failed to execute faceted search` |

## Testing Strategy

### Unit Tests
- Date parsing (relative + absolute formats)
- Filter construction (tags, type, dates)
- Edge cases (empty filters, invalid inputs)
- Pagination math

### Integration Tests
- Single facet filtering (tags only, type only, date only)
- Multi-facet combinations (tags + type + date)
- Tag logic (AND vs OR)
- Pagination (multiple pages, last page)
- Empty result sets

### Test Cases
```python
# Test: OR tag logic
faceted_search(tags=["python", "api"], tag_match_all=False)
# Returns memories with python OR api tag

# Test: AND tag logic
faceted_search(tags=["python", "api"], tag_match_all=True)
# Returns memories with python AND api tags

# Test: Relative date
faceted_search(date_from="7d")
# Returns memories from last 7 days

# Test: Multi-facet
faceted_search(
    tags=["project-alpha"],
    memory_type="decision",
    date_from="30d"
)
# Returns decision-type memories tagged project-alpha from last 30 days
```

## Performance Considerations

- **Qdrant native filtering**: No post-fetch filtering in Python
- **Index usage**: Qdrant indexes `tags_str`, `type`, `created_at` fields
- **Pagination**: Fetch only requested page, not all results
- **Expected latency**: < 50ms for typical queries (< 100K memories)

## Future Enhancements

Potential future facets:
- `salience_min/max`: Filter by salience score range
- `access_count_min`: Filter by access frequency
- `has_relations`: Filter memories with knowledge graph edges
- `metadata_filters`: Filter by custom metadata fields

## Migration & Rollout

- **No migration needed**: Uses existing Memory fields
- **Backward compatible**: New tool, no changes to existing APIs
- **Feature flag**: Not required (pure addition)
- **Documentation**: Add to MCP tools reference

## Acceptance Criteria

- [ ] `faceted_search()` MCP tool implemented
- [ ] Supports tags (AND/OR), memory_type, date range filters
- [ ] Relative date parsing works ("7d", "30d", "1y")
- [ ] Results ordered chronologically (newest first)
- [ ] TOON format output with pagination
- [ ] Unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] Quality gates pass (ruff, pytest)

---

**Approved by**: Mayor
**Implemented by**: Polecat Imperator
