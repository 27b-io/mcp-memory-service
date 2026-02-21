# Design: Faceted Search with Counts

**Date**: 2026-02-17
**Author**: polecat/nux
**Bead**: mm-w3j6e

## Overview

Extend `retrieve_memory` to optionally return facet counts alongside search results. This enables analytics, data exploration, and UI filtering without separate API calls.

## Distinction from mm-mc4z

Imperator's mm-mc4z implemented faceted *filtering* (separate `faceted_search()` method). This feature adds facet *counts* to existing search results - answering "of the memories matching my search, how are they distributed?"

## Requirements

From bead mm-w3j6e:
- Facet by: tags (with counts), memory_type, creation date buckets
- Support multi-select facets (AND/OR logic) - **deferred to filtering layer**
- Return facet counts with search results
- Efficient query performance (<100ms p95, <20ms facet overhead)
- Add to retrieve_memory tool as optional params

## Architecture

### Approach A: Extend retrieve_memory (Approved)

Add `include_facets: bool = False` parameter to `retrieve_memories()`. When enabled, compute facet counts from the result set and include in response.

```
retrieve_memory(query, include_facets=True)
    ├── Execute search (vector/hybrid/tag)
    ├── Get result set (memories)
    └── If include_facets:
        ├── Count tag occurrences in results
        ├── Count memory_type distribution
        └── Bucket by date ranges
```

**Rationale**: Single API call, lower latency, counts reflect actual search results (not global corpus).

## Components

### 1. Facet Computation Utility

**Location**: `src/mcp_memory_service/utils/faceting.py` (new file)

```python
from collections import Counter
from datetime import datetime, timedelta
from typing import TypedDict

class DateBuckets(TypedDict):
    last_7d: int
    last_30d: int
    last_90d: int
    last_year: int
    older: int

class FacetCounts(TypedDict):
    tags: dict[str, int]
    memory_types: dict[str, int]
    date_buckets: DateBuckets

def compute_facets(memories: list[Memory]) -> FacetCounts:
    """
    Compute facet counts from a result set.

    Single-pass algorithm using Counter for tags/types,
    date bucketing for temporal distribution.

    Args:
        memories: Search result memories

    Returns:
        Facet counts for tags, types, and date buckets
    """
```

**Date Buckets**:
- `last_7d`: Created in last 7 days
- `last_30d`: Created in last 30 days (inclusive of 7d)
- `last_90d`: Created in last 90 days (inclusive of 30d)
- `last_year`: Created in last 365 days (inclusive of 90d)
- `older`: Created before last year

**Performance**: O(n) where n = result set size (typically 10-100, not full corpus)

### 2. Modified MemoryService.retrieve_memories()

**Location**: `src/mcp_memory_service/services/memory_service.py`

**Changes**:
- Add `include_facets: bool = False` parameter
- After pagination, if `include_facets=True`: call `compute_facets(all_results)`
- Add `facets` key to response dict

**Backward Compatibility**: Default `include_facets=False` preserves existing behavior.

### 3. MCP Tool Schema Update

**Location**: `src/mcp_memory_service/mcp_server.py`

Update `retrieve_memory` tool schema to document `include_facets` parameter.

## Response Format

```json
{
  "memories": [...],
  "page": 1,
  "page_size": 10,
  "total": 58,
  "has_more": true,
  "facets": {
    "tags": {
      "python": 15,
      "api": 8,
      "database": 12
    },
    "memory_types": {
      "note": 42,
      "decision": 10,
      "task": 6
    },
    "date_buckets": {
      "last_7d": 12,
      "last_30d": 35,
      "last_90d": 48,
      "last_year": 56,
      "older": 2
    }
  }
}
```

**Note**: `facets` key only present when `include_facets=True`.

## Data Flow

```
User: retrieve_memory(query="python api", include_facets=true)
    │
    ▼
MemoryService.retrieve_memories()
    ├── Execute search (hybrid/vector)
    ├── Get 58 total results
    ├── Apply pagination → 10 memories for page 1
    │
    ├── [NEW] If include_facets:
    │   ├── compute_facets(all_58_results)  ← Full result set
    │   └── Add facets to response
    │
    └── Return {memories: [...10...], total: 58, facets: {...}}
```

**Key Decision**: Compute facets from **full result set** (before pagination), not just the current page. This gives accurate counts across all search results.

## Error Handling

- **Facet computation fails**: Log warning, return response without facets (graceful degradation)
- **Invalid memory dates**: Skip from date bucketing, log warning
- **Empty result set**: Return empty facet counts (all zeros)

## Testing Strategy

### Unit Tests

**Location**: `tests/unit/test_faceting.py`

| Test | Description |
|------|-------------|
| `test_compute_facets_tags` | Verify tag counting with duplicates |
| `test_compute_facets_memory_types` | Verify type distribution |
| `test_compute_facets_date_buckets` | Verify date bucketing logic |
| `test_compute_facets_empty` | Empty input → empty counts |
| `test_date_bucket_boundaries` | Edge cases: exactly 7d, 30d, etc. |

### Integration Tests

**Location**: `tests/integration/test_retrieve_with_facets.py`

| Test | Description |
|------|-------------|
| `test_retrieve_with_facets_enabled` | Facets returned when flag=true |
| `test_retrieve_facets_disabled_default` | No facets when flag=false/omitted |
| `test_facets_match_result_set` | Facet counts sum to total results |
| `test_facets_pagination_independent` | Same facets across all pages |

## Performance

- **Target**: <20ms overhead for facet computation
- **Method**: Single pass over result set, Counter aggregation
- **Complexity**: O(n * m) where n=results, m=avg tags per memory
- **Expected**: ~5-10ms for typical result sets (10-100 memories)

## Assumptions (Documented per Witness directive)

1. **Facets computed from result set**, not full corpus - reflects distribution of search matches
2. **Date buckets are cumulative** - "last_30d" includes "last_7d"
3. **Multi-select AND/OR logic deferred** - bead requirement noted but filtering already exists in mm-mc4z's `faceted_search()`, not duplicating
4. **Tags counted with multiplicity** - if one memory has tag "python" twice, counts as 2 (or 1 depending on tag storage - will verify in implementation)

## Out of Scope

- Multi-select facet filtering with AND/OR (already exists in `faceted_search()` from mm-mc4z)
- Dynamic date bucket ranges (fixed buckets: 7d, 30d, 90d, 1y)
- Hierarchical facets (flat tags only)
- Facet value limits/truncation (return all counts)

## Implementation Order

1. Create `utils/faceting.py` with `compute_facets()` and unit tests
2. Extend `MemoryService.retrieve_memories()` with `include_facets` param
3. Update MCP tool schema documentation
4. Add integration tests
5. Test performance with large result sets

## Success Metrics

- Facet computation overhead: <20ms p95
- Unit test coverage: 100% for faceting.py
- Integration test: facets match manual count
- Backward compatibility: existing calls work unchanged
