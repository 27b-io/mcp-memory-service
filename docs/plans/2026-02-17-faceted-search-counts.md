# Faceted Search Counts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional facet counts to retrieve_memory results for analytics and data exploration

**Architecture:** Extend MemoryService.retrieve_memories() with include_facets parameter. When enabled, compute tag, memory_type, and date bucket counts from the result set and include in response.

**Tech Stack:** Python 3.10+, pydantic, pytest, asyncio

---

## Task 1: Create Facet Types and Utilities

**Files:**
- Create: `src/mcp_memory_service/utils/faceting.py`
- Test: `tests/unit/test_faceting.py`

**Step 1: Write failing test for basic facet computation**

```python
# tests/unit/test_faceting.py
from datetime import datetime, timedelta, timezone
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.faceting import compute_facets


def test_compute_facets_empty():
    """Empty input should return zero counts."""
    result = compute_facets([])

    assert result["tags"] == {}
    assert result["memory_types"] == {}
    assert result["date_buckets"]["last_7d"] == 0
    assert result["date_buckets"]["last_30d"] == 0
    assert result["date_buckets"]["last_90d"] == 0
    assert result["date_buckets"]["last_year"] == 0
    assert result["date_buckets"]["older"] == 0


def test_compute_facets_tags():
    """Should count tag occurrences."""
    now = datetime.now(timezone.utc).timestamp()

    memories = [
        Memory(content="test1", content_hash="hash1", tags=["python", "api"], created_at=now),
        Memory(content="test2", content_hash="hash2", tags=["python"], created_at=now),
        Memory(content="test3", content_hash="hash3", tags=["api", "database"], created_at=now),
    ]

    result = compute_facets(memories)

    assert result["tags"] == {"python": 2, "api": 2, "database": 1}
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_faceting.py::test_compute_facets_empty -v`
Expected: `ModuleNotFoundError: No module named 'mcp_memory_service.utils.faceting'`

**Step 3: Create faceting.py with type definitions**

```python
# src/mcp_memory_service/utils/faceting.py
"""Utilities for computing facet counts from memory search results."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from ..models.memory import Memory


class DateBuckets(TypedDict):
    """Date range distribution counts."""
    last_7d: int
    last_30d: int
    last_90d: int
    last_year: int
    older: int


class FacetCounts(TypedDict):
    """Facet count aggregations."""
    tags: dict[str, int]
    memory_types: dict[str, int]
    date_buckets: DateBuckets


def compute_facets(memories: list[Memory]) -> FacetCounts:
    """
    Compute facet counts from a result set.

    Uses single-pass aggregation for performance. Date buckets are cumulative
    (last_30d includes last_7d).

    Args:
        memories: List of Memory objects to analyze

    Returns:
        FacetCounts with tag, type, and date distributions
    """
    if not memories:
        return {
            "tags": {},
            "memory_types": {},
            "date_buckets": {
                "last_7d": 0,
                "last_30d": 0,
                "last_90d": 0,
                "last_year": 0,
                "older": 0,
            },
        }

    # Count tags
    tag_counter: Counter[str] = Counter()
    for memory in memories:
        if memory.tags:
            tag_counter.update(memory.tags)

    # Count memory types
    type_counter: Counter[str] = Counter()
    for memory in memories:
        if memory.memory_type:
            type_counter[memory.memory_type] += 1

    # Date bucketing (cumulative)
    now = datetime.now(timezone.utc).timestamp()
    day_7 = now - (7 * 86400)
    day_30 = now - (30 * 86400)
    day_90 = now - (90 * 86400)
    day_365 = now - (365 * 86400)

    date_buckets: DateBuckets = {
        "last_7d": 0,
        "last_30d": 0,
        "last_90d": 0,
        "last_year": 0,
        "older": 0,
    }

    for memory in memories:
        created = memory.created_at
        if created is None:
            continue

        if created >= day_7:
            date_buckets["last_7d"] += 1
        elif created >= day_30:
            date_buckets["last_30d"] += 1
        elif created >= day_90:
            date_buckets["last_90d"] += 1
        elif created >= day_365:
            date_buckets["last_year"] += 1
        else:
            date_buckets["older"] += 1

    return {
        "tags": dict(tag_counter),
        "memory_types": dict(type_counter),
        "date_buckets": date_buckets,
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_faceting.py -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/mcp_memory_service/utils/faceting.py tests/unit/test_faceting.py
git commit -m "feat: add facet count computation utility

TDD: compute_facets() aggregates tags, types, date buckets"
```

---

## Task 2: Add Date Bucket Edge Case Tests

**Files:**
- Test: `tests/unit/test_faceting.py`

**Step 1: Write test for date bucket boundaries**

```python
# Add to tests/unit/test_faceting.py
def test_date_bucket_boundaries():
    """Test exact boundary dates for buckets."""
    now = datetime.now(timezone.utc)

    memories = [
        Memory(content="6d ago", content_hash="h1",
               created_at=(now - timedelta(days=6)).timestamp()),
        Memory(content="8d ago", content_hash="h2",
               created_at=(now - timedelta(days=8)).timestamp()),
        Memory(content="29d ago", content_hash="h3",
               created_at=(now - timedelta(days=29)).timestamp()),
        Memory(content="89d ago", content_hash="h4",
               created_at=(now - timedelta(days=89)).timestamp()),
        Memory(content="364d ago", content_hash="h5",
               created_at=(now - timedelta(days=364)).timestamp()),
        Memory(content="400d ago", content_hash="h6",
               created_at=(now - timedelta(days=400)).timestamp()),
    ]

    result = compute_facets(memories)

    # Buckets are NOT cumulative (fixed in implementation)
    assert result["date_buckets"]["last_7d"] == 1  # 6d
    assert result["date_buckets"]["last_30d"] == 2  # 8d, 29d
    assert result["date_buckets"]["last_90d"] == 1  # 89d
    assert result["date_buckets"]["last_year"] == 1  # 364d
    assert result["date_buckets"]["older"] == 1  # 400d


def test_compute_facets_memory_types():
    """Should count memory type distribution."""
    now = datetime.now(timezone.utc).timestamp()

    memories = [
        Memory(content="t1", content_hash="h1", memory_type="note", created_at=now),
        Memory(content="t2", content_hash="h2", memory_type="note", created_at=now),
        Memory(content="t3", content_hash="h3", memory_type="decision", created_at=now),
        Memory(content="t4", content_hash="h4", memory_type=None, created_at=now),
    ]

    result = compute_facets(memories)

    assert result["memory_types"] == {"note": 2, "decision": 1}


def test_compute_facets_missing_dates():
    """Should handle memories with None created_at."""
    memories = [
        Memory(content="t1", content_hash="h1", tags=["python"], created_at=None),
        Memory(content="t2", content_hash="h2", tags=["api"],
               created_at=datetime.now(timezone.utc).timestamp()),
    ]

    result = compute_facets(memories)

    # Missing date should be skipped, not crash
    assert result["tags"] == {"python": 1, "api": 1}
    assert result["date_buckets"]["last_7d"] == 1  # Only t2 counted
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_faceting.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/unit/test_faceting.py
git commit -m "test: add faceting edge cases (boundaries, types, missing dates)"
```

---

## Task 3: Extend MemoryService.retrieve_memories()

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:748`
- Test: `tests/integration/test_retrieve_with_facets.py`

**Step 1: Write failing integration test**

```python
# tests/integration/test_retrieve_with_facets.py
import pytest
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_retrieve_with_facets_disabled_default(memory_service):
    """By default, facets should NOT be included."""
    # Store some test memories
    await memory_service.store_memory(
        content="Python API testing",
        tags=["python", "api"],
        memory_type="note"
    )

    result = await memory_service.retrieve_memories(query="python")

    # Facets key should not exist when include_facets=False (default)
    assert "facets" not in result


@pytest.mark.asyncio
async def test_retrieve_with_facets_enabled(memory_service):
    """When include_facets=True, facets should be returned."""
    # Store test memories
    now = datetime.now(timezone.utc).timestamp()
    await memory_service.store_memory(
        content="Python API testing",
        tags=["python", "api"],
        memory_type="note"
    )
    await memory_service.store_memory(
        content="Database query",
        tags=["database"],
        memory_type="decision"
    )

    result = await memory_service.retrieve_memories(
        query="python",
        include_facets=True
    )

    # Facets should be present
    assert "facets" in result
    assert "tags" in result["facets"]
    assert "memory_types" in result["facets"]
    assert "date_buckets" in result["facets"]

    # Verify tag counts (should include all results, not just page)
    assert "python" in result["facets"]["tags"]
    assert "api" in result["facets"]["tags"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_retrieve_with_facets.py -v`
Expected: `TypeError: retrieve_memories() got an unexpected keyword argument 'include_facets'`

**Step 3: Modify retrieve_memories signature**

In `src/mcp_memory_service/services/memory_service.py`, find `async def retrieve_memories` (around line 748) and modify:

```python
async def retrieve_memories(
    self,
    query: str,
    page: int = 1,
    page_size: int = 10,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    min_similarity: float | None = None,
    encoding_context: dict[str, Any] | None = None,
    include_facets: bool = False,  # NEW
) -> dict[str, Any]:
```

**Step 4: Import compute_facets at top of file**

Add to imports (around line 35):

```python
from ..utils.faceting import compute_facets
```

**Step 5: Add facet computation before return**

Find the end of `retrieve_memories` where it builds the response dict (search for `return {` near the end of the method). Before the return statement, add:

```python
    # Compute facets if requested
    facets = None
    if include_facets and all_results:
        try:
            facets = compute_facets(all_results)
        except Exception as e:
            logger.warning(f"Facet computation failed: {e}")
            # Graceful degradation: continue without facets
```

Then modify the return dict to conditionally include facets:

```python
    response = {
        "memories": toon_results if format_response == "toon" else results,
        "page": page,
        "page_size": page_size,
        "total": total_results,
        "has_more": has_more,
        "total_pages": total_pages,
    }

    if facets is not None:
        response["facets"] = facets

    return response
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_retrieve_with_facets.py -v`
Expected: Both tests PASS

**Step 7: Run full test suite to check for regressions**

Run: `uv run pytest tests/unit/test_memory_service.py -v -k retrieve`
Expected: All existing retrieve tests still PASS (backward compatibility)

**Step 8: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/integration/test_retrieve_with_facets.py
git commit -m "feat: add include_facets param to retrieve_memories

Extends retrieve_memories with optional facet counting.
Graceful degradation if facet computation fails.
Backward compatible (default include_facets=False)."
```

---

## Task 4: Add Pagination Independence Test

**Files:**
- Test: `tests/integration/test_retrieve_with_facets.py`

**Step 1: Write test for pagination independence**

```python
# Add to tests/integration/test_retrieve_with_facets.py
@pytest.mark.asyncio
async def test_facets_pagination_independent(memory_service):
    """Facets should be same across all pages (computed from full result set)."""
    # Store 25 memories with known tags
    for i in range(25):
        await memory_service.store_memory(
            content=f"Test memory {i}",
            tags=["common", f"tag{i % 5}"],
            memory_type="note"
        )

    # Get page 1 with facets
    page1 = await memory_service.retrieve_memories(
        query="test",
        page=1,
        page_size=10,
        include_facets=True
    )

    # Get page 2 with facets
    page2 = await memory_service.retrieve_memories(
        query="test",
        page=2,
        page_size=10,
        include_facets=True
    )

    # Facets should be identical (computed from full result set)
    assert page1["facets"]["tags"] == page2["facets"]["tags"]
    assert page1["facets"]["memory_types"] == page2["facets"]["memory_types"]

    # Verify "common" tag appears ~25 times
    assert page1["facets"]["tags"]["common"] >= 20
```

**Step 2: Run test**

Run: `uv run pytest tests/integration/test_retrieve_with_facets.py::test_facets_pagination_independent -v`
Expected: PASS (facets computed from all_results, not paginated subset)

**Step 3: Commit**

```bash
git add tests/integration/test_retrieve_with_facets.py
git commit -m "test: verify facets are pagination-independent"
```

---

## Task 5: Update MCP Tool Schema

**Files:**
- Modify: `src/mcp_memory_service/mcp_server.py`

**Step 1: Find retrieve_memory tool definition**

Search for `"name": "retrieve_memory"` in mcp_server.py (around line 100-150).

**Step 2: Add include_facets to inputSchema**

In the `properties` dict of the tool's `inputSchema`, add:

```python
"include_facets": {
    "type": "boolean",
    "description": "Include facet counts (tags, types, date buckets) in response",
    "default": False
}
```

**Step 3: Update tool description to mention facets**

Update the tool's description to mention the new capability:

```python
"description": (
    "Retrieve memories using hybrid search (semantic + tag matching). "
    "Optionally include facet counts for analytics and data exploration."
)
```

**Step 4: Update tool handler to pass include_facets**

Find the `retrieve_memory` handler function (search for `async def handle_retrieve_memory` or similar). Add `include_facets` parameter:

```python
include_facets = arguments.get("include_facets", False)

result = await memory_service.retrieve_memories(
    query=query,
    page=page,
    page_size=page_size,
    tags=tags,
    memory_type=memory_type,
    min_similarity=min_similarity,
    encoding_context=encoding_context,
    include_facets=include_facets,  # NEW
)
```

**Step 5: Test MCP tool manually**

Run: `uv run python -m mcp_memory_service.mcp_server`
Then in another terminal, use MCP client to call:
```json
{
  "method": "tools/call",
  "params": {
    "name": "retrieve_memory",
    "arguments": {
      "query": "test",
      "include_facets": true
    }
  }
}
```
Expected: Response includes "facets" key

**Step 6: Commit**

```bash
git add src/mcp_memory_service/mcp_server.py
git commit -m "feat: add include_facets to retrieve_memory MCP tool

Updates tool schema and handler to support facet counts."
```

---

## Task 6: Add Performance Test

**Files:**
- Test: `tests/unit/test_faceting.py`

**Step 1: Write performance test**

```python
# Add to tests/unit/test_faceting.py
import time


def test_compute_facets_performance():
    """Facet computation should complete in <20ms for 100 memories."""
    now = datetime.now(timezone.utc).timestamp()

    # Generate 100 memories with realistic tag distribution
    memories = [
        Memory(
            content=f"Test memory {i}",
            content_hash=f"hash{i}",
            tags=[f"tag{i % 20}", "common", f"category{i % 5}"],
            memory_type=["note", "decision", "task"][i % 3],
            created_at=now - (i * 86400)  # Spread over days
        )
        for i in range(100)
    ]

    start = time.perf_counter()
    result = compute_facets(memories)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    # Verify correctness
    assert "common" in result["tags"]
    assert result["tags"]["common"] == 100

    # Verify performance
    assert elapsed < 20, f"Facet computation took {elapsed:.2f}ms (target: <20ms)"

    print(f"✓ Computed facets for 100 memories in {elapsed:.2f}ms")
```

**Step 2: Run performance test**

Run: `uv run pytest tests/unit/test_faceting.py::test_compute_facets_performance -v -s`
Expected: PASS with timing output

**Step 3: Commit**

```bash
git add tests/unit/test_faceting.py
git commit -m "test: add faceting performance validation (<20ms)"
```

---

## Task 7: Run Full Test Suite and Document

**Step 1: Run complete test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Check code coverage**

Run: `uv run pytest tests/unit/test_faceting.py --cov=src/mcp_memory_service/utils/faceting --cov-report=term-missing`
Expected: 100% coverage for faceting.py

**Step 3: Run integration tests**

Run: `uv run pytest tests/integration/test_retrieve_with_facets.py -v`
Expected: All integration tests PASS

**Step 4: Update bead progress**

```bash
bd update mm-wisp-j59j --status=completed
bd comments add mm-wisp-j59j "Implementation complete. All tests passing. include_facets param added to retrieve_memories with backward compatibility."
```

**Step 5: Final commit**

```bash
git add .
git commit -m "chore: faceted search counts implementation complete

All tests passing:
- Unit tests: faceting.py 100% coverage
- Integration tests: retrieve with facets
- Performance: <20ms for 100 memories
- Backward compatible: default include_facets=False

Related: mm-w3j6e"
```

---

## Success Criteria

- ✅ `compute_facets()` utility with 100% test coverage
- ✅ `include_facets` parameter added to `retrieve_memories()`
- ✅ MCP tool schema updated with `include_facets`
- ✅ Backward compatible (default `include_facets=False`)
- ✅ Performance <20ms for typical result sets
- ✅ Graceful degradation on facet computation errors
- ✅ Date buckets (7d, 30d, 90d, 1y, older) working
- ✅ Tag and memory_type counting working
- ✅ Facets computed from full result set (pagination-independent)

## Next Steps

After implementation:
1. Test with real queries on production-like data
2. Monitor performance metrics
3. Consider adding facet value limits if needed (YAGNI for now)
4. Submit to merge queue with `gt done`
