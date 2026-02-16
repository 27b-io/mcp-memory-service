# Faceted Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-facet search to the MCP Memory Service, allowing users to filter memories by tags, date ranges, and memory type in a single query with optional semantic ranking.

**Architecture:** Storage-layer filtering using Qdrant compound filters. Extend the existing `search()` MCP tool with `mode="faceted"` that accepts date range parameters. When semantic query is provided, use vector search + filter; otherwise, use scroll API with filter + recency sort.

**Tech Stack:** Python 3.13, FastAPI, Qdrant, FastMCP

---

## Task 1: Extend Storage Interface with Date Range Parameters

**Files:**
- Modify: `src/mcp_memory_service/storage/base.py:95-110`
- Test: `tests/storage/test_base_interface.py` (verify signature only)

**Step 1: Add date range parameters to retrieve() signature**

Update the abstract method signature in `MemoryStorage.retrieve()`:

```python
@abstractmethod
async def retrieve(
    self,
    query: str,
    n_results: int = 5,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    min_similarity: float | None = None,
    offset: int = 0,
    start_timestamp: float | None = None,  # NEW
    end_timestamp: float | None = None,    # NEW
) -> list[MemoryQueryResult]:
    """Retrieve memories with optional date range filtering.

    Args:
        start_timestamp: Unix timestamp - filter memories from this time
        end_timestamp: Unix timestamp - filter memories until this time
    """
    pass
```

**Step 2: Add date range parameters to count_semantic_search() signature**

Update the count method:

```python
@abstractmethod
async def count_semantic_search(
    self,
    query: str,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    min_similarity: float | None = None,
    start_timestamp: float | None = None,  # NEW
    end_timestamp: float | None = None,    # NEW
) -> int:
    """Count memories matching semantic search with optional date range."""
    pass
```

**Step 3: Commit interface changes**

```bash
git add src/mcp_memory_service/storage/base.py
git commit -m "feat(storage): add date range params to retrieve interface"
```

---

## Task 2: Implement Qdrant Date Range Filtering

**Files:**
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:retrieve()` (~line 200)
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:count_semantic_search()` (~line 400)
- Test: Manual verification (integration test in Task 5)

**Step 1: Add date range filter construction helper**

Add this helper method to `QdrantStorage` class (~line 150):

```python
def _build_date_range_filter(
    self,
    start_timestamp: float | None,
    end_timestamp: float | None,
) -> list[dict[str, Any]]:
    """Build Qdrant filter conditions for date range.

    Returns list of filter conditions to be added to 'must' clause.
    """
    conditions = []

    if start_timestamp is not None:
        conditions.append({
            "key": "created_at",
            "range": {"gte": start_timestamp}
        })

    if end_timestamp is not None:
        conditions.append({
            "key": "created_at",
            "range": {"lte": end_timestamp}
        })

    return conditions
```

**Step 2: Update retrieve() to accept and use date range parameters**

Modify the `retrieve()` method signature and implementation:

```python
async def retrieve(
    self,
    query: str,
    n_results: int = 5,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    min_similarity: float | None = None,
    offset: int = 0,
    start_timestamp: float | None = None,  # NEW
    end_timestamp: float | None = None,    # NEW
) -> list[MemoryQueryResult]:
    """Retrieve memories with optional date range filtering."""
    # Build filter conditions
    must_conditions = []

    # Existing tag/memory_type filter logic...
    if tags:
        # ... existing code ...
        pass

    if memory_type:
        # ... existing code ...
        pass

    # NEW: Add date range filters
    must_conditions.extend(
        self._build_date_range_filter(start_timestamp, end_timestamp)
    )

    # Build final filter
    filter_dict = None
    if must_conditions:
        filter_dict = {"must": must_conditions}

    # Execute search with filter...
    # ... rest of existing implementation
```

**Step 3: Update count_semantic_search() similarly**

Apply the same date range filter logic to the count method.

**Step 4: Commit Qdrant implementation**

```bash
git add src/mcp_memory_service/storage/qdrant_storage.py
git commit -m "feat(qdrant): implement date range filtering"
```

---

## Task 3: Add Faceted Search to MemoryService

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (add new method)
- Test: `tests/services/test_faceted_search.py` (create new)

**Step 1: Write failing test for faceted search**

Create `tests/services/test_faceted_search.py`:

```python
import pytest
from datetime import datetime, timezone
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory


@pytest.mark.asyncio
async def test_faceted_search_with_all_facets(memory_service, sample_memories):
    """Test faceted search with tags, date range, and memory type."""
    # Arrange: sample_memories fixture contains test data
    # Expected: memories matching all facets

    result = await memory_service.faceted_search(
        query="test query",
        tags=["python", "testing"],
        match_all=False,
        start_date="2024-01-01",
        end_date="2024-12-31",
        memory_type="note",
        page=1,
        page_size=10,
    )

    assert result["success"] is True
    assert len(result["memories"]) > 0
    assert result["total"] >= len(result["memories"])

    # Verify all results match facets
    for mem in result["memories"]:
        assert mem["memory_type"] == "note"
        assert any(tag in mem["tags"] for tag in ["python", "testing"])


@pytest.mark.asyncio
async def test_faceted_search_without_query_sorts_by_recency(memory_service):
    """Test faceted search without semantic query returns recent memories."""
    result = await memory_service.faceted_search(
        query="",  # No semantic query
        tags=["python"],
        page=1,
        page_size=5,
    )

    assert result["success"] is True

    # Verify results are sorted by created_at descending
    timestamps = [mem["created_at"] for mem in result["memories"]]
    assert timestamps == sorted(timestamps, reverse=True)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/services/test_faceted_search.py -v
```

Expected: `AttributeError: 'MemoryService' object has no attribute 'faceted_search'`

**Step 3: Implement faceted_search() method**

Add to `MemoryService` class (~line 965):

```python
async def faceted_search(
    self,
    query: str = "",
    tags: list[str] | None = None,
    match_all: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    memory_type: str | None = None,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float | None = 0.6,
    encoding_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Faceted search with multi-dimensional filtering.

    Combines tags, date ranges, and memory type filters with optional
    semantic ranking. All facets use AND logic.

    Args:
        query: Optional semantic query. If empty, sort by recency.
        tags: Optional tag filter (OR if match_all=False, AND if match_all=True)
        match_all: Tag matching logic (default: False = OR)
        start_date: Filter from date (ISO 8601: YYYY-MM-DD)
        end_date: Filter until date (ISO 8601: YYYY-MM-DD)
        memory_type: Filter by type (note/decision/task/reference)
        page: Page number (1-indexed)
        page_size: Results per page
        min_similarity: Similarity threshold for semantic search
        encoding_context: Context-dependent retrieval boost

    Returns:
        Dictionary with memories, pagination metadata, and facet info
    """
    try:
        # Parse date strings to timestamps
        start_timestamp = None
        end_timestamp = None

        if start_date:
            dt = datetime.fromisoformat(start_date)
            start_timestamp = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp()

        if end_date:
            dt = datetime.fromisoformat(end_date)
            end_timestamp = datetime(dt.year, dt.month, dt.day, 23, 59, 59, tzinfo=timezone.utc).timestamp()

        # Calculate offset
        offset = (page - 1) * page_size

        # Get total count for pagination
        if query.strip():
            # Semantic search with facets
            total = await self.storage.count_semantic_search(
                query=query,
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

            # Retrieve with all facets
            query_results = await self.storage.retrieve(
                query=query,
                n_results=page_size,
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                offset=offset,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        else:
            # Filter-only mode: no semantic ranking
            total = await self.storage.count_time_range(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                tags=tags,
                memory_type=memory_type,
            )

            query_results = await self.storage.recall(
                query=None,
                n_results=page_size,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                tags=tags,
                memory_type=memory_type,
                offset=offset,
            )

            # Sort by recency
            query_results.sort(key=lambda r: r.memory.created_at or 0.0, reverse=True)

        # Format results
        results = []
        result_hashes = []
        for result in query_results:
            memory_dict = self._format_memory_response(result.memory)
            if query.strip():
                memory_dict["similarity_score"] = result.similarity_score
            results.append(memory_dict)
            result_hashes.append(result.memory.content_hash)

        # Track access counts
        self._fire_access_count_updates(result_hashes)

        return {
            "success": True,
            "memories": results,
            "query": query,
            "facets": {
                "tags": tags,
                "match_all": match_all,
                "start_date": start_date,
                "end_date": end_date,
                "memory_type": memory_type,
            },
            **self._build_pagination_metadata(total, page, page_size),
        }

    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format: {e}. Use ISO 8601 (YYYY-MM-DD)",
            "memories": [],
        }
    except Exception as e:
        logger.error(f"Faceted search failed: {e}")
        return {
            "success": False,
            "error": f"Faceted search failed: {str(e)}",
            "memories": [],
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/services/test_faceted_search.py -v
```

Expected: PASS

**Step 5: Commit service layer implementation**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/services/test_faceted_search.py
git commit -m "feat(service): add faceted_search method with multi-facet filtering"
```

---

## Task 4: Add Faceted Mode to MCP Search Tool

**Files:**
- Modify: `src/mcp_memory_service/mcp_server.py:search()` (~line 260)
- Test: Manual MCP tool invocation

**Step 1: Add start_date/end_date parameters to search() signature**

Update the `@mcp.tool()` decorated function:

```python
@mcp.tool()
async def search(
    ctx: Context,
    query: str = "",
    mode: str = "hybrid",
    tags: str | list[str] | None = None,
    match_all: bool = False,
    start_date: str | None = None,  # NEW
    end_date: str | None = None,    # NEW
    k: int = 10,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float = 0.6,
    output: str = "full",
    memory_type: str | None = None,
    encoding_context: dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    """Search and retrieve memories. Consolidates all retrieval modes into one tool.

    Args:
        query: Natural language search query
        mode: Search strategy:
            - "hybrid": Semantic + tag-boosted (default)
            - "scan": Returns summaries (token-efficient)
            - "similar": Pure k-NN vector search
            - "tag": Exact tag matching
            - "recent": Chronological (newest first)
            - "faceted": Multi-facet filtering (tags + date + type) [NEW]
        tags: Tags to filter by
        match_all: For "tag"/"faceted" modes — True=AND, False=OR
        start_date: For "faceted" mode — filter from date (YYYY-MM-DD) [NEW]
        end_date: For "faceted" mode — filter until date (YYYY-MM-DD) [NEW]
        ...
    """
```

**Step 2: Add routing logic for mode="faceted"**

Add routing case after the existing mode handlers (~line 350):

```python
# ... existing mode handlers (hybrid, scan, similar, tag, recent) ...

elif mode == "faceted":
    # Faceted search: multi-dimensional filtering
    normalized_tags = _normalize_tags(tags)

    result = await memory_service.faceted_search(
        query=query.strip(),
        tags=normalized_tags or None,
        match_all=match_all,
        start_date=start_date,
        end_date=end_date,
        memory_type=memory_type,
        page=page,
        page_size=page_size,
        min_similarity=min_similarity,
        encoding_context=encoding_context,
    )

    if not result.get("success", False):
        return _inject_latency({"error": result.get("error", "Faceted search failed")}, _t0)

    # Format as TOON
    toon_output = format_search_results_as_toon(
        memories=result["memories"],
        page=result["page"],
        page_size=result["page_size"],
        total=result["total"],
        has_more=result["has_more"],
        total_pages=result["total_pages"],
    )

    return _inject_latency(toon_output, _t0)

else:
    return _inject_latency({"error": f"Unknown search mode: {mode}"}, _t0)
```

**Step 3: Update docstring with faceted mode examples**

Add examples to the tool docstring:

```python
    """
    ...

    Examples:
        # Faceted search with all filters
        search(
            mode="faceted",
            query="async patterns",
            tags=["python", "concurrency"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            memory_type="note"
        )

        # Faceted filter-only (no semantic query)
        search(
            mode="faceted",
            tags=["urgent"],
            start_date="2024-02-01",
            memory_type="task"
        )
    """
```

**Step 4: Commit MCP tool changes**

```bash
git add src/mcp_memory_service/mcp_server.py
git commit -m "feat(mcp): add faceted mode to search tool with date range support"
```

---

## Task 5: Integration Testing

**Files:**
- Create: `tests/integration/test_faceted_search_integration.py`

**Step 1: Write integration test**

```python
import pytest
from datetime import datetime, timezone
from mcp_memory_service.models.memory import Memory


@pytest.mark.asyncio
@pytest.mark.integration
async def test_faceted_search_end_to_end(test_storage, memory_service):
    """Test complete faceted search flow through all layers."""
    # Arrange: Store test memories with known facets
    memories = [
        Memory(
            content="Python async testing guide",
            content_hash="hash1",
            tags=["python", "testing", "async"],
            memory_type="note",
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp(),
        ),
        Memory(
            content="JavaScript testing tutorial",
            content_hash="hash2",
            tags=["javascript", "testing"],
            memory_type="note",
            created_at=datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp(),
        ),
        Memory(
            content="Task: Review Python code",
            content_hash="hash3",
            tags=["python", "review"],
            memory_type="task",
            created_at=datetime(2024, 8, 1, tzinfo=timezone.utc).timestamp(),
        ),
    ]

    for mem in memories:
        await test_storage.store(mem)

    # Act: Search with all facets
    result = await memory_service.faceted_search(
        query="testing",
        tags=["python", "testing"],
        match_all=False,  # OR logic
        start_date="2024-05-01",
        end_date="2024-07-31",
        memory_type="note",
        page=1,
        page_size=10,
    )

    # Assert: Only hash1 matches (python OR testing + date range + memory_type=note)
    assert result["success"] is True
    assert len(result["memories"]) == 1
    assert result["memories"][0]["content_hash"] == "hash1"
    assert result["facets"]["tags"] == ["python", "testing"]
    assert result["facets"]["memory_type"] == "note"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_faceted_search_without_query(test_storage, memory_service):
    """Test faceted filter-only mode returns results by recency."""
    # Arrange: memories created at different times
    # ... similar setup ...

    # Act: Search without semantic query
    result = await memory_service.faceted_search(
        query="",
        tags=["python"],
        start_date="2024-01-01",
        page=1,
        page_size=10,
    )

    # Assert: Results sorted by recency
    assert result["success"] is True
    timestamps = [m["created_at"] for m in result["memories"]]
    assert timestamps == sorted(timestamps, reverse=True)
```

**Step 2: Run integration tests**

```bash
uv run pytest tests/integration/test_faceted_search_integration.py -v -m integration
```

Expected: PASS

**Step 3: Commit integration tests**

```bash
git add tests/integration/test_faceted_search_integration.py
git commit -m "test: add faceted search integration tests"
```

---

## Task 6: Quality Gates and Finalization

**Files:**
- Run: Quality gate checks
- Update: `CHANGELOG.md` (if exists)

**Step 1: Run full test suite**

```bash
uv run pytest -x -m "not slow"
```

Expected: All tests pass

**Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: No issues

**Step 3: Manual MCP tool test**

Test the MCP tool via CLI or client:

```bash
# Example MCP call (adjust based on setup)
mcp-client call search --mode faceted --query "python" --tags "testing" --start-date "2024-01-01"
```

Expected: TOON-formatted results with pagination

**Step 4: Update memory with implementation notes**

Store a memory about this implementation:

```bash
# Via MCP tool or direct storage
echo "Implemented faceted search (mm-2jmnr): multi-facet filtering with tags, date ranges, memory type. Storage-layer filtering using Qdrant compound filters. Supports optional semantic queries." > /tmp/memo.txt
```

**Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete faceted search implementation (mm-2jmnr)

Adds mode='faceted' to search() MCP tool supporting multi-facet
filtering by tags, date ranges, and memory type with optional
semantic ranking.

- Storage layer: date range filtering in Qdrant
- Service layer: faceted_search() method
- MCP tool: faceted mode routing
- Tests: unit + integration coverage

Closes mm-2jmnr"
```

---

## Verification Checklist

- [ ] All tests pass (`pytest -x -m "not slow"`)
- [ ] Linters pass (ruff check + format)
- [ ] MCP tool accepts new parameters
- [ ] Faceted search returns TOON-formatted results
- [ ] Date range filtering works correctly
- [ ] Tag AND/OR logic behaves as expected
- [ ] Query-less mode sorts by recency
- [ ] Pagination metadata is correct
- [ ] No regression in existing search modes

---

## File Summary

**Modified:**
- `src/mcp_memory_service/storage/base.py` - Interface changes
- `src/mcp_memory_service/storage/qdrant_storage.py` - Date filtering
- `src/mcp_memory_service/services/memory_service.py` - faceted_search()
- `src/mcp_memory_service/mcp_server.py` - Faceted mode routing

**Created:**
- `tests/services/test_faceted_search.py` - Service layer tests
- `tests/integration/test_faceted_search_integration.py` - E2E tests
- `docs/plans/2026-02-17-faceted-search-design.md` - Design doc

**Not Modified:**
- `src/mcp_memory_service/web/api/search.py` - HTTP API (out of scope)
- `src/mcp_memory_service/formatters/toon.py` - Already supports format

---

## Notes

- Date filtering uses Unix timestamps internally (created_at field)
- ISO 8601 dates converted to timestamps in service layer
- Qdrant filter DSL: https://qdrant.tech/documentation/concepts/filtering/
- Existing `search_by_tag()` uses similar date filtering pattern (proven approach)
- No breaking changes to existing search modes
