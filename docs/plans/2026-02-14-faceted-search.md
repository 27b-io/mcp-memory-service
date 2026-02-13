# Faceted Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `faceted_search()` MCP tool to filter memories by tags (AND/OR), memory_type, and date range.

**Architecture:** Three-layer implementation (MCP tool → MemoryService → QdrantStorage) with Qdrant native filtering for performance. Results ordered chronologically, output in TOON format.

**Tech Stack:** Python 3.11+, Qdrant (vector DB), FastMCP (MCP framework), pytest

---

## Task 1: Date Parsing Utilities

**Files:**
- Create: `src/mcp_memory_service/utils/date_parsing.py`
- Test: `tests/unit/test_date_parsing.py`

**Step 1: Write failing test for relative date parsing**

```python
# tests/unit/test_date_parsing.py
import time
import pytest
from mcp_memory_service.utils.date_parsing import parse_date_filter


def test_parse_relative_days():
    """Test parsing relative days format."""
    now = time.time()
    result = parse_date_filter("7d")

    # Should be approximately 7 days ago
    expected = now - (7 * 24 * 60 * 60)
    assert abs(result - expected) < 2  # Within 2 seconds


def test_parse_relative_months():
    """Test parsing relative months format."""
    now = time.time()
    result = parse_date_filter("1m")

    # Should be approximately 30 days ago
    expected = now - (30 * 24 * 60 * 60)
    assert abs(result - expected) < 2


def test_parse_relative_years():
    """Test parsing relative years format."""
    now = time.time()
    result = parse_date_filter("1y")

    # Should be approximately 365 days ago
    expected = now - (365 * 24 * 60 * 60)
    assert abs(result - expected) < 2


def test_parse_iso8601_format():
    """Test parsing ISO8601 absolute dates."""
    result = parse_date_filter("2026-01-15T10:30:00Z")

    # Should parse to specific timestamp
    from datetime import datetime, timezone
    expected = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
    assert result == expected


def test_parse_invalid_format_raises_error():
    """Test that invalid formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_date_filter("invalid")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_date_parsing.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mcp_memory_service.utils.date_parsing'"

**Step 3: Write minimal implementation**

```python
# src/mcp_memory_service/utils/date_parsing.py
"""Date parsing utilities for faceted search."""

import re
import time
from datetime import datetime, timezone


def parse_date_filter(date_str: str) -> float:
    """
    Parse relative or absolute date to Unix timestamp.

    Supported formats:
    - Relative: "7d" (days), "1m" (months ~30d), "1y" (years ~365d)
    - Absolute: ISO8601 "2026-01-15T10:30:00Z"

    Args:
        date_str: Date string to parse

    Returns:
        Unix timestamp (float)

    Raises:
        ValueError: If format is invalid
    """
    # Try relative format first
    relative_match = re.match(r'^(\d+)([dmy])$', date_str.lower())
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)

        now = time.time()
        if unit == 'd':
            return now - (amount * 24 * 60 * 60)
        elif unit == 'm':
            return now - (amount * 30 * 24 * 60 * 60)  # Approximate month
        elif unit == 'y':
            return now - (amount * 365 * 24 * 60 * 60)  # Approximate year

    # Try ISO8601 format
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except (ValueError, AttributeError):
        pass

    raise ValueError(
        f"Invalid date format: {date_str}. "
        "Use relative (e.g., '7d', '1m', '1y') or ISO8601 (e.g., '2026-01-15T10:30:00Z')"
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_date_parsing.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/utils/date_parsing.py tests/unit/test_date_parsing.py
git commit -m "feat: add date parsing utilities for faceted search

- Support relative dates (7d, 1m, 1y)
- Support ISO8601 absolute dates
- Raise ValueError on invalid formats"
```

---

## Task 2: Storage Layer - Faceted Search

**Files:**
- Modify: `src/mcp_memory_service/storage/base.py` (add abstract method)
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:877` (implement method)
- Test: `tests/unit/test_qdrant_faceted_search.py`

**Step 1: Write failing test for storage layer**

```python
# tests/unit/test_qdrant_faceted_search.py
import pytest
from mcp_memory_service.storage.qdrant_storage import QdrantStorage
from mcp_memory_service.models.memory import Memory


@pytest.mark.asyncio
async def test_faceted_search_by_tags_or_logic(qdrant_storage):
    """Test faceted search with OR tag logic."""
    # Setup: Create test memories
    await qdrant_storage.store_memory(
        Memory(
            content="Python API development",
            content_hash="hash1",
            tags=["python", "api"],
            memory_type="note"
        )
    )
    await qdrant_storage.store_memory(
        Memory(
            content="JavaScript testing",
            content_hash="hash2",
            tags=["javascript", "testing"],
            memory_type="note"
        )
    )

    # Test: Search with OR logic (any tag matches)
    results = await qdrant_storage.faceted_search(
        tags=["python", "testing"],
        tag_match_all=False,
        page=1,
        page_size=10
    )

    # Should return both memories (python OR testing)
    assert results["total"] == 2
    assert len(results["memories"]) == 2


@pytest.mark.asyncio
async def test_faceted_search_by_tags_and_logic(qdrant_storage):
    """Test faceted search with AND tag logic."""
    # Setup: Create test memories
    await qdrant_storage.store_memory(
        Memory(
            content="Python API development",
            content_hash="hash1",
            tags=["python", "api"],
            memory_type="note"
        )
    )
    await qdrant_storage.store_memory(
        Memory(
            content="Python testing",
            content_hash="hash2",
            tags=["python", "testing"],
            memory_type="note"
        )
    )

    # Test: Search with AND logic (all tags must match)
    results = await qdrant_storage.faceted_search(
        tags=["python", "api"],
        tag_match_all=True,
        page=1,
        page_size=10
    )

    # Should return only first memory (python AND api)
    assert results["total"] == 1
    assert results["memories"][0].content_hash == "hash1"


@pytest.mark.asyncio
async def test_faceted_search_by_memory_type(qdrant_storage):
    """Test faceted search by memory type."""
    # Setup: Create test memories
    await qdrant_storage.store_memory(
        Memory(content="A note", content_hash="hash1", memory_type="note")
    )
    await qdrant_storage.store_memory(
        Memory(content="A decision", content_hash="hash2", memory_type="decision")
    )

    # Test: Filter by type
    results = await qdrant_storage.faceted_search(
        memory_type="decision",
        page=1,
        page_size=10
    )

    assert results["total"] == 1
    assert results["memories"][0].memory_type == "decision"


@pytest.mark.asyncio
async def test_faceted_search_by_date_range(qdrant_storage):
    """Test faceted search by date range."""
    import time

    # Setup: Create memories with different timestamps
    old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
    recent_time = time.time() - (5 * 24 * 60 * 60)  # 5 days ago

    await qdrant_storage.store_memory(
        Memory(
            content="Old memory",
            content_hash="hash1",
            created_at=old_time
        )
    )
    await qdrant_storage.store_memory(
        Memory(
            content="Recent memory",
            content_hash="hash2",
            created_at=recent_time
        )
    )

    # Test: Filter by date range (last 7 days)
    cutoff = time.time() - (7 * 24 * 60 * 60)
    results = await qdrant_storage.faceted_search(
        date_from=cutoff,
        page=1,
        page_size=10
    )

    # Should return only recent memory
    assert results["total"] == 1
    assert results["memories"][0].content_hash == "hash2"


@pytest.mark.asyncio
async def test_faceted_search_combined_filters(qdrant_storage):
    """Test combining multiple filters."""
    import time

    recent_time = time.time() - (5 * 24 * 60 * 60)

    # Setup: Create test memories
    await qdrant_storage.store_memory(
        Memory(
            content="Recent Python decision",
            content_hash="hash1",
            tags=["python"],
            memory_type="decision",
            created_at=recent_time
        )
    )
    await qdrant_storage.store_memory(
        Memory(
            content="Old Python note",
            content_hash="hash2",
            tags=["python"],
            memory_type="note",
            created_at=time.time() - (30 * 24 * 60 * 60)
        )
    )

    # Test: Combine tag + type + date filters
    cutoff = time.time() - (7 * 24 * 60 * 60)
    results = await qdrant_storage.faceted_search(
        tags=["python"],
        memory_type="decision",
        date_from=cutoff,
        page=1,
        page_size=10
    )

    # Should return only hash1 (matches all criteria)
    assert results["total"] == 1
    assert results["memories"][0].content_hash == "hash1"


@pytest.mark.asyncio
async def test_faceted_search_pagination(qdrant_storage):
    """Test pagination in faceted search."""
    # Setup: Create 15 memories
    for i in range(15):
        await qdrant_storage.store_memory(
            Memory(
                content=f"Memory {i}",
                content_hash=f"hash{i}",
                tags=["test"]
            )
        )

    # Test: First page
    page1 = await qdrant_storage.faceted_search(
        tags=["test"],
        page=1,
        page_size=10
    )
    assert len(page1["memories"]) == 10
    assert page1["total"] == 15
    assert page1["has_more"] is True

    # Test: Second page
    page2 = await qdrant_storage.faceted_search(
        tags=["test"],
        page=2,
        page_size=10
    )
    assert len(page2["memories"]) == 5
    assert page2["has_more"] is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_qdrant_faceted_search.py -v`
Expected: FAIL with "AttributeError: 'QdrantStorage' object has no attribute 'faceted_search'"

**Step 3: Add abstract method to base class**

```python
# src/mcp_memory_service/storage/base.py (add after search_by_tag method)

@abstractmethod
async def faceted_search(
    self,
    tags: list[str] | None = None,
    tag_match_all: bool = False,
    memory_type: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    """
    Filter memories by multiple metadata facets.

    Args:
        tags: Filter by tags (optional)
        tag_match_all: If True, require all tags; if False, require any tag
        memory_type: Filter by memory type (optional)
        date_from: Filter by created_at >= this timestamp (optional)
        date_to: Filter by created_at <= this timestamp (optional)
        page: Page number (1-indexed)
        page_size: Results per page

    Returns:
        Dict with:
        - memories: List[Memory]
        - total: int (total matching count)
        - page: int
        - page_size: int
        - has_more: bool
        - total_pages: int
    """
    pass
```

**Step 4: Implement in QdrantStorage**

```python
# src/mcp_memory_service/storage/qdrant_storage.py (add after search_by_tag method ~line 950)

async def faceted_search(
    self,
    tags: list[str] | None = None,
    tag_match_all: bool = False,
    memory_type: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    """Filter memories by multiple metadata facets."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, Range

    must_conditions = []
    should_conditions = []

    # Tag filtering
    if tags:
        if tag_match_all:
            # AND logic: all tags must match
            for tag in tags:
                must_conditions.append(
                    FieldCondition(key="tags_str", match=MatchText(text=tag))
                )
        else:
            # OR logic: any tag matches
            for tag in tags:
                should_conditions.append(
                    FieldCondition(key="tags_str", match=MatchText(text=tag))
                )

    # Memory type filtering
    if memory_type:
        must_conditions.append(
            FieldCondition(key="type", match=MatchValue(value=memory_type))
        )

    # Date range filtering
    if date_from or date_to:
        range_params = {}
        if date_from:
            range_params["gte"] = date_from
        if date_to:
            range_params["lte"] = date_to

        must_conditions.append(
            FieldCondition(key="created_at", range=Range(**range_params))
        )

    # Build filter
    filter_params = {}
    if must_conditions:
        filter_params["must"] = must_conditions
    if should_conditions:
        filter_params["should"] = should_conditions

    query_filter = Filter(**filter_params) if filter_params else None

    # Calculate offset
    offset = (page - 1) * page_size

    # Execute scroll query with filter
    scroll_result = await self.qdrant_client.scroll(
        collection_name=self.collection_name,
        scroll_filter=query_filter,
        limit=page_size,
        offset=offset,
        with_payload=True,
        with_vectors=False,
        order_by="created_at",  # Chronological order (newest first)
    )

    points, next_offset = scroll_result

    # Get total count
    count_result = await self.qdrant_client.count(
        collection_name=self.collection_name,
        count_filter=query_filter,
        exact=True,
    )
    total_count = count_result.count

    # Convert points to Memory objects
    memories = []
    for point in points:
        memory = Memory.from_dict(point.payload, embedding=None)
        memories.append(memory)

    # Calculate pagination
    total_pages = (total_count + page_size - 1) // page_size
    has_more = page < total_pages

    return {
        "memories": memories,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "has_more": has_more,
        "total_pages": total_pages,
    }
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_qdrant_faceted_search.py -v`
Expected: PASS (all 6 tests)

**Step 6: Commit**

```bash
git add src/mcp_memory_service/storage/base.py src/mcp_memory_service/storage/qdrant_storage.py tests/unit/test_qdrant_faceted_search.py
git commit -m "feat(storage): add faceted_search to storage layer

- Add abstract method to MemoryStorage base class
- Implement in QdrantStorage using native filters
- Support tags (AND/OR), memory_type, date range
- Chronological ordering (newest first)
- Full pagination support"
```

---

## Task 3: Service Layer - Faceted Search

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:527` (add method after list_memories)
- Test: `tests/unit/test_memory_service_faceted.py`

**Step 1: Write failing test for service layer**

```python
# tests/unit/test_memory_service_faceted.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp_memory_service.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_faceted_search_parses_relative_dates():
    """Test that service layer parses relative dates."""
    # Setup mock storage
    storage = AsyncMock()
    storage.faceted_search.return_value = {
        "memories": [],
        "total": 0,
        "page": 1,
        "page_size": 10,
        "has_more": False,
        "total_pages": 0,
    }

    # Create service
    settings = MagicMock()
    service = MemoryService(storage=storage, settings=settings)

    # Test: Call with relative date
    await service.faceted_search(date_from="7d", page=1, page_size=10)

    # Verify: Storage called with parsed timestamp (not string)
    call_args = storage.faceted_search.call_args
    date_from_arg = call_args.kwargs.get("date_from")

    assert isinstance(date_from_arg, float)
    # Should be approximately 7 days ago
    import time
    expected = time.time() - (7 * 24 * 60 * 60)
    assert abs(date_from_arg - expected) < 10  # Within 10 seconds


@pytest.mark.asyncio
async def test_faceted_search_validates_memory_type():
    """Test that service validates memory_type."""
    storage = AsyncMock()
    settings = MagicMock()
    service = MemoryService(storage=storage, settings=settings)

    # Test: Invalid memory type
    with pytest.raises(ValueError, match="memory_type must be one of"):
        await service.faceted_search(memory_type="invalid", page=1, page_size=10)


@pytest.mark.asyncio
async def test_faceted_search_validates_pagination():
    """Test that service validates pagination params."""
    storage = AsyncMock()
    settings = MagicMock()
    service = MemoryService(storage=storage, settings=settings)

    # Test: Invalid page
    with pytest.raises(ValueError, match="page must be >= 1"):
        await service.faceted_search(page=0, page_size=10)

    # Test: Invalid page_size
    with pytest.raises(ValueError, match="page_size must be <= 100"):
        await service.faceted_search(page=1, page_size=200)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_memory_service_faceted.py -v`
Expected: FAIL with "AttributeError: 'MemoryService' object has no attribute 'faceted_search'"

**Step 3: Implement service method**

```python
# src/mcp_memory_service/services/memory_service.py (add after list_memories method ~line 580)

async def faceted_search(
    self,
    tags: list[str] | None = None,
    tag_match_all: bool = False,
    memory_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    """
    Filter memories by multiple metadata facets.

    Args:
        tags: Filter by tags (optional)
        tag_match_all: If True, require all tags; if False, require any tag
        memory_type: Filter by memory type (optional)
        date_from: Filter by created_at >= this (ISO8601 or relative like "7d")
        date_to: Filter by created_at <= this (ISO8601 or relative)
        page: Page number (1-indexed)
        page_size: Results per page (max 100)

    Returns:
        Dict with memories, pagination metadata

    Raises:
        ValueError: If parameters are invalid
    """
    from ..utils.date_parsing import parse_date_filter

    # Validate memory_type
    valid_types = ["note", "decision", "task", "reference"]
    if memory_type and memory_type not in valid_types:
        raise ValueError(f"memory_type must be one of: {', '.join(valid_types)}")

    # Validate pagination
    if page < 1:
        raise ValueError("page must be >= 1")
    if page_size > 100:
        raise ValueError("page_size must be <= 100")

    # Parse date filters
    date_from_ts = None
    date_to_ts = None

    if date_from:
        date_from_ts = parse_date_filter(date_from)
    if date_to:
        date_to_ts = parse_date_filter(date_to)

    # Call storage layer
    result = await self.storage.faceted_search(
        tags=tags,
        tag_match_all=tag_match_all,
        memory_type=memory_type,
        date_from=date_from_ts,
        date_to=date_to_ts,
        page=page,
        page_size=page_size,
    )

    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_memory_service_faceted.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_memory_service_faceted.py
git commit -m "feat(service): add faceted_search to MemoryService

- Parse relative date filters (7d, 1m, 1y)
- Validate memory_type and pagination
- Delegate to storage layer with parsed timestamps"
```

---

## Task 4: MCP Tool - Faceted Search

**Files:**
- Modify: `src/mcp_memory_service/mcp_server.py:578` (add tool after list_memories)
- Test: `tests/integration/test_mcp_faceted_search.py`

**Step 1: Write failing integration test**

```python
# tests/integration/test_mcp_faceted_search.py
import pytest
from mcp_memory_service.mcp_server import faceted_search
from mcp_memory_service.models.memory import Memory
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_faceted_search_tool_returns_toon_format():
    """Test that MCP tool returns TOON formatted output."""
    # Setup mock context
    ctx = MagicMock()
    memory_service = AsyncMock()
    ctx.request_context.lifespan_context.memory_service = memory_service

    # Mock service response
    memory_service.faceted_search.return_value = {
        "memories": [
            Memory(
                content="Test memory",
                content_hash="abc123",
                tags=["python", "api"],
                memory_type="note",
                created_at=1707907200.0,
                created_at_iso="2024-02-14T10:00:00Z",
                updated_at=1707907200.0,
                updated_at_iso="2024-02-14T10:00:00Z",
                metadata={"key": "value"}
            )
        ],
        "total": 1,
        "page": 1,
        "page_size": 10,
        "has_more": False,
        "total_pages": 1,
    }

    # Test: Call tool
    result = await faceted_search(
        ctx=ctx,
        tags=["python"],
        page=1,
        page_size=10
    )

    # Verify: TOON format
    assert isinstance(result, str)
    assert "# page=1 total=1" in result
    assert "Test memory|python,api|" in result


@pytest.mark.asyncio
async def test_faceted_search_tool_calls_service_correctly():
    """Test that tool passes parameters correctly to service."""
    # Setup
    ctx = MagicMock()
    memory_service = AsyncMock()
    ctx.request_context.lifespan_context.memory_service = memory_service
    memory_service.faceted_search.return_value = {
        "memories": [],
        "total": 0,
        "page": 1,
        "page_size": 10,
        "has_more": False,
        "total_pages": 0,
    }

    # Test: Call with all parameters
    await faceted_search(
        ctx=ctx,
        tags=["python", "api"],
        tag_match_all=True,
        memory_type="decision",
        date_from="7d",
        date_to="2026-02-14T00:00:00Z",
        page=2,
        page_size=20
    )

    # Verify: Service called with correct params
    memory_service.faceted_search.assert_called_once_with(
        tags=["python", "api"],
        tag_match_all=True,
        memory_type="decision",
        date_from="7d",
        date_to="2026-02-14T00:00:00Z",
        page=2,
        page_size=20
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_mcp_faceted_search.py -v`
Expected: FAIL with "ImportError: cannot import name 'faceted_search'"

**Step 3: Implement MCP tool**

```python
# src/mcp_memory_service/mcp_server.py (add after list_memories tool ~line 578)

@mcp.tool()
async def faceted_search(
    ctx: Context,
    tags: list[str] | None = None,
    tag_match_all: bool = False,
    memory_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 10,
) -> str:
    """
    Filter memories by multiple metadata facets.

    Pure filtering tool with chronological ordering (newest first).
    Use retrieve_memory() for semantic search with ranking.

    Args:
        tags: Filter by tags (e.g., ["python", "api"])
        tag_match_all: Matching mode (default: False = OR)
            - False (OR): Returns memories with at least one matching tag
            - True (AND): Returns only memories with all specified tags
        memory_type: Filter by type (note, decision, task, reference)
        date_from: Filter by created_at >= this
            - Relative: "7d" (7 days ago), "1m" (1 month), "1y" (1 year)
            - Absolute: ISO8601 "2026-02-14T10:00:00Z"
        date_to: Filter by created_at <= this (same formats as date_from)
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=1 total=42 page_size=10 has_more=true total_pages=5

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash

        For complete TOON specification, see resource: toon://format/documentation

    Examples:
        - faceted_search(tags=["python", "api"], tag_match_all=False)
          → memories with python OR api tag
        - faceted_search(tags=["python", "api"], tag_match_all=True)
          → memories with python AND api tags
        - faceted_search(memory_type="decision", date_from="30d")
          → decision-type memories from last 30 days
        - faceted_search(tags=["project-alpha"], memory_type="task", date_from="7d")
          → task-type memories tagged project-alpha from last 7 days

    Use this for: Multi-dimensional filtering, browsing by category and date,
    finding specific types of memories within timeframes.
    """
    _t0 = time.perf_counter()

    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.faceted_search(
        tags=tags,
        tag_match_all=tag_match_all,
        memory_type=memory_type,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return _inject_latency(toon_output, _t0)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_mcp_faceted_search.py -v`
Expected: PASS (all 2 tests)

**Step 5: Run all tests to ensure nothing broke**

Run: `uv run pytest tests/unit tests/integration -v --tb=short`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/mcp_memory_service/mcp_server.py tests/integration/test_mcp_faceted_search.py
git commit -m "feat(mcp): add faceted_search MCP tool

- Filter by tags (AND/OR), memory_type, date range
- Support relative dates (7d, 1m, 1y) and ISO8601
- TOON format output with pagination
- Chronological ordering (newest first)"
```

---

## Task 5: Documentation and Quality Gates

**Files:**
- Create: `docs/guides/faceted-search.md`
- Modify: `README.md` (add faceted search to features list)

**Step 1: Write user documentation**

```markdown
# Faceted Search Guide

## Overview

The `faceted_search()` tool enables filtering memories by multiple metadata dimensions simultaneously:
- **Tags**: Filter by one or more tags with AND/OR logic
- **Memory Type**: Filter by type (note, decision, task, reference)
- **Date Range**: Filter by creation date using relative or absolute formats

Results are ordered chronologically (newest first), making this ideal for browsing and filtering rather than semantic search.

## Basic Usage

### Filter by Tags (OR Logic)

Find memories with **any** of the specified tags:

```python
faceted_search(tags=["python", "api"])
# Returns memories tagged with "python" OR "api"
```

### Filter by Tags (AND Logic)

Find memories with **all** of the specified tags:

```python
faceted_search(tags=["python", "api"], tag_match_all=True)
# Returns memories tagged with "python" AND "api"
```

### Filter by Memory Type

```python
faceted_search(memory_type="decision")
# Returns only decision-type memories
```

### Filter by Date Range

**Relative dates** (recommended for common cases):

```python
# Last 7 days
faceted_search(date_from="7d")

# Last month
faceted_search(date_from="1m")

# Last year
faceted_search(date_from="1y")

# Between 30 days ago and 7 days ago
faceted_search(date_from="30d", date_to="7d")
```

**Absolute dates** (for specific timestamps):

```python
faceted_search(
    date_from="2026-01-01T00:00:00Z",
    date_to="2026-01-31T23:59:59Z"
)
```

## Combining Filters

Combine multiple facets for precise filtering:

```python
# Project-alpha tasks from last month
faceted_search(
    tags=["project-alpha"],
    memory_type="task",
    date_from="30d"
)

# High-priority decisions with specific tags
faceted_search(
    tags=["architecture", "security"],
    tag_match_all=True,
    memory_type="decision",
    date_from="90d"
)
```

## Pagination

Control result pages:

```python
# First page (10 results)
faceted_search(tags=["python"], page=1, page_size=10)

# Second page
faceted_search(tags=["python"], page=2, page_size=10)

# Larger page size (max 100)
faceted_search(tags=["python"], page=1, page_size=50)
```

## When to Use

**Use `faceted_search()` when:**
- Browsing memories by category and timeframe
- Finding specific types of memories (decisions, tasks, etc.)
- Filtering by multiple tags with precise logic
- Need chronological ordering (newest first)

**Use `retrieve_memory()` instead when:**
- Searching by semantic meaning/content
- Need relevance-ranked results
- Query is natural language, not metadata

**Use `search_by_tag()` instead when:**
- Only filtering by tags (no type or date needed)
- Simple tag search is sufficient

## Date Format Reference

| Format | Example | Description |
|--------|---------|-------------|
| Relative days | `"7d"` | 7 days ago |
| Relative months | `"1m"` | ~30 days ago |
| Relative years | `"1y"` | ~365 days ago |
| ISO8601 | `"2026-02-14T10:00:00Z"` | Absolute UTC timestamp |

## Output Format

Results returned in TOON (Terser Object Notation) format:

```
# page=1 total=42 page_size=10 has_more=true total_pages=5
Memory content|tag1,tag2|{"meta":"data"}|2026-02-14T10:00:00Z|2026-02-14T10:00:00Z|hash123
Another memory|tag3|{}|2026-02-13T15:00:00Z|2026-02-13T15:00:00Z|hash456
```

First line: pagination metadata
Subsequent lines: pipe-delimited memory records

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Invalid date format | Wrong date string | Use "7d" or ISO8601 format |
| Invalid memory_type | Unknown type | Use: note, decision, task, or reference |
| Page < 1 | Invalid page number | Use page >= 1 |
| Page size > 100 | Too large | Use page_size <= 100 |
```

Save to: `docs/guides/faceted-search.md`

**Step 2: Update README**

Add to features list in README.md:

```markdown
- **Faceted Search**: Filter memories by tags (AND/OR), memory type, and date range
```

**Step 3: Run quality gates**

```bash
# Format check
uv run ruff format --check src/ tests/

# Lint check
uv run ruff check src/ tests/

# Run all tests
uv run pytest tests/ -v
```

**Step 4: Commit documentation**

```bash
git add docs/guides/faceted-search.md README.md
git commit -m "docs: add faceted search user guide

- Usage examples for tags, type, date filtering
- Relative vs absolute date formats
- When to use faceted_search vs retrieve_memory
- Error handling reference"
```

**Step 5: Final verification**

Run complete test suite:

```bash
uv run pytest tests/ -v --cov=src/mcp_memory_service --cov-report=term-missing
```

Expected: All tests pass, coverage >90% for new code

---

## Summary

**Implementation complete when:**
- [ ] Date parsing utilities (Task 1) ✓
- [ ] Storage layer faceted_search (Task 2) ✓
- [ ] Service layer faceted_search (Task 3) ✓
- [ ] MCP tool faceted_search (Task 4) ✓
- [ ] Documentation and quality gates (Task 5) ✓

**Total commits:** 5 (one per task)

**Files created:** 3
- `src/mcp_memory_service/utils/date_parsing.py`
- `tests/unit/test_date_parsing.py`
- `docs/guides/faceted-search.md`

**Files modified:** 5
- `src/mcp_memory_service/storage/base.py`
- `src/mcp_memory_service/storage/qdrant_storage.py`
- `src/mcp_memory_service/services/memory_service.py`
- `src/mcp_memory_service/mcp_server.py`
- `README.md`

**Next steps after implementation:**
1. Mark bead mm-mc4z as completed
2. Create PR for review (if using refinery workflow)
3. Deploy to staging for integration testing
