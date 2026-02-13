from unittest.mock import AsyncMock

import pytest

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
    service = MemoryService(storage=storage)

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
    service = MemoryService(storage=storage)

    # Test: Invalid memory type
    with pytest.raises(ValueError, match="memory_type must be one of"):
        await service.faceted_search(memory_type="invalid", page=1, page_size=10)


@pytest.mark.asyncio
async def test_faceted_search_validates_pagination():
    """Test that service validates pagination params."""
    storage = AsyncMock()
    service = MemoryService(storage=storage)

    # Test: Invalid page
    with pytest.raises(ValueError, match="page must be >= 1"):
        await service.faceted_search(page=0, page_size=10)

    # Test: Invalid page_size
    with pytest.raises(ValueError, match="page_size must be <= 100"):
        await service.faceted_search(page=1, page_size=200)
