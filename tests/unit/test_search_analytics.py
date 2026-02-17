"""Tests for search analytics functionality."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.search_log import SearchLog
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    return storage


@pytest.fixture
def memory_service_with_logs(mock_storage):
    """Create a MemoryService with some search logs."""
    service = MemoryService(storage=mock_storage)

    # Simulate some searches
    service._log_search(
        query="test query 1",
        start_time=1000.0,
        result_count=5,
        tags=["python", "testing"],
        hybrid_enabled=True,
        keywords_extracted=["python"],
    )
    service._log_search(
        query="test query 2",
        start_time=1001.0,
        result_count=3,
        tags=None,
        hybrid_enabled=True,
        keywords_extracted=[],
    )
    service._log_search(
        query="test query 1",  # Duplicate query
        start_time=1002.0,
        result_count=5,
        tags=["python"],
        hybrid_enabled=False,
    )
    service._log_search(
        query="error query",
        start_time=1003.0,
        result_count=0,
        error="Something failed",
    )

    return service


def test_search_log_creation():
    """Test SearchLog model creation."""
    log = SearchLog(
        query="test",
        timestamp=1000.0,
        response_time_ms=50.0,
        result_count=5,
        tags=["python"],
        hybrid_enabled=True,
        keywords_extracted=["python"],
    )

    assert log.query == "test"
    assert log.timestamp == 1000.0
    assert log.response_time_ms == 50.0
    assert log.result_count == 5
    assert log.tags == ["python"]
    assert log.hybrid_enabled is True
    assert log.keywords_extracted == ["python"]
    assert log.error is None


def test_search_log_to_dict():
    """Test SearchLog to_dict conversion."""
    log = SearchLog(
        query="test",
        timestamp=1000.0,
        response_time_ms=50.0,
        result_count=5,
    )

    data = log.to_dict()
    assert data["query"] == "test"
    assert data["timestamp"] == 1000.0
    assert data["response_time_ms"] == 50.0
    assert data["result_count"] == 5


def test_log_search(mock_storage):
    """Test _log_search method."""
    service = MemoryService(storage=mock_storage)

    # Initially no logs
    assert len(service._search_logs) == 0

    # Log a search
    service._log_search(
        query="test",
        start_time=1000.0,
        result_count=5,
    )

    # Should have one log
    assert len(service._search_logs) == 1
    log = service._search_logs[0]
    assert log.query == "test"
    assert log.result_count == 5


def test_get_search_analytics_empty(mock_storage):
    """Test get_search_analytics with no logs."""
    service = MemoryService(storage=mock_storage)

    analytics = service.get_search_analytics()

    assert analytics["total_searches"] == 0
    assert analytics["avg_response_time_ms"] is None
    assert analytics["popular_queries"] == []
    assert analytics["popular_tags"] == []
    assert analytics["error_rate"] == 0.0


def test_get_search_analytics_with_logs(memory_service_with_logs):
    """Test get_search_analytics with search logs."""
    analytics = memory_service_with_logs.get_search_analytics()

    assert analytics["total_searches"] == 4
    assert analytics["avg_response_time_ms"] is not None
    assert len(analytics["popular_queries"]) > 0
    assert analytics["popular_queries"][0]["query"] == "test query 1"  # Most frequent
    assert analytics["popular_queries"][0]["count"] == 2
    assert len(analytics["popular_tags"]) > 0
    assert analytics["search_types"]["hybrid"] == 3  # 3 out of 4 are hybrid
    assert analytics["search_types"]["vector_only"] == 1
    assert analytics["error_rate"] == 25.0  # 1 error out of 4


def test_search_analytics_circular_buffer(mock_storage):
    """Test that search logs use circular buffer (max 10K)."""
    service = MemoryService(storage=mock_storage)

    # Add more than max capacity
    for i in range(service._MAX_SEARCH_LOGS + 100):
        service._log_search(
            query=f"query {i}",
            start_time=1000.0 + i,
            result_count=1,
        )

    # Should be capped at max
    assert len(service._search_logs) == service._MAX_SEARCH_LOGS
