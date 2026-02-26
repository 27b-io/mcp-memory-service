"""
Tests for temporal memory query features (mm-ja4jf).

Covers:
1. MemoryService.query_time_range — date-range queries with NL time expressions
2. MemoryService.temporal_analysis — memory distribution over time buckets
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    return storage


@pytest.fixture
def service(mock_storage):
    return MemoryService(storage=mock_storage)


def _make_memory(content_hash: str, content: str, days_ago: float, tags: list[str] | None = None) -> Memory:
    ts = time.time() - (days_ago * 86400)
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or [],
        memory_type="note",
        created_at=ts,
        updated_at=ts,
    )


# ---------------------------------------------------------------------------
# query_time_range
# ---------------------------------------------------------------------------


class TestQueryTimeRange:
    """Tests for MemoryService.query_time_range."""

    @pytest.mark.asyncio
    async def test_explicit_date_range_passes_timestamps_to_storage(self, service, mock_storage):
        """start_date/end_date should be converted to timestamps and passed to storage."""
        memories = [_make_memory("h1", "Test memory", 2)]
        mock_storage.get_all_memories.return_value = memories
        mock_storage.count_all_memories.return_value = 1

        result = await service.query_time_range(start_date="2026-02-01", end_date="2026-02-28")

        assert result["total"] == 1
        assert len(result["memories"]) == 1
        # Storage must have been called with timestamp bounds
        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["start_timestamp"] is not None
        assert call_kwargs["end_timestamp"] is not None
        # start should be before end
        assert call_kwargs["start_timestamp"] < call_kwargs["end_timestamp"]

    @pytest.mark.asyncio
    async def test_natural_language_expression_resolved(self, service, mock_storage):
        """NL time_expr should be parsed and timestamps forwarded to storage."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        result = await service.query_time_range(time_expr="last week")

        assert "time_range" in result
        assert result["time_range"]["start"] is not None
        assert result["time_range"]["end"] is not None
        # Storage called with parsed timestamps
        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["start_timestamp"] is not None

    @pytest.mark.asyncio
    async def test_time_range_in_response_shows_resolved_bounds(self, service, mock_storage):
        """Response includes resolved ISO timestamps for transparency."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        result = await service.query_time_range(start_date="2026-01-01", end_date="2026-01-31")

        assert "time_range" in result
        assert "start" in result["time_range"]
        assert "end" in result["time_range"]

    @pytest.mark.asyncio
    async def test_tag_filter_forwarded_to_storage(self, service, mock_storage):
        """tags param should be forwarded to storage alongside time range."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        await service.query_time_range(start_date="2026-02-01", end_date="2026-02-28", tags=["rathole"])

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["tags"] == ["rathole"]

    @pytest.mark.asyncio
    async def test_memory_type_filter_forwarded(self, service, mock_storage):
        """memory_type filter is forwarded to storage."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        await service.query_time_range(start_date="2026-02-01", end_date="2026-02-28", memory_type="decision")

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["memory_type"] == "decision"

    @pytest.mark.asyncio
    async def test_pagination_metadata_returned(self, service, mock_storage):
        """Standard pagination fields are present in result."""
        mock_storage.get_all_memories.return_value = [_make_memory("h1", "x", 1)]
        mock_storage.count_all_memories.return_value = 15

        result = await service.query_time_range(start_date="2026-02-01", end_date="2026-02-28", page=1, page_size=10)

        assert result["total"] == 15
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert result["has_more"] is True
        assert result["total_pages"] == 2

    @pytest.mark.asyncio
    async def test_no_time_bounds_raises_or_errors(self, service, mock_storage):
        """Calling with no time constraints returns an error response."""
        result = await service.query_time_range()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_storage_error_returns_safe_response(self, service, mock_storage):
        """Storage failures should not propagate as exceptions."""
        mock_storage.get_all_memories.side_effect = RuntimeError("Qdrant down")

        result = await service.query_time_range(start_date="2026-02-01", end_date="2026-02-28")

        assert "error" in result
        assert result["memories"] == []


# ---------------------------------------------------------------------------
# temporal_analysis
# ---------------------------------------------------------------------------


class TestTemporalAnalysis:
    """Tests for MemoryService.temporal_analysis."""

    @pytest.mark.asyncio
    async def test_returns_buckets(self, service, mock_storage):
        """Result must include a non-empty buckets list."""
        memories = [
            _make_memory("h1", "mem 1", 1),
            _make_memory("h2", "mem 2", 2),
            _make_memory("h3", "mem 3", 3),
        ]
        mock_storage.get_all_memories.return_value = memories

        result = await service.temporal_analysis(bucket="day", lookback_days=7)

        assert "buckets" in result
        assert isinstance(result["buckets"], list)

    @pytest.mark.asyncio
    async def test_bucket_counts_sum_to_total(self, service, mock_storage):
        """Sum of bucket counts equals total memories returned."""
        memories = [
            _make_memory("h1", "mem 1", 0),  # today
            _make_memory("h2", "mem 2", 0),  # today
            _make_memory("h3", "mem 3", 1),  # yesterday
        ]
        mock_storage.get_all_memories.return_value = memories

        result = await service.temporal_analysis(bucket="day", lookback_days=7)

        total_in_buckets = sum(b["count"] for b in result["buckets"])
        assert total_in_buckets == 3

    @pytest.mark.asyncio
    async def test_weekly_bucket_groups_correctly(self, service, mock_storage):
        """With bucket=week, memories within the same week share a bucket."""
        # Two memories 2 days apart but same week, one 8 days ago (previous week)
        monday_offset = datetime.now(timezone.utc).weekday()  # days since Monday
        memories = [
            _make_memory("h1", "this week a", monday_offset),  # Monday this week
            _make_memory("h2", "this week b", monday_offset - 1),  # Tuesday this week
            _make_memory("h3", "last week", monday_offset + 7),  # Last week
        ]
        mock_storage.get_all_memories.return_value = memories

        result = await service.temporal_analysis(bucket="week", lookback_days=14)

        # Should have at most 2 unique week buckets
        assert len(result["buckets"]) <= 2

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_buckets(self, service, mock_storage):
        """No memories → empty buckets list and zero total."""
        mock_storage.get_all_memories.return_value = []

        result = await service.temporal_analysis(bucket="day", lookback_days=7)

        assert result["buckets"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_tag_filter_forwarded_to_storage(self, service, mock_storage):
        """Tags filter is passed to storage."""
        mock_storage.get_all_memories.return_value = []

        await service.temporal_analysis(bucket="day", lookback_days=7, tags=["project-x"])

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["tags"] == ["project-x"]

    @pytest.mark.asyncio
    async def test_result_includes_summary_stats(self, service, mock_storage):
        """Result includes total memories and lookback window info."""
        mock_storage.get_all_memories.return_value = [_make_memory("h1", "m", 1)]

        result = await service.temporal_analysis(bucket="day", lookback_days=7)

        assert "total" in result
        assert "lookback_days" in result
        assert result["lookback_days"] == 7

    @pytest.mark.asyncio
    async def test_monthly_bucket_groups_by_month(self, service, mock_storage):
        """bucket=month should group all memories from same month together."""
        # Create memories all in same month
        same_month_ts = time.time() - 5 * 86400  # 5 days ago
        memories = [
            Memory(content=f"m{i}", content_hash=f"h{i}", created_at=same_month_ts + i * 3600, updated_at=same_month_ts)
            for i in range(3)
        ]
        mock_storage.get_all_memories.return_value = memories

        result = await service.temporal_analysis(bucket="month", lookback_days=30)

        # All memories in the same month → 1 bucket
        non_zero_buckets = [b for b in result["buckets"] if b["count"] > 0]
        assert len(non_zero_buckets) == 1
        assert non_zero_buckets[0]["count"] == 3

    @pytest.mark.asyncio
    async def test_storage_error_returns_safe_response(self, service, mock_storage):
        """Storage exceptions don't propagate."""
        mock_storage.get_all_memories.side_effect = RuntimeError("chaos")

        result = await service.temporal_analysis(bucket="day", lookback_days=7)

        assert "error" in result
