"""Unit tests for QuotaService."""

import time
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.config import QuotaSettings
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.quota_service import QuotaService
from mcp_memory_service.utils.quota import QuotaExceededError


@pytest.fixture
def quota_settings():
    """Create quota settings for testing."""
    return QuotaSettings(
        enabled=True,
        max_memories=100,
        max_storage_bytes=1000,
        max_memories_per_hour=10,
        rate_limit_window_seconds=3600,
        warning_threshold_low=0.8,
        warning_threshold_high=0.9,
    )


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = AsyncMock()
    # Set up async method
    storage.search_by_tags = AsyncMock()
    return storage


@pytest.fixture
def quota_service(mock_storage, quota_settings):
    """Create QuotaService instance."""
    return QuotaService(storage=mock_storage, settings=quota_settings)


class TestQuotaServiceMemoryCount:
    """Test memory count quota enforcement."""

    @pytest.mark.asyncio
    async def test_under_limit(self, quota_service, mock_storage):
        """Test when memory count is under limit."""
        # Mock 50 memories for client (created 2 hours ago to avoid rate limit)
        old_time = time.time() - 7200  # 2 hours ago
        memories = [
            Memory(
                content=f"Memory {i}",
                content_hash=f"hash_{i}",
                tags=["source:test-client"],
                created_at=old_time,
                updated_at=old_time,
            )
            for i in range(50)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_count == 50
        assert status.memory_limit == 100
        assert status.memory_usage_pct == 0.5
        assert status.has_warning is False

    @pytest.mark.asyncio
    async def test_warning_threshold_low(self, quota_service, mock_storage):
        """Test low warning threshold (80%)."""
        # 85 memories = 85% usage (created 2 hours ago to avoid rate limit)
        old_time = time.time() - 7200  # 2 hours ago
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=old_time,
                updated_at=old_time,
            )
            for i in range(85)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_usage_pct == 0.85
        assert status.has_warning is True
        assert status.warning_level == "low"

    @pytest.mark.asyncio
    async def test_warning_threshold_high(self, quota_service, mock_storage):
        """Test high warning threshold (90%)."""
        # 92 memories = 92% usage (created 2 hours ago to avoid rate limit)
        old_time = time.time() - 7200  # 2 hours ago
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=old_time,
                updated_at=old_time,
            )
            for i in range(92)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_usage_pct == 0.92
        assert status.has_warning is True
        assert status.warning_level == "high"

    @pytest.mark.asyncio
    async def test_limit_exceeded(self, quota_service, mock_storage):
        """Test when memory count limit is exceeded."""
        # 101 memories exceeds limit of 100
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(101)
        ]
        mock_storage.search_by_tags.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "memory_count"
        assert exc_info.value.current == 101
        assert exc_info.value.limit == 100


class TestQuotaServiceStorageSize:
    """Test storage size quota enforcement."""

    @pytest.mark.asyncio
    async def test_storage_under_limit(self, quota_service, mock_storage):
        """Test when storage size is under limit."""
        # Each memory has 10 bytes, 50 memories = 500 bytes (created 2 hours ago to avoid rate limit)
        old_time = time.time() - 7200  # 2 hours ago
        memories = [
            Memory(
                content="0123456789",  # 10 bytes
                content_hash=f"hash_{i}",
                tags=["source:test-client"],
                created_at=old_time,
                updated_at=old_time,
            )
            for i in range(50)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.storage_bytes == 500
        assert status.storage_limit == 1000
        assert status.storage_usage_pct == 0.5

    @pytest.mark.asyncio
    async def test_storage_limit_exceeded(self, quota_service, mock_storage):
        """Test when storage size limit is exceeded."""
        # Create memory with 1001 bytes
        memories = [
            Memory(
                content="a" * 1001,
                content_hash="hash_big",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
        ]
        mock_storage.search_by_tags.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "storage_size"
        assert exc_info.value.current == 1001
        assert exc_info.value.limit == 1000


class TestQuotaServiceRateLimit:
    """Test rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_rate_under_limit(self, quota_service, mock_storage):
        """Test when rate is under limit."""
        now = time.time()
        # 5 memories in last hour
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=now - 1800,  # 30 minutes ago
                updated_at=now - 1800,
            )
            for i in range(5)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memories_last_hour == 5
        assert status.rate_limit == 10
        assert status.rate_usage_pct == 0.5

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, quota_service, mock_storage):
        """Test when rate limit is exceeded."""
        now = time.time()
        # 11 memories in last hour (exceeds limit of 10)
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=now - 1800,
                updated_at=now - 1800,
            )
            for i in range(11)
        ]
        mock_storage.search_by_tags.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "rate_limit"
        assert exc_info.value.current == 11
        assert exc_info.value.limit == 10
        assert exc_info.value.retry_after is not None

    @pytest.mark.asyncio
    async def test_rate_limit_old_memories_ignored(self, quota_service, mock_storage):
        """Test that memories outside rate window are ignored."""
        now = time.time()
        # 5 recent + 10 old (outside window)
        memories = [
            Memory(
                content=f"Recent{i}",
                content_hash=f"hr{i}",
                tags=["source:test-client"],
                created_at=now - 1800,  # 30 min ago (inside window)
                updated_at=now - 1800,
            )
            for i in range(5)
        ] + [
            Memory(
                content=f"Old{i}",
                content_hash=f"ho{i}",
                tags=["source:test-client"],
                created_at=now - 7200,  # 2 hours ago (outside window)
                updated_at=now - 7200,
            )
            for i in range(10)
        ]
        mock_storage.search_by_tags.return_value = memories

        status = await quota_service.check_quota("test-client")

        # Only 5 recent memories should count
        assert status.memories_last_hour == 5
        assert status.has_warning is False


class TestQuotaServiceGetStatus:
    """Test get_quota_status method (no exceptions)."""

    @pytest.mark.asyncio
    async def test_get_status_no_exception(self, quota_service, mock_storage):
        """Test get_quota_status never raises, even when exceeded."""
        # Exceed all limits
        memories = [
            Memory(
                content="a" * 20,  # 20 bytes each
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(101)  # Exceeds memory count
        ]
        mock_storage.search_by_tags.return_value = memories

        # Should NOT raise, just return status
        status = await quota_service.get_quota_status("test-client")

        assert status.memory_count == 101
        assert status.memory_usage_pct > 1.0  # Over 100%
