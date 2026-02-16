"""Unit tests for MemoryService quota integration."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.config import QuotaSettings
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.services.quota_service import QuotaService
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.utils.quota import QuotaExceededError


@pytest.fixture
def mock_storage():
    """Create mock storage."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 10000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.search_by_tags.return_value = []
    return storage


@pytest.fixture
def quota_settings():
    """Create quota settings."""
    return QuotaSettings(
        enabled=True,
        max_memories=100,
        max_storage_bytes=10000,
        max_memories_per_hour=10,
    )


@pytest.fixture
def quota_service(mock_storage, quota_settings):
    """Create quota service."""
    return QuotaService(storage=mock_storage, settings=quota_settings)


@pytest.fixture
def memory_service_with_quota(mock_storage, quota_service):
    """Create MemoryService with quota enforcement."""
    return MemoryService(
        storage=mock_storage,
        quota_service=quota_service,
    )


@pytest.fixture
def memory_service_without_quota(mock_storage):
    """Create MemoryService without quota enforcement."""
    return MemoryService(storage=mock_storage)


class TestMemoryServiceQuotaIntegration:
    """Test quota integration in MemoryService."""

    @pytest.mark.asyncio
    async def test_store_without_quota_service(self, memory_service_without_quota, mock_storage):
        """Test storing when quota service is None (no enforcement)."""
        result = await memory_service_without_quota.store_memory(
            content="Test memory",
            client_hostname="test-client",
        )

        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_under_limit(self, memory_service_with_quota, mock_storage):
        """Test storing when under quota limits."""
        # Mock quota check returns status with no warnings
        mock_storage.search_by_tags.return_value = []

        result = await memory_service_with_quota.store_memory(
            content="Test memory",
            client_hostname="test-client",
        )

        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_warning(self, memory_service_with_quota, mock_storage):
        """Test storing when approaching quota limits (warning)."""
        # Mock 85 memories (85% usage triggers low warning)
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=1000.0,
                updated_at=1000.0,
            )
            for i in range(85)
        ]
        mock_storage.search_by_tags.return_value = memories

        result = await memory_service_with_quota.store_memory(
            content="Test memory",
            client_hostname="test-client",
        )

        # Should succeed but log warning
        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_exceeded(self, memory_service_with_quota, mock_storage):
        """Test storing when quota limit is exceeded."""
        # Mock 101 memories (exceeds limit of 100)
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=1000.0,
                updated_at=1000.0,
            )
            for i in range(101)
        ]
        mock_storage.search_by_tags.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await memory_service_with_quota.store_memory(
                content="Test memory",
                client_hostname="test-client",
            )

        assert exc_info.value.quota_type == "memory_count"
        # Store should NOT be called
        mock_storage.store.assert_not_called()
