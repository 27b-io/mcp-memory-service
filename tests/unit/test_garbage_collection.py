"""
Unit tests for memory garbage collection.

Tests the GC cycle: finding orphaned memories, age filtering, and deletion.
All graph/storage interactions are mocked — integration tests with real backends are separate.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.delete.return_value = (True, "Deleted")
    return storage


@pytest.fixture
def mock_graph():
    """Create a mock GraphClient with GC methods."""
    graph = AsyncMock()
    graph.get_orphan_nodes.return_value = []
    graph.delete_memory_node = AsyncMock()
    return graph


@pytest.fixture
def gc_config():
    """Create a mock GC config."""
    config = MagicMock()
    config.enabled = True
    config.retention_days = 90
    config.max_deletions_per_run = 1000
    return config


@pytest.fixture
def service_with_graph(mock_storage, mock_graph):
    """Create a MemoryService with graph layer enabled."""
    return MemoryService(storage=mock_storage, graph_client=mock_graph)


@pytest.fixture
def service_without_graph(mock_storage):
    """Create a MemoryService without graph layer."""
    return MemoryService(storage=mock_storage)


def _make_memory_dict(content_hash: str, created_at: float) -> dict:
    """Helper to create memory dict for get_memory_by_hash return value."""
    return {
        "content": "test",
        "content_hash": content_hash,
        "tags": [],
        "memory_type": "note",
        "created_at": created_at,
        "updated_at": created_at,
    }


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gc_disabled(service_with_graph, gc_config):
    """Test GC returns early when disabled."""
    gc_config.enabled = False

    with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
        result = await service_with_graph.garbage_collect()

    assert result["enabled"] is False
    assert result["orphans_found"] == 0
    assert result["orphans_deleted"] == 0


@pytest.mark.asyncio
async def test_gc_no_graph_layer(service_without_graph, gc_config):
    """Test GC returns error when graph layer not enabled."""
    with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
        result = await service_without_graph.garbage_collect()

    assert result["enabled"] is True
    assert "Graph layer not enabled" in result["errors"]
    assert result["orphans_found"] == 0


@pytest.mark.asyncio
async def test_gc_no_orphans(service_with_graph, mock_graph, gc_config):
    """Test GC when no orphans exist."""
    mock_graph.get_orphan_nodes.return_value = []

    with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
        result = await service_with_graph.garbage_collect()

    assert result["orphans_found"] == 0
    assert result["orphans_deleted"] == 0
    assert result.get("success", True) is True
    mock_graph.get_orphan_nodes.assert_called_once_with(limit=gc_config.max_deletions_per_run)


@pytest.mark.asyncio
async def test_gc_orphans_too_young(service_with_graph, mock_graph, gc_config):
    """Test GC skips orphans younger than retention period."""
    # Orphans created recently (within retention period)
    recent_time = time.time() - (30 * 86400)  # 30 days ago (< 90 day retention)
    orphan_hashes = ["recent_orphan_1", "recent_orphan_2"]
    mock_graph.get_orphan_nodes.return_value = orphan_hashes

    # Mock get_memory_by_hash to return recent memories
    async def mock_get_memory(content_hash):
        return _make_memory_dict(content_hash, created_at=recent_time)

    with patch.object(service_with_graph, "get_memory_by_hash", side_effect=mock_get_memory):
        with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
            result = await service_with_graph.garbage_collect()

    assert result["orphans_found"] == 2
    assert result["orphans_eligible"] == 0  # None old enough
    assert result["orphans_deleted"] == 0


@pytest.mark.asyncio
async def test_gc_deletes_old_orphans(service_with_graph, mock_graph, gc_config):
    """Test GC successfully deletes old orphaned memories."""
    # Orphans created 120 days ago (> 90 day retention)
    old_time = time.time() - (120 * 86400)
    orphan_hashes = ["old_orphan_1", "old_orphan_2", "old_orphan_3"]
    mock_graph.get_orphan_nodes.return_value = orphan_hashes

    # Mock get_memory_by_hash to return old memories
    async def mock_get_memory(content_hash):
        return _make_memory_dict(content_hash, created_at=old_time)

    # Mock delete_memory to succeed
    async def mock_delete(content_hash):
        return {"success": True, "message": f"Deleted {content_hash}"}

    with patch.object(service_with_graph, "get_memory_by_hash", side_effect=mock_get_memory):
        with patch.object(service_with_graph, "delete_memory", side_effect=mock_delete):
            with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
                result = await service_with_graph.garbage_collect()

    assert result["orphans_found"] == 3
    assert result["orphans_eligible"] == 3
    assert result["orphans_deleted"] == 3
    assert result["success"] is True


@pytest.mark.asyncio
async def test_gc_mixed_ages(service_with_graph, mock_graph, gc_config):
    """Test GC correctly filters by age in mixed scenarios."""
    # Mix of old and recent orphans
    old_time = time.time() - (120 * 86400)  # 120 days old
    recent_time = time.time() - (30 * 86400)  # 30 days old
    orphan_hashes = ["old_1", "recent_1", "old_2", "recent_2"]
    mock_graph.get_orphan_nodes.return_value = orphan_hashes

    # Mock get_memory_by_hash to return mixed ages
    async def mock_get_memory(content_hash):
        created_at = old_time if "old" in content_hash else recent_time
        return _make_memory_dict(content_hash, created_at=created_at)

    # Mock delete_memory to succeed
    async def mock_delete(content_hash):
        return {"success": True, "message": f"Deleted {content_hash}"}

    with patch.object(service_with_graph, "get_memory_by_hash", side_effect=mock_get_memory):
        with patch.object(service_with_graph, "delete_memory", side_effect=mock_delete):
            with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
                result = await service_with_graph.garbage_collect()

    assert result["orphans_found"] == 4
    assert result["orphans_eligible"] == 2  # Only the old ones
    assert result["orphans_deleted"] == 2
    assert result["success"] is True


@pytest.mark.asyncio
async def test_gc_handles_deletion_errors(service_with_graph, mock_graph, gc_config):
    """Test GC handles errors during deletion gracefully."""
    old_time = time.time() - (120 * 86400)
    orphan_hashes = ["orphan_1", "orphan_2"]
    mock_graph.get_orphan_nodes.return_value = orphan_hashes

    # Mock get_memory_by_hash
    async def mock_get_memory(content_hash):
        return _make_memory_dict(content_hash, created_at=old_time)

    # Mock delete_memory to fail on second call
    delete_call_count = 0

    async def mock_delete(content_hash):
        nonlocal delete_call_count
        delete_call_count += 1
        if delete_call_count == 2:
            raise Exception("Deletion failed")
        return {"success": True, "message": f"Deleted {content_hash}"}

    with patch.object(service_with_graph, "get_memory_by_hash", side_effect=mock_get_memory):
        with patch.object(service_with_graph, "delete_memory", side_effect=mock_delete):
            with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
                result = await service_with_graph.garbage_collect()

    assert result["orphans_found"] == 2
    assert result["orphans_eligible"] == 2
    assert result["orphans_deleted"] == 1  # One succeeded, one failed
    assert len(result["errors"]) > 0  # Error was logged
    assert result["success"] is False


@pytest.mark.asyncio
async def test_gc_respects_max_deletions(service_with_graph, mock_graph, gc_config):
    """Test GC respects max deletions per run limit."""
    gc_config.max_deletions_per_run = 5

    with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
        await service_with_graph.garbage_collect()

    # Verify get_orphan_nodes was called with the limit
    mock_graph.get_orphan_nodes.assert_called_once_with(limit=5)


@pytest.mark.asyncio
async def test_gc_custom_retention_period(service_with_graph, mock_graph, gc_config):
    """Test GC with custom retention period."""
    gc_config.retention_days = 30  # Shorter retention period

    # Orphan created 45 days ago (> 30 day retention, < 90 day default)
    mid_age_time = time.time() - (45 * 86400)
    orphan_hashes = ["mid_age_orphan"]
    mock_graph.get_orphan_nodes.return_value = orphan_hashes

    async def mock_get_memory(content_hash):
        return _make_memory_dict(content_hash, created_at=mid_age_time)

    async def mock_delete(content_hash):
        return {"success": True, "message": f"Deleted {content_hash}"}

    with patch.object(service_with_graph, "get_memory_by_hash", side_effect=mock_get_memory):
        with patch.object(service_with_graph, "delete_memory", side_effect=mock_delete):
            with patch("mcp_memory_service.services.memory_service.settings.gc", gc_config):
                result = await service_with_graph.garbage_collect()

    # Should be deleted with 30-day retention, but not with 90-day
    assert result["orphans_eligible"] == 1
    assert result["orphans_deleted"] == 1
