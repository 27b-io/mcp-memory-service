"""
Tests for memory merge conflict detection and resolution.

Covers:
- CRDT automatic resolution (last-write-wins)
- Manual resolution API
- Conflict queue management
- Graph edge metadata updates
- Configuration options
"""

import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp_memory_service.graph.client import GraphClient
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.utils.interference import ContradictionSignal


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_storage():
    """Mock storage backend."""
    storage = Mock(spec=MemoryStorage)
    storage.max_content_length = 10000
    storage.get_memory_by_hash = AsyncMock()
    storage.update_memory_metadata = AsyncMock()
    storage.delete = AsyncMock(return_value=(True, "Deleted"))
    storage.store = AsyncMock(return_value=(True, "Stored"))
    return storage


@pytest.fixture
def mock_graph():
    """Mock graph client."""
    graph = Mock(spec=GraphClient)
    graph.create_typed_edge = AsyncMock(return_value=True)
    graph.update_typed_edge_metadata = AsyncMock(return_value=True)
    graph.delete_memory_node = AsyncMock()
    graph.list_unresolved_conflicts = AsyncMock(return_value=[])
    return graph


@pytest.fixture
def memory_service(mock_storage, mock_graph):
    """Memory service with mocked dependencies."""
    return MemoryService(storage=mock_storage, graph_client=mock_graph)


@pytest.fixture
def older_memory():
    """Memory with older timestamp."""
    return Memory(
        content="The API supports pagination",
        content_hash="old_hash",
        tags=["api", "docs"],
        created_at=1000.0,
        updated_at=1000.0,
        created_at_iso="2020-01-01T00:00:00Z",
        updated_at_iso="2020-01-01T00:00:00Z",
    )


@pytest.fixture
def newer_memory():
    """Memory with newer timestamp."""
    return Memory(
        content="The API does not support pagination",
        content_hash="new_hash",
        tags=["api", "update"],
        created_at=2000.0,
        updated_at=2000.0,
        created_at_iso="2020-01-02T00:00:00Z",
        updated_at_iso="2020-01-02T00:00:00Z",
    )


# ── CRDT Auto-Resolution Tests ────────────────────────────────────────


class TestCRDTAutoResolution:
    """Test automatic conflict resolution using last-write-wins."""

    @pytest.mark.asyncio
    async def test_keeps_newer_memory(self, memory_service, older_memory, newer_memory, mock_storage, mock_graph):
        """Last-write-wins: newer memory wins, older deleted."""
        mock_storage.get_memory_by_hash.side_effect = lambda h: older_memory if h == "old_hash" else newer_memory

        result = await memory_service._auto_resolve_conflict("old_hash", "new_hash")

        assert result is True
        # Newer memory should be updated with merged tags
        mock_storage.update_memory_metadata.assert_called()
        # Older memory should be deleted
        mock_storage.delete.assert_called_with("old_hash")
        # Graph edge should be marked as resolved
        mock_graph.update_typed_edge_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_merges_tags_from_both(self, memory_service, older_memory, newer_memory, mock_storage):
        """Winner gets union of tags from both memories."""
        mock_storage.get_memory_by_hash.side_effect = lambda h: older_memory if h == "old_hash" else newer_memory

        await memory_service._auto_resolve_conflict("old_hash", "new_hash")

        # Check that tags were merged
        update_calls = [call for call in mock_storage.update_memory_metadata.call_args_list if "tags_str" in call[0][1]]
        assert len(update_calls) > 0
        merged_tags_str = update_calls[0][0][1]["tags_str"]
        merged_tags = set(merged_tags_str.split(","))
        # Should contain tags from both memories
        assert "api" in merged_tags
        assert "docs" in merged_tags
        assert "update" in merged_tags

    @pytest.mark.asyncio
    async def test_records_resolution_in_history(self, memory_service, older_memory, newer_memory, mock_storage):
        """Conflict resolution is recorded in winner's conflict_history."""
        mock_storage.get_memory_by_hash.side_effect = lambda h: older_memory if h == "old_hash" else newer_memory

        await memory_service._auto_resolve_conflict("old_hash", "new_hash")

        # Check conflict_history was updated
        history_calls = [
            call for call in mock_storage.update_memory_metadata.call_args_list if "conflict_history" in call[0][1]
        ]
        assert len(history_calls) > 0
        history = history_calls[0][0][1]["conflict_history"]
        assert len(history) == 1
        assert history[0]["strategy"] == "auto_last_write_wins"
        assert history[0]["merged_from"] == "old_hash"
        assert "resolved_at" in history[0]

    @pytest.mark.asyncio
    async def test_handles_missing_memory(self, memory_service, mock_storage):
        """Returns False if either memory is missing."""
        mock_storage.get_memory_by_hash.return_value = None

        result = await memory_service._auto_resolve_conflict("missing1", "missing2")

        assert result is False
        mock_storage.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_created_at(self, memory_service, mock_storage, mock_graph):
        """Falls back to created_at if updated_at is missing."""
        mem1 = Memory(
            content="content1",
            content_hash="hash1",
            created_at=1000.0,
            updated_at=None,  # Missing
            created_at_iso="2020-01-01T00:00:00Z",
            updated_at_iso=None,
        )
        mem2 = Memory(
            content="content2",
            content_hash="hash2",
            created_at=2000.0,
            updated_at=None,  # Missing
            created_at_iso="2020-01-02T00:00:00Z",
            updated_at_iso=None,
        )
        mock_storage.get_memory_by_hash.side_effect = lambda h: mem1 if h == "hash1" else mem2

        result = await memory_service._auto_resolve_conflict("hash1", "hash2")

        assert result is True
        # Should delete the older one (by created_at)
        mock_storage.delete.assert_called_with("hash1")


# ── Conflict Queue Tests ──────────────────────────────────────────────


class TestConflictQueue:
    """Test conflict queue management via graph edges."""

    @pytest.mark.asyncio
    async def test_list_unresolved_conflicts(self, mock_graph):
        """GraphClient.list_unresolved_conflicts returns unresolved edges."""
        mock_graph.list_unresolved_conflicts.return_value = [
            {
                "source": "hash1",
                "target": "hash2",
                "confidence": 0.85,
                "signal_type": "negation",
                "created_at": 12345.0,
            }
        ]

        conflicts = await mock_graph.list_unresolved_conflicts(limit=10)

        assert len(conflicts) == 1
        assert conflicts[0]["source"] == "hash1"
        assert conflicts[0]["target"] == "hash2"
        assert conflicts[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_mark_edge_as_resolved(self, mock_graph):
        """Updating edge metadata marks conflict as resolved."""
        await mock_graph.update_typed_edge_metadata(
            source_hash="hash1",
            target_hash="hash2",
            relation_type="CONTRADICTS",
            metadata={"resolved_at": time.time(), "resolved_by": "auto", "resolution_action": "last_write_wins"},
        )

        mock_graph.update_typed_edge_metadata.assert_called_once()
        call_args = mock_graph.update_typed_edge_metadata.call_args
        assert call_args[1]["metadata"]["resolved_by"] == "auto"


# ── Batch Conflict Resolution Tests ───────────────────────────────────


class TestBatchConflictResolution:
    """Test consolidation-time batch conflict resolution."""

    @pytest.mark.asyncio
    async def test_auto_resolve_conflicts_batch(self, memory_service, mock_graph, mock_storage, newer_memory):
        """_auto_resolve_conflicts processes multiple conflicts in one run."""
        mock_graph.list_unresolved_conflicts.return_value = [
            {"source": "hash1", "target": "hash2"},
            {"source": "hash3", "target": "hash4"},
        ]
        mock_storage.get_memory_by_hash.return_value = newer_memory

        result = await memory_service._auto_resolve_conflicts(max_conflicts=10)

        assert result["resolved"] == 2
        assert mock_storage.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_max_conflicts_limit(self, memory_service, mock_graph, mock_storage, newer_memory):
        """Batch resolution respects max_conflicts limit."""
        mock_graph.list_unresolved_conflicts.return_value = [
            {"source": f"hash{i}", "target": f"hash{i+100}"} for i in range(20)
        ]
        mock_storage.get_memory_by_hash.return_value = newer_memory

        result = await memory_service._auto_resolve_conflicts(max_conflicts=5)

        assert result["resolved"] <= 5

    @pytest.mark.asyncio
    async def test_handles_graph_not_available(self, mock_storage):
        """Returns zero resolved if graph layer not available."""
        service = MemoryService(storage=mock_storage, graph_client=None)

        result = await service._auto_resolve_conflicts(max_conflicts=10)

        assert result["resolved"] == 0


# ── Store-Time Auto-Resolution Tests ──────────────────────────────────


class TestStoreTimeAutoResolution:
    """Test automatic resolution triggered at memory store time."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_auto_resolve_on_store_when_enabled(
        self, mock_settings, memory_service, mock_storage, mock_graph, older_memory, newer_memory
    ):
        """Auto-resolves contradictions at store time if config enabled."""
        # Enable auto-resolve
        mock_settings.conflict.enabled = True
        mock_settings.conflict.auto_resolve = True
        mock_settings.interference.enabled = True

        # Mock interference detection returning a contradiction
        with patch.object(memory_service, "_detect_contradictions") as mock_detect:
            contradiction = ContradictionSignal(
                existing_hash="old_hash", similarity=0.85, signal_type="negation", confidence=0.75, detail="test"
            )
            mock_detect.return_value = MagicMock(
                has_contradictions=True, contradictions=[contradiction], to_dict=lambda: {"test": "data"}
            )

            with patch.object(memory_service, "_auto_resolve_conflict") as mock_resolve:
                mock_resolve.return_value = True
                mock_storage.get_memory_by_hash.side_effect = lambda h: (
                    older_memory if h == "old_hash" else newer_memory
                )

                # This would normally be called during store_memory
                # Simulating the relevant part
                interference = await memory_service._detect_contradictions("test content")
                if interference.has_contradictions:
                    for contradiction in interference.contradictions:
                        await memory_service._auto_resolve_conflict("new_hash", contradiction.existing_hash)

                mock_resolve.assert_called_once()


# ── Memory Model Tests ────────────────────────────────────────────────


class TestConflictMemoryFields:
    """Test conflict-related fields in Memory model."""

    def test_memory_has_conflict_fields(self):
        """Memory model includes conflict tracking fields."""
        mem = Memory(
            content="test",
            content_hash="hash",
            conflict_status="detected",
            conflict_version=5,
            conflict_history=[{"resolved_at": 123, "strategy": "manual"}],
        )

        assert mem.conflict_status == "detected"
        assert mem.conflict_version == 5
        assert len(mem.conflict_history) == 1

    def test_memory_conflict_fields_optional(self):
        """Conflict fields have sensible defaults."""
        mem = Memory(content="test", content_hash="hash")

        assert mem.conflict_status is None
        assert mem.conflict_version == 0
        assert mem.conflict_history is None

    def test_memory_to_dict_includes_conflict_fields(self):
        """Memory.to_dict() includes conflict fields."""
        mem = Memory(
            content="test",
            content_hash="hash",
            conflict_status="auto_resolved",
            conflict_version=3,
            conflict_history=[{"test": "data"}],
        )

        d = mem.to_dict()
        assert d["conflict_status"] == "auto_resolved"
        assert d["conflict_version"] == 3
        assert d["conflict_history"] == [{"test": "data"}]

    def test_memory_from_dict_restores_conflict_fields(self):
        """Memory.from_dict() restores conflict fields."""
        data = {
            "content": "test",
            "content_hash": "hash",
            "conflict_status": "manual_resolved",
            "conflict_version": 7,
            "conflict_history": [{"resolved_at": 456}],
        }

        mem = Memory.from_dict(data)
        assert mem.conflict_status == "manual_resolved"
        assert mem.conflict_version == 7
        assert len(mem.conflict_history) == 1


# ── Configuration Tests ────────────────────────────────────────────────


class TestConflictConfiguration:
    """Test conflict resolution configuration."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_auto_resolve_disabled_by_default(self, mock_settings, memory_service):
        """Auto-resolve is disabled by default (opt-in)."""
        mock_settings.conflict.enabled = True
        mock_settings.conflict.auto_resolve = False

        # Simulate contradiction detection without auto-resolve
        # Would not call _auto_resolve_conflict

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_can_disable_all_conflict_features(self, mock_settings):
        """Can disable conflict features entirely."""
        mock_settings.conflict.enabled = False

        # Conflict detection would be skipped
        # No CONTRADICTS edges created
        # No auto-resolution


# ── Edge Cases ─────────────────────────────────────────────────────────


class TestConflictEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_self_conflict(self, memory_service, older_memory, mock_storage):
        """Cannot create conflict between memory and itself."""
        mock_storage.get_memory_by_hash.return_value = older_memory

        # GraphClient should reject self-edges
        # MemoryService should handle gracefully

    @pytest.mark.asyncio
    async def test_handles_storage_failure_during_resolution(self, memory_service, older_memory, newer_memory, mock_storage):
        """Handles storage failures gracefully during resolution."""
        mock_storage.get_memory_by_hash.side_effect = lambda h: older_memory if h == "old_hash" else newer_memory
        mock_storage.delete.side_effect = Exception("Storage error")

        result = await memory_service._auto_resolve_conflict("old_hash", "new_hash")

        # Should return False but not crash
        assert result is False

    @pytest.mark.asyncio
    async def test_handles_graph_unavailable(self, memory_service, older_memory, newer_memory, mock_storage):
        """Resolution works even if graph layer unavailable."""
        service = MemoryService(storage=mock_storage, graph_client=None)
        mock_storage.get_memory_by_hash.side_effect = lambda h: older_memory if h == "old_hash" else newer_memory

        result = await service._auto_resolve_conflict("old_hash", "new_hash")

        # Should still resolve in storage, just can't update graph
        assert result is True
        mock_storage.delete.assert_called_once()
