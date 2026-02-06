"""
Unit tests for memory consolidation.

Tests the consolidation cycle: global decay, stale decay, pruning,
orphan detection, and duplicate merging. All graph/storage interactions
are mocked — integration tests with real backends are separate.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
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
    storage.get_all_memories.return_value = []
    storage.retrieve.return_value = []
    storage.delete.return_value = (True, "Deleted")
    storage.update_memory_metadata.return_value = (True, "Updated")
    return storage


@pytest.fixture
def mock_graph():
    """Create a mock GraphClient with consolidation methods."""
    graph = AsyncMock()
    graph.decay_all_edges.return_value = 50
    graph.decay_stale_edges.return_value = 10
    graph.prune_weak_edges.return_value = 5
    graph.get_orphan_nodes.return_value = ["orphan_1", "orphan_2"]
    graph.delete_memory_node = AsyncMock()
    return graph


@pytest.fixture
def consolidation_config():
    """Create a mock consolidation config."""
    config = MagicMock()
    config.decay_factor = 0.9
    config.prune_threshold = 0.05
    config.stale_edge_days = 30
    config.stale_decay_factor = 0.5
    config.max_edges_per_run = 10000
    config.duplicate_similarity_threshold = 0.95
    config.max_duplicates_per_run = 100
    return config


@pytest.fixture
def service_with_graph(mock_storage, mock_graph):
    """Create a MemoryService with graph layer enabled."""
    return MemoryService(storage=mock_storage, graph_client=mock_graph)


@pytest.fixture
def service_without_graph(mock_storage):
    """Create a MemoryService without graph layer."""
    return MemoryService(storage=mock_storage)


def _make_memory(content_hash: str, content: str = "test", tags: list | None = None, created_at: float = 0.0) -> Memory:
    """Helper to create Memory instances."""
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or [],
        memory_type="note",
        metadata={},
        created_at=created_at or time.time(),
        updated_at=created_at or time.time(),
    )


# =============================================================================
# Consolidation Orchestration Tests
# =============================================================================


class TestConsolidationOrchestration:
    """Test the full consolidation cycle."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_full_consolidation_with_graph(self, mock_settings, service_with_graph, mock_graph, consolidation_config):
        mock_settings.consolidation = consolidation_config

        result = await service_with_graph.consolidate()

        assert result["success"] is True
        assert result["edges_decayed"] == 50
        assert result["stale_edges_decayed"] == 10
        assert result["edges_pruned"] == 5
        assert result["orphan_nodes"] == 2
        assert result["errors"] == []

        # Verify phases were called in order with correct params
        mock_graph.decay_all_edges.assert_called_once_with(decay_factor=0.9, limit=10000)
        mock_graph.prune_weak_edges.assert_called_once_with(threshold=0.05, limit=10000)
        mock_graph.get_orphan_nodes.assert_called_once_with(limit=100)

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_consolidation_without_graph_skips_edge_ops(self, mock_settings, service_without_graph, consolidation_config):
        mock_settings.consolidation = consolidation_config

        result = await service_without_graph.consolidate()

        assert result["success"] is True
        assert result["edges_decayed"] == 0
        assert result["edges_pruned"] == 0
        assert result["orphan_nodes"] == 0

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_consolidation_continues_on_phase_failure(
        self, mock_settings, service_with_graph, mock_graph, consolidation_config
    ):
        """Consolidation should continue even if one phase fails."""
        mock_settings.consolidation = consolidation_config
        mock_graph.decay_all_edges.side_effect = RuntimeError("Redis down")
        # Other phases should still run

        result = await service_with_graph.consolidate()

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "global_decay" in result["errors"][0]
        # Prune and orphan detection still ran
        assert result["edges_pruned"] == 5
        assert result["orphan_nodes"] == 2

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_consolidation_is_idempotent(self, mock_settings, service_with_graph, mock_graph, consolidation_config):
        """Running consolidation twice should produce valid results both times."""
        mock_settings.consolidation = consolidation_config

        result1 = await service_with_graph.consolidate()
        result2 = await service_with_graph.consolidate()

        # Both runs succeed — idempotent
        assert result1["success"] is True
        assert result2["success"] is True
        assert mock_graph.decay_all_edges.call_count == 2
        assert mock_graph.prune_weak_edges.call_count == 2


# =============================================================================
# Stale Edge Decay Tests
# =============================================================================


class TestStaleEdgeDecay:
    """Test the stale edge detection and extra decay logic."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_stale_threshold_calculated_from_config(
        self, mock_settings, service_with_graph, mock_graph, consolidation_config
    ):
        mock_settings.consolidation = consolidation_config

        await service_with_graph.consolidate()

        # Verify stale_before was calculated correctly (30 days ago)
        call_args = mock_graph.decay_stale_edges.call_args
        stale_before = call_args.kwargs["stale_before"]
        expected_stale = time.time() - (30 * 86400)
        assert abs(stale_before - expected_stale) < 2  # Within 2 seconds

        assert call_args.kwargs["decay_factor"] == 0.5

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_stale_decay_failure_is_non_fatal(self, mock_settings, service_with_graph, mock_graph, consolidation_config):
        mock_settings.consolidation = consolidation_config
        mock_graph.decay_stale_edges.side_effect = RuntimeError("timeout")

        result = await service_with_graph.consolidate()

        assert "stale_decay" in result["errors"][0]
        # Other phases still completed
        assert result["edges_decayed"] == 50
        assert result["edges_pruned"] == 5


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestDuplicateDetection:
    """Test the near-duplicate memory detection and merging."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_detects_and_merges_duplicates(self, mock_settings, service_with_graph, mock_storage, consolidation_config):
        mock_settings.consolidation = consolidation_config

        # Set up: two memories with high similarity
        original = _make_memory("hash_old", "Python is a language", tags=["python"], created_at=1000.0)
        duplicate = _make_memory("hash_new", "Python is a programming language", tags=["code"], created_at=2000.0)

        mock_storage.get_all_memories.side_effect = [
            [original, duplicate],  # First batch
            [],  # No more
        ]
        mock_storage.retrieve.side_effect = [
            # Search for original's content finds duplicate with high similarity
            [MemoryQueryResult(memory=duplicate, relevance_score=0.97, debug_info={})],
            # Search for duplicate's content — already seen, won't happen
        ]

        result = await service_with_graph.consolidate()

        assert result["duplicates_found"] == 1
        assert result["duplicates_merged"] == 1

        # Kept the older one (hash_old), deleted the newer (hash_new)
        mock_storage.delete.assert_called_with("hash_new")

        # Tags were merged: keeper got victim's tags
        call_args = mock_storage.update_memory_metadata.call_args
        assert call_args.args[0] == "hash_old"
        assert set(call_args.args[1]["tags"]) == {"python", "code"}
        assert call_args.kwargs["preserve_timestamps"] is True

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_keeps_older_memory_on_merge(self, mock_settings, service_without_graph, mock_storage, consolidation_config):
        mock_settings.consolidation = consolidation_config

        older = _make_memory("hash_a", "content A", created_at=100.0)
        newer = _make_memory("hash_b", "content B", created_at=200.0)

        mock_storage.get_all_memories.side_effect = [
            [newer],  # We iterate newer first
            [],
        ]
        mock_storage.retrieve.return_value = [MemoryQueryResult(memory=older, relevance_score=0.98, debug_info={})]

        result = await service_without_graph.consolidate()

        # Should keep older (hash_a), delete newer (hash_b)
        mock_storage.delete.assert_called_with("hash_b")
        assert result["duplicates_merged"] == 1

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_skips_self_in_duplicate_search(
        self, mock_settings, service_without_graph, mock_storage, consolidation_config
    ):
        mock_settings.consolidation = consolidation_config

        memory = _make_memory("hash_1", "unique content")

        mock_storage.get_all_memories.side_effect = [
            [memory],
            [],
        ]
        # Search returns only the memory itself
        mock_storage.retrieve.return_value = [MemoryQueryResult(memory=memory, relevance_score=1.0, debug_info={})]

        result = await service_without_graph.consolidate()

        assert result["duplicates_found"] == 0
        assert result["duplicates_merged"] == 0
        mock_storage.delete.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_respects_max_duplicates_limit(self, mock_settings, service_without_graph, mock_storage, consolidation_config):
        consolidation_config.max_duplicates_per_run = 2
        mock_settings.consolidation = consolidation_config

        # Create 5 memories that are all duplicates of each other
        memories = [_make_memory(f"hash_{i}", f"same content {i}", created_at=float(i)) for i in range(5)]
        mock_storage.get_all_memories.side_effect = [memories, []]

        # Each search finds the "original" (hash_0) as duplicate
        mock_storage.retrieve.return_value = [MemoryQueryResult(memory=memories[0], relevance_score=0.99, debug_info={})]

        result = await service_without_graph.consolidate()

        # Should stop at max_duplicates_per_run = 2
        assert result["duplicates_found"] <= 2

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_no_duplicates_when_storage_empty(
        self, mock_settings, service_without_graph, mock_storage, consolidation_config
    ):
        mock_settings.consolidation = consolidation_config
        mock_storage.get_all_memories.return_value = []

        result = await service_without_graph.consolidate()

        assert result["duplicates_found"] == 0
        assert result["duplicates_merged"] == 0

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_duplicate_merge_failure_is_non_fatal(
        self, mock_settings, service_without_graph, mock_storage, consolidation_config
    ):
        mock_settings.consolidation = consolidation_config

        original = _make_memory("hash_1", "content", created_at=1.0)
        duplicate = _make_memory("hash_2", "content copy", created_at=2.0)

        mock_storage.get_all_memories.side_effect = [[original, duplicate], []]
        mock_storage.retrieve.return_value = [MemoryQueryResult(memory=duplicate, relevance_score=0.99, debug_info={})]
        mock_storage.delete.side_effect = RuntimeError("storage error")

        result = await service_without_graph.consolidate()

        # Found but couldn't merge — should not crash
        assert result["duplicates_found"] == 1
        assert result["duplicates_merged"] == 0


# =============================================================================
# Graph Client Consolidation Method Tests
# =============================================================================


class TestGraphClientConsolidation:
    """Test the GraphClient bulk consolidation queries."""

    @pytest.mark.asyncio
    async def test_decay_all_edges(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[42]]
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        count = await client.decay_all_edges(decay_factor=0.9, limit=1000)

        assert count == 42
        call_args = mock_graph.query.call_args
        assert "e.weight * $decay" in call_args.args[0]
        assert call_args.kwargs["params"]["decay"] == 0.9
        assert call_args.kwargs["params"]["lim"] == 1000

    @pytest.mark.asyncio
    async def test_decay_stale_edges(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[15]]
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        count = await client.decay_stale_edges(stale_before=1000.0, decay_factor=0.5, limit=500)

        assert count == 15
        call_args = mock_graph.query.call_args
        assert "last_co_access < $ts" in call_args.args[0]
        assert call_args.kwargs["params"]["ts"] == 1000.0
        assert call_args.kwargs["params"]["decay"] == 0.5

    @pytest.mark.asyncio
    async def test_prune_weak_edges(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[7]]
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        count = await client.prune_weak_edges(threshold=0.05, limit=2000)

        assert count == 7
        call_args = mock_graph.query.call_args
        assert "weight < $thresh" in call_args.args[0]
        assert "DELETE e" in call_args.args[0]
        assert call_args.kwargs["params"]["thresh"] == 0.05

    @pytest.mark.asyncio
    async def test_get_orphan_nodes(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [["hash_a"], ["hash_b"]]
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        orphans = await client.get_orphan_nodes(limit=100)

        assert orphans == ["hash_a", "hash_b"]
        call_args = mock_graph.query.call_args
        assert "NOT (m)-[:HEBBIAN]-()" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_decay_empty_graph_returns_zero(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[0]]
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        count = await client.decay_all_edges(0.9)
        assert count == 0

    @pytest.mark.asyncio
    async def test_prune_empty_result_set(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient()
        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        client._graph = mock_graph
        client._initialized = True

        count = await client.prune_weak_edges(0.05)
        assert count == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConsolidationConfig:
    """Test ConsolidationSettings validation."""

    def test_default_config_values(self):
        from mcp_memory_service.config import ConsolidationSettings

        config = ConsolidationSettings()
        assert config.decay_factor == 0.9
        assert config.prune_threshold == 0.05
        assert config.stale_edge_days == 30
        assert config.stale_decay_factor == 0.5
        assert config.max_edges_per_run == 10000
        assert config.duplicate_similarity_threshold == 0.95
        assert config.max_duplicates_per_run == 100

    def test_env_prefix(self):
        from mcp_memory_service.config import ConsolidationSettings

        config = ConsolidationSettings()
        assert config.model_config.get("env_prefix") == "MCP_CONSOLIDATION_"

    def test_decay_factor_bounds(self):
        from pydantic import ValidationError

        from mcp_memory_service.config import ConsolidationSettings

        with pytest.raises(ValidationError):
            ConsolidationSettings(decay_factor=0.0)  # Below minimum

        with pytest.raises(ValidationError):
            ConsolidationSettings(decay_factor=1.0)  # Above maximum
