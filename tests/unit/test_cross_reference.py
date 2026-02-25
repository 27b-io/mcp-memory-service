"""
Tests for automatic memory cross-referencing.

Covers:
- _create_cross_references creates RELATES_TO edges for similar memories
- Bidirectional edge creation (A→B and B→A)
- Contradiction hashes are skipped
- The new memory itself is skipped
- Graph-disabled path returns early
- Config-disabled path returns early
- create_relation with RELATES_TO creates reverse edge when bidirectional=True
- create_relation with non-RELATES_TO types does NOT create reverse edge
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


# =============================================================================
# Fixtures
# =============================================================================


def _make_memory(hash_val: str, content: str = "content") -> Memory:
    return Memory(
        content=content,
        content_hash=hash_val,
        tags=[],
        created_at=1700000000.0,
        updated_at=1700000000.0,
    )


def _make_result(hash_val: str, similarity: float, content: str = "content") -> MemoryQueryResult:
    mem = _make_memory(hash_val, content)
    return MemoryQueryResult(memory=mem, relevance_score=similarity)


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 10000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    return storage


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    # create_typed_edge returns True by default (edge created)
    graph.create_typed_edge = AsyncMock(return_value=True)
    return graph


@pytest.fixture
def service_with_graph(mock_storage, mock_graph):
    svc = MemoryService(storage=mock_storage)
    svc._graph = mock_graph
    return svc


@pytest.fixture
def service_no_graph(mock_storage):
    return MemoryService(storage=mock_storage)


# =============================================================================
# _create_cross_references — core behavior
# =============================================================================


class TestCreateCrossReferences:
    @pytest.mark.asyncio
    async def test_no_graph_returns_early(self, service_no_graph, mock_storage):
        """With no graph, _create_cross_references is a no-op."""
        # storage.retrieve should NOT be called
        mock_storage.retrieve = AsyncMock(return_value=[])
        await service_no_graph._create_cross_references("newhash", "content", set())
        mock_storage.retrieve.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_disabled_config_returns_early(self, mock_settings, service_with_graph, mock_storage):
        """With cross_reference.enabled=False, no storage or graph calls."""
        mock_settings.cross_reference.enabled = False
        mock_storage.retrieve = AsyncMock(return_value=[])
        await service_with_graph._create_cross_references("newhash", "content", set())
        mock_storage.retrieve.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_creates_relates_to_for_similar_memories(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """Creates RELATES_TO edge for each similar result above threshold."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = False

        similar = [_make_result("existhash1", 0.90), _make_result("existhash2", 0.87)]
        mock_storage.retrieve = AsyncMock(return_value=similar)

        await service_with_graph._create_cross_references("newhash", "content", set())

        assert mock_graph.create_typed_edge.call_count == 2
        calls = [c.kwargs for c in mock_graph.create_typed_edge.call_args_list]
        assert {"source_hash": "newhash", "target_hash": "existhash1", "relation_type": "RELATES_TO"} in calls
        assert {"source_hash": "newhash", "target_hash": "existhash2", "relation_type": "RELATES_TO"} in calls

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_bidirectional_creates_reverse_edges(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """When bidirectional=True, also creates reverse RELATES_TO edge."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = True

        similar = [_make_result("existhash1", 0.92)]
        mock_storage.retrieve = AsyncMock(return_value=similar)

        await service_with_graph._create_cross_references("newhash", "content", set())

        # Should create 2 edges: newhash→existhash1 and existhash1→newhash
        assert mock_graph.create_typed_edge.call_count == 2
        calls = [c.kwargs for c in mock_graph.create_typed_edge.call_args_list]
        assert {"source_hash": "newhash", "target_hash": "existhash1", "relation_type": "RELATES_TO"} in calls
        assert {"source_hash": "existhash1", "target_hash": "newhash", "relation_type": "RELATES_TO"} in calls

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_skips_contradiction_hashes(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """Memories already flagged as contradictions are not cross-referenced."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = False

        # existhash1 is a contradiction, existhash2 is a regular similar memory
        similar = [_make_result("existhash1", 0.92), _make_result("existhash2", 0.88)]
        mock_storage.retrieve = AsyncMock(return_value=similar)

        await service_with_graph._create_cross_references("newhash", "content", {"existhash1"})

        # Only existhash2 gets a RELATES_TO edge
        assert mock_graph.create_typed_edge.call_count == 1
        calls = [c.kwargs for c in mock_graph.create_typed_edge.call_args_list]
        assert {"source_hash": "newhash", "target_hash": "existhash2", "relation_type": "RELATES_TO"} in calls

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_skips_self_hash(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """The new memory's own hash is never linked to itself."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = False

        # Retrieve returns the new memory itself (exact duplicate scenario)
        similar = [_make_result("newhash", 1.0)]
        mock_storage.retrieve = AsyncMock(return_value=similar)

        await service_with_graph._create_cross_references("newhash", "content", set())

        mock_graph.create_typed_edge.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_no_similar_memories_no_edges(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """No similar memories = no edges created."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = True

        mock_storage.retrieve = AsyncMock(return_value=[])

        await service_with_graph._create_cross_references("newhash", "content", set())

        mock_graph.create_typed_edge.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_graph_edge_failure_is_non_fatal(self, mock_settings, service_with_graph, mock_storage, mock_graph):
        """Individual edge creation failure doesn't abort the rest."""
        mock_settings.cross_reference.enabled = True
        mock_settings.cross_reference.similarity_threshold = 0.85
        mock_settings.cross_reference.max_links = 5
        mock_settings.cross_reference.bidirectional = False

        similar = [_make_result("existhash1", 0.92), _make_result("existhash2", 0.88)]
        mock_storage.retrieve = AsyncMock(return_value=similar)
        # First call raises, second succeeds
        mock_graph.create_typed_edge = AsyncMock(side_effect=[Exception("graph down"), True])

        # Should not raise
        await service_with_graph._create_cross_references("newhash", "content", set())

        # Second edge should still be attempted
        assert mock_graph.create_typed_edge.call_count == 2


# =============================================================================
# create_relation — bidirectional RELATES_TO
# =============================================================================


class TestCreateRelationBidirectional:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_relates_to_creates_reverse_edge(self, mock_settings, service_with_graph, mock_graph):
        """create_relation with RELATES_TO creates the reverse edge when bidirectional=True."""
        mock_settings.cross_reference.bidirectional = True

        await service_with_graph.create_relation("hashA", "hashB", "RELATES_TO")

        assert mock_graph.create_typed_edge.call_count == 2
        calls = [c.kwargs for c in mock_graph.create_typed_edge.call_args_list]
        assert {"source_hash": "hashA", "target_hash": "hashB", "relation_type": "RELATES_TO"} in calls
        assert {"source_hash": "hashB", "target_hash": "hashA", "relation_type": "RELATES_TO"} in calls

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_relates_to_unidirectional_when_disabled(self, mock_settings, service_with_graph, mock_graph):
        """create_relation with RELATES_TO does NOT create reverse edge when bidirectional=False."""
        mock_settings.cross_reference.bidirectional = False

        await service_with_graph.create_relation("hashA", "hashB", "RELATES_TO")

        assert mock_graph.create_typed_edge.call_count == 1
        call = mock_graph.create_typed_edge.call_args.kwargs
        assert call == {"source_hash": "hashA", "target_hash": "hashB", "relation_type": "RELATES_TO"}

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_precedes_does_not_create_reverse_edge(self, mock_settings, service_with_graph, mock_graph):
        """PRECEDES is directional — no reverse edge ever created."""
        mock_settings.cross_reference.bidirectional = True

        await service_with_graph.create_relation("hashA", "hashB", "PRECEDES")

        assert mock_graph.create_typed_edge.call_count == 1

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_contradicts_does_not_create_reverse_edge(self, mock_settings, service_with_graph, mock_graph):
        """CONTRADICTS is directional — bidirectional setting does not apply."""
        mock_settings.cross_reference.bidirectional = True

        await service_with_graph.create_relation("hashA", "hashB", "CONTRADICTS")

        assert mock_graph.create_typed_edge.call_count == 1

    @pytest.mark.asyncio
    async def test_no_graph_returns_error(self, service_no_graph):
        """create_relation returns error dict when graph is disabled."""
        result = await service_no_graph.create_relation("hashA", "hashB", "RELATES_TO")
        assert result["success"] is False
        assert "graph" in result["error"].lower()


# =============================================================================
# CrossReferenceSettings config defaults
# =============================================================================


class TestCrossReferenceSettings:
    def test_default_values(self):
        from mcp_memory_service.config import CrossReferenceSettings

        cfg = CrossReferenceSettings()
        assert cfg.enabled is True
        assert cfg.similarity_threshold == 0.85
        assert cfg.max_links == 5
        assert cfg.bidirectional is True

    def test_threshold_bounds(self):
        from mcp_memory_service.config import CrossReferenceSettings

        cfg = CrossReferenceSettings(similarity_threshold=0.5)
        assert cfg.similarity_threshold == 0.5

        cfg2 = CrossReferenceSettings(similarity_threshold=0.99)
        assert cfg2.similarity_threshold == 0.99
