"""
Tests for spreading activation (multi-hop graph traversal with decay).

Tests the GraphClient.spreading_activation() method and its integration
into MemoryService retrieval paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.graph.client import GraphClient
from mcp_memory_service.models.memory import Memory, MemoryQueryResult

# ── GraphClient.spreading_activation() ───────────────────────────────


def _make_graph_client(mock_graph) -> GraphClient:
    """Helper to create a GraphClient with a mocked graph."""
    client = GraphClient.__new__(GraphClient)
    client._graph = mock_graph
    client._initialized = True
    return client


class TestSpreadingActivation:
    """Tests for GraphClient.spreading_activation()."""

    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self):
        mock_graph = AsyncMock()
        client = _make_graph_client(mock_graph)

        result = await client.spreading_activation(seed_hashes=[])

        assert result == {}
        mock_graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_seed_single_hop(self):
        """One seed, one neighbor at 1 hop with weight 0.8. decay=0.5 → 0.8*0.5=0.4"""
        mock_result = MagicMock()
        mock_result.result_set = [
            ["neighbor_a", 0.8, 1],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["seed_1"], decay_factor=0.5
        )

        assert "neighbor_a" in activations
        assert activations["neighbor_a"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_two_hop_decay(self):
        """Neighbor at 2 hops, path_weight=0.64 (0.8*0.8), decay=0.5 → 0.64*0.25=0.16"""
        mock_result = MagicMock()
        mock_result.result_set = [
            ["neighbor_b", 0.64, 2],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["seed_1"], decay_factor=0.5
        )

        assert activations["neighbor_b"] == pytest.approx(0.16)

    @pytest.mark.asyncio
    async def test_multi_path_takes_max(self):
        """Same destination via two paths: take max activation."""
        mock_result = MagicMock()
        mock_result.result_set = [
            # Path 1: 1 hop, weight 0.3, activation = 0.3 * 0.5 = 0.15
            ["neighbor_c", 0.3, 1],
            # Path 2: 1 hop via different seed, weight 0.9, activation = 0.9 * 0.5 = 0.45
            ["neighbor_c", 0.9, 1],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["seed_1", "seed_2"], decay_factor=0.5
        )

        # Should take the max: 0.45
        assert activations["neighbor_c"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_min_activation_filters(self):
        """Activations below min_activation are excluded."""
        mock_result = MagicMock()
        mock_result.result_set = [
            # activation = 0.1 * 0.5^2 = 0.025 (above 0.01)
            ["above", 0.1, 2],
            # activation = 0.02 * 0.5^2 = 0.005 (below 0.01)
            ["below", 0.02, 2],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["seed_1"], decay_factor=0.5, min_activation=0.01
        )

        assert "above" in activations
        assert "below" not in activations

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        """Only top N results returned when limit is set."""
        mock_result = MagicMock()
        mock_result.result_set = [
            ["n1", 0.9, 1],
            ["n2", 0.8, 1],
            ["n3", 0.7, 1],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["seed_1"], decay_factor=0.5, limit=2
        )

        assert len(activations) == 2
        # Top 2 by activation
        assert "n1" in activations
        assert "n2" in activations

    @pytest.mark.asyncio
    async def test_max_hops_capped_at_3(self):
        """max_hops is capped at 3 regardless of input."""
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        await client.spreading_activation(
            seed_hashes=["seed_1"], max_hops=10
        )

        query = mock_graph.query.call_args[0][0]
        assert "*1..3" in query

    @pytest.mark.asyncio
    async def test_seeds_excluded_from_results(self):
        """Seeds should not appear in their own activation results."""
        mock_result = MagicMock()
        mock_result.result_set = []  # Query already filters via NOT IN $seeds
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        await client.spreading_activation(seed_hashes=["seed_1"])

        query = mock_graph.query.call_args[0][0]
        assert "NOT dst.content_hash IN $seeds" in query

    @pytest.mark.asyncio
    async def test_no_neighbors_returns_empty(self):
        """Seeds with no graph connections return empty dict."""
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        activations = await client.spreading_activation(
            seed_hashes=["orphan_hash"]
        )

        assert activations == {}


# ── MemoryService integration ────────────────────────────────────────


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    """Helper to create a Memory with a given hash."""
    return Memory(content=content, content_hash=content_hash)


def _make_query_result(content_hash: str, score: float) -> MemoryQueryResult:
    """Helper to create a MemoryQueryResult."""
    return MemoryQueryResult(
        memory=_make_memory(content_hash), relevance_score=score
    )


class TestMemoryServiceGraphBoost:
    """Tests for graph boost integration in MemoryService retrieval."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_vector_results_boosted_by_graph(self, mock_settings):
        """Vector search results get boosted by spreading activation scores."""
        from mcp_memory_service.services.memory_service import MemoryService

        # Configure settings
        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.2
        mock_falkordb.spreading_activation_max_hops = 2
        mock_falkordb.spreading_activation_decay = 0.5
        mock_falkordb.spreading_activation_min_activation = 0.01
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=3)
        mock_storage.retrieve = AsyncMock(return_value=[
            _make_query_result("hash_a", 0.9),
            _make_query_result("hash_b", 0.8),
            _make_query_result("hash_c", 0.7),
        ])

        # Mock graph: hash_b is activated by hash_a's neighbors
        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={
            "hash_b": 0.4,  # hash_b gets boosted
        })

        mock_write_queue = None

        service = MemoryService(mock_storage, mock_graph, mock_write_queue)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        # hash_b should be boosted: 0.8 + 0.2 * 0.4 = 0.88
        hash_b_result = next(m for m in memories if m["content_hash"] == "hash_b")
        assert hash_b_result["similarity_score"] == pytest.approx(0.88)
        assert hash_b_result["graph_boost"] == pytest.approx(0.4)

        # hash_a should still be first (0.9, no boost)
        assert memories[0]["content_hash"] == "hash_a"

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_no_graph_means_no_boost(self, mock_settings):
        """Without a graph client, no boost is applied."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.2
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=2)
        mock_storage.retrieve = AsyncMock(return_value=[
            _make_query_result("hash_a", 0.9),
            _make_query_result("hash_b", 0.8),
        ])

        # No graph client
        service = MemoryService(mock_storage, graph_client=None, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        # Scores unchanged
        assert memories[0]["similarity_score"] == pytest.approx(0.9)
        assert memories[1]["similarity_score"] == pytest.approx(0.8)
        # No graph_boost key
        assert "graph_boost" not in memories[0]

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_graph_boost_reorders_results(self, mock_settings):
        """A strong graph boost can reorder results."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.5  # Strong boost
        mock_falkordb.spreading_activation_max_hops = 2
        mock_falkordb.spreading_activation_decay = 0.5
        mock_falkordb.spreading_activation_min_activation = 0.01
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=2)
        mock_storage.retrieve = AsyncMock(return_value=[
            _make_query_result("hash_a", 0.7),
            _make_query_result("hash_b", 0.6),
        ])

        # hash_b gets a strong activation
        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={
            "hash_b": 0.8,  # 0.6 + 0.5*0.8 = 1.0
        })

        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        # hash_b (1.0) now beats hash_a (0.7)
        assert memories[0]["content_hash"] == "hash_b"
        assert memories[0]["similarity_score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_graph_failure_is_non_fatal(self, mock_settings):
        """Graph layer errors don't break retrieval."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.2
        mock_falkordb.spreading_activation_max_hops = 2
        mock_falkordb.spreading_activation_decay = 0.5
        mock_falkordb.spreading_activation_min_activation = 0.01
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=1)
        mock_storage.retrieve = AsyncMock(return_value=[
            _make_query_result("hash_a", 0.9),
        ])

        # Graph raises an error
        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(
            side_effect=ConnectionError("FalkorDB down")
        )

        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        # Still returns results, unboosted
        memories = result["memories"]
        assert len(memories) == 1
        assert memories[0]["similarity_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_zero_boost_weight_skips_graph(self, mock_settings):
        """When boost weight is 0, graph is not queried."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.0  # Disabled
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=1)
        mock_storage.retrieve = AsyncMock(return_value=[
            _make_query_result("hash_a", 0.9),
        ])

        mock_graph = AsyncMock()
        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        # Graph should NOT be queried when boost=0
        mock_graph.spreading_activation.assert_not_called()
