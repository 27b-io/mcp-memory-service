"""
Tests for Hebbian-weighted search boosting (issue #72).

After vector search returns top-K, query FalkorDB for Hebbian edges
between result nodes. Apply configurable boost: score * (1 + max_weight * BOOST_FACTOR).
Zero boost when no edges exist (graceful degradation).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.graph.client import GraphClient
from mcp_memory_service.models.memory import Memory, MemoryQueryResult

# ── Helpers ─────────────────────────────────────────────────────────


def _make_graph_client(mock_graph) -> GraphClient:
    """Create a GraphClient with a mocked FalkorDB graph."""
    client = GraphClient.__new__(GraphClient)
    client._graph = mock_graph
    client._initialized = True
    return client


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    return Memory(content=content, content_hash=content_hash)


def _make_query_result(content_hash: str, score: float) -> MemoryQueryResult:
    return MemoryQueryResult(memory=_make_memory(content_hash), relevance_score=score)


# ── GraphClient.hebbian_boosts_within() ─────────────────────────────


class TestHebbianBoostsWithin:
    """Tests for GraphClient.hebbian_boosts_within()."""

    @pytest.mark.asyncio
    async def test_empty_hashes_returns_empty(self):
        mock_graph = AsyncMock()
        client = _make_graph_client(mock_graph)

        result = await client.hebbian_boosts_within([])

        assert result == {}
        mock_graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_hash_returns_empty(self):
        """A single node can't have edges to other results."""
        mock_graph = AsyncMock()
        client = _make_graph_client(mock_graph)

        result = await client.hebbian_boosts_within(["hash_a"])

        assert result == {}
        mock_graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_max_weight_per_node(self):
        """For each node, return max Hebbian weight to any other result node."""
        mock_result = MagicMock()
        mock_result.result_set = [
            # hash_a has outgoing edges to hash_b (0.8) and hash_c (0.3)
            ["hash_a", 0.8],
            # hash_b has outgoing edge to hash_a (0.6)
            ["hash_b", 0.6],
        ]
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        result = await client.hebbian_boosts_within(["hash_a", "hash_b", "hash_c"])

        assert result["hash_a"] == pytest.approx(0.8)
        assert result["hash_b"] == pytest.approx(0.6)
        assert "hash_c" not in result  # No outgoing edges

    @pytest.mark.asyncio
    async def test_no_edges_returns_empty(self):
        """Nodes with no mutual Hebbian edges return empty dict."""
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        result = await client.hebbian_boosts_within(["hash_a", "hash_b"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_query_constrains_both_endpoints(self):
        """Cypher query must require both endpoints are in the hash set."""
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph = AsyncMock()
        mock_graph.query.return_value = mock_result

        client = _make_graph_client(mock_graph)
        await client.hebbian_boosts_within(["hash_a", "hash_b"])

        query = mock_graph.query.call_args[0][0]
        # Both endpoints constrained to result set
        assert "a.content_hash IN $hashes" in query
        assert "b.content_hash IN $hashes" in query


# ── Config ──────────────────────────────────────────────────────────


class TestHebbianBoostConfig:
    """Tests for MCP_FALKORDB_HEBBIAN_BOOST config field."""

    def test_default_value(self):
        from mcp_memory_service.config import FalkorDBSettings

        s = FalkorDBSettings()
        assert s.hebbian_boost == pytest.approx(0.15)

    def test_configurable_via_env(self, monkeypatch):
        from mcp_memory_service.config import FalkorDBSettings

        monkeypatch.setenv("MCP_FALKORDB_HEBBIAN_BOOST", "0.25")
        s = FalkorDBSettings()
        assert s.hebbian_boost == pytest.approx(0.25)

    def test_zero_disables(self):
        from mcp_memory_service.config import FalkorDBSettings

        s = FalkorDBSettings(hebbian_boost=0.0)
        assert s.hebbian_boost == 0.0


# ── MemoryService integration ──────────────────────────────────────


class TestHebbianBoostIntegration:
    """Tests for Hebbian boost in MemoryService retrieval pipeline."""

    def _mock_settings(self, mock_settings, hebbian_boost=0.15):
        """Configure mock settings with Hebbian boost enabled."""
        mock_falkordb = MagicMock()
        mock_falkordb.spreading_activation_boost = 0.0  # Disable spreading to isolate
        mock_falkordb.hebbian_boost = hebbian_boost
        mock_settings.falkordb = mock_falkordb
        mock_settings.hybrid_search = MagicMock(recency_decay=0)
        mock_settings.salience = MagicMock(enabled=False)
        mock_settings.spaced_repetition = MagicMock(enabled=False)
        mock_settings.encoding_context = MagicMock(enabled=False)
        return mock_settings

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_hebbian_boost_applied_to_results(self, mock_settings):
        """Results with Hebbian edges to other results get boosted."""
        from mcp_memory_service.services.memory_service import MemoryService

        self._mock_settings(mock_settings, hebbian_boost=0.15)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=3)
        mock_storage.retrieve = AsyncMock(
            return_value=[
                _make_query_result("hash_a", 0.9),
                _make_query_result("hash_b", 0.8),
                _make_query_result("hash_c", 0.7),
            ]
        )

        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={})
        # hash_a and hash_b have mutual Hebbian edges
        mock_graph.hebbian_boosts_within = AsyncMock(
            return_value={
                "hash_a": 0.8,  # Strong co-access with hash_b
                "hash_b": 0.6,  # Co-access with hash_a
            }
        )

        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        # hash_a: 0.9 * (1 + 0.8 * 0.15) = 0.9 * 1.12 = 1.008
        hash_a = next(m for m in memories if m["content_hash"] == "hash_a")
        assert hash_a["similarity_score"] == pytest.approx(0.9 * (1 + 0.8 * 0.15))
        assert hash_a["hebbian_boost"] == pytest.approx(0.8)

        # hash_b: 0.8 * (1 + 0.6 * 0.15) = 0.8 * 1.09 = 0.872
        hash_b = next(m for m in memories if m["content_hash"] == "hash_b")
        assert hash_b["similarity_score"] == pytest.approx(0.8 * (1 + 0.6 * 0.15))

        # hash_c: no boost, original score
        hash_c = next(m for m in memories if m["content_hash"] == "hash_c")
        assert hash_c["similarity_score"] == pytest.approx(0.7)
        assert "hebbian_boost" not in hash_c

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_hebbian_boost_can_reorder(self, mock_settings):
        """Strong Hebbian connections can reorder results."""
        from mcp_memory_service.services.memory_service import MemoryService

        self._mock_settings(mock_settings, hebbian_boost=0.15)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=2)
        mock_storage.retrieve = AsyncMock(
            return_value=[
                _make_query_result("hash_a", 0.80),
                _make_query_result("hash_b", 0.78),
            ]
        )

        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={})
        mock_graph.hebbian_boosts_within = AsyncMock(
            return_value={
                "hash_b": 1.0,  # Max Hebbian weight
                # hash_a: no edges
            }
        )

        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        # hash_b: 0.78 * (1 + 1.0 * 0.15) = 0.78 * 1.15 = 0.897
        # hash_a: 0.80 (no boost)
        # hash_b should now be first
        assert memories[0]["content_hash"] == "hash_b"

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_zero_boost_skips_query(self, mock_settings):
        """When hebbian_boost=0, graph is not queried for Hebbian edges."""
        from mcp_memory_service.services.memory_service import MemoryService

        self._mock_settings(mock_settings, hebbian_boost=0.0)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=1)
        mock_storage.retrieve = AsyncMock(
            return_value=[_make_query_result("hash_a", 0.9)]
        )

        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={})
        service = MemoryService(mock_storage, mock_graph, write_queue=None)

        await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        mock_graph.hebbian_boosts_within.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_no_graph_client_graceful(self, mock_settings):
        """Without graph client, no Hebbian boost and no errors."""
        from mcp_memory_service.services.memory_service import MemoryService

        self._mock_settings(mock_settings, hebbian_boost=0.15)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=2)
        mock_storage.retrieve = AsyncMock(
            return_value=[
                _make_query_result("hash_a", 0.9),
                _make_query_result("hash_b", 0.8),
            ]
        )

        service = MemoryService(mock_storage, graph_client=None, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        assert memories[0]["similarity_score"] == pytest.approx(0.9)
        assert "hebbian_boost" not in memories[0]

    @pytest.mark.asyncio
    @patch("mcp_memory_service.services.memory_service.settings")
    async def test_graph_failure_non_fatal(self, mock_settings):
        """Hebbian boost query failure doesn't break retrieval."""
        from mcp_memory_service.services.memory_service import MemoryService

        self._mock_settings(mock_settings, hebbian_boost=0.15)

        mock_storage = AsyncMock()
        mock_storage.count_semantic_search = AsyncMock(return_value=1)
        mock_storage.retrieve = AsyncMock(
            return_value=[_make_query_result("hash_a", 0.9)]
        )

        mock_graph = AsyncMock()
        mock_graph.spreading_activation = AsyncMock(return_value={})
        mock_graph.hebbian_boosts_within = AsyncMock(
            side_effect=ConnectionError("FalkorDB down")
        )

        service = MemoryService(mock_storage, mock_graph, write_queue=None)
        result = await service._retrieve_vector_only(
            query="test", page=1, page_size=10,
            tags=None, memory_type=None, min_similarity=None,
        )

        memories = result["memories"]
        assert len(memories) == 1
        assert memories[0]["similarity_score"] == pytest.approx(0.9)
