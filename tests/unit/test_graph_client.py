"""
Unit tests for GraphClient.

Tests the FalkorDB graph client with mocked FalkorDB/Redis connections.
Validates node operations, read queries, and schema initialization.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_pool():
    """Create a mock Redis connection pool."""
    pool = MagicMock()
    pool.aclose = AsyncMock()
    return pool


@pytest.fixture
def mock_graph():
    """Create a mock FalkorDB graph."""
    graph = AsyncMock()
    return graph


@pytest.fixture
def mock_db(mock_graph):
    """Create a mock FalkorDB instance."""
    db = MagicMock()
    db.select_graph.return_value = mock_graph
    return db


class TestGraphClientInit:
    """Test GraphClient initialization and schema application."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_initialize_creates_pool_and_applies_schema(
        self, mock_falkordb_cls, mock_pool_cls
    ):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool_instance = MagicMock()
        mock_pool_instance.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool_instance

        mock_graph_instance = AsyncMock()
        mock_db_instance = MagicMock()
        mock_db_instance.select_graph.return_value = mock_graph_instance
        mock_falkordb_cls.return_value = mock_db_instance

        client = GraphClient(
            host="testhost", port=6380, graph_name="test_graph", max_connections=8
        )
        await client.initialize()

        # Pool created with correct params
        mock_pool_cls.assert_called_once_with(
            host="testhost",
            port=6380,
            password=None,
            max_connections=8,
            timeout=None,
            decode_responses=True,
        )

        # Graph selected
        mock_db_instance.select_graph.assert_called_once_with("test_graph")

        # Schema applied (2 index statements)
        assert mock_graph_instance.query.call_count == 2

        # Idempotent: second call is no-op
        await client.initialize()
        assert mock_graph_instance.query.call_count == 2  # No additional calls

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_initialize_handles_existing_index(
        self, mock_falkordb_cls, mock_pool_cls
    ):
        """Schema statements that fail with 'already indexed' are silently ignored."""
        from mcp_memory_service.graph.client import GraphClient

        mock_pool_cls.return_value = MagicMock(aclose=AsyncMock())
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = Exception("Index already indexed")
        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()  # Should not raise


class TestGraphClientNodeOps:
    """Test node create/delete operations."""

    @pytest.mark.asyncio
    async def test_ensure_memory_node(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        await client.ensure_memory_node("abc123", 1700000000.0)

        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MERGE" in call_args[0][0]
        assert call_args[1]["params"]["hash"] == "abc123"
        assert call_args[1]["params"]["ts"] == 1700000000.0

    @pytest.mark.asyncio
    async def test_delete_memory_node(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        await client.delete_memory_node("abc123")

        mock_graph.query.assert_called_once()
        assert "DETACH DELETE" in mock_graph.query.call_args[0][0]


class TestGraphClientReads:
    """Test read operations."""

    @pytest.mark.asyncio
    async def test_get_neighbors(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        # Mock query result
        mock_result = MagicMock()
        mock_result.result_set = [
            ["hash_b", 0.85, 1],
            ["hash_c", 0.6, 2],
        ]
        mock_graph.query.return_value = mock_result

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        neighbors = await client.get_neighbors("hash_a", max_hops=2, min_weight=0.1)

        assert len(neighbors) == 2
        assert neighbors[0]["content_hash"] == "hash_b"
        assert neighbors[0]["weight"] == 0.85
        assert neighbors[0]["hops"] == 1
        assert neighbors[1]["content_hash"] == "hash_c"

    @pytest.mark.asyncio
    async def test_get_neighbors_caps_hops_at_3(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        await client.get_neighbors("hash_a", max_hops=10)

        # Verify the query uses *1..3, not *1..10
        query = mock_graph.query.call_args[0][0]
        assert "*1..3" in query

    @pytest.mark.asyncio
    async def test_get_edge_found(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        mock_result = MagicMock()
        mock_result.result_set = [[0.75, 5, 1700000000.0]]
        mock_graph.query.return_value = mock_result

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        edge = await client.get_edge("src", "dst")
        assert edge is not None
        assert edge["weight"] == 0.75
        assert edge["co_access_count"] == 5

    @pytest.mark.asyncio
    async def test_get_edge_not_found(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True

        edge = await client.get_edge("src", "dst")
        assert edge is None

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        node_result = MagicMock()
        node_result.result_set = [[42]]
        edge_result = MagicMock()
        edge_result.result_set = [[100]]
        mock_graph.query.side_effect = [node_result, edge_result]

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True
        client.graph_name = "test_graph"

        stats = await client.get_graph_stats()
        assert stats["node_count"] == 42
        assert stats["edge_count"] == 100
        assert stats["status"] == "operational"

    @pytest.mark.asyncio
    async def test_get_graph_stats_error(self, mock_graph):
        from mcp_memory_service.graph.client import GraphClient

        mock_graph.query.side_effect = Exception("connection refused")

        client = GraphClient.__new__(GraphClient)
        client._graph = mock_graph
        client._initialized = True
        client.graph_name = "test_graph"

        stats = await client.get_graph_stats()
        assert stats["status"] == "error"


class TestGraphClientClose:
    """Test connection cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()

        client = GraphClient.__new__(GraphClient)
        client._pool = mock_pool
        client._db = MagicMock()
        client._graph = MagicMock()
        client._initialized = True

        await client.close()

        mock_pool.aclose.assert_called_once()
        assert client._pool is None
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self):
        from mcp_memory_service.graph.client import GraphClient

        client = GraphClient.__new__(GraphClient)
        client._pool = None
        client._db = None
        client._graph = None
        client._initialized = False

        await client.close()  # Should not raise
