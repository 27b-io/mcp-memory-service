"""
Unit tests for contradiction resolution: memory_supersede and memory_contradictions.

Covers:
- supersede_memory: validates inputs, marks metadata, creates graph edge
- get_contradictions_dashboard: lists CONTRADICTS pairs with content previews
- _filter_superseded: filters superseded memories from results
- _get_contradiction_warnings: enriches results with contradiction info
- GraphClient.get_contradictions_for_hashes and get_all_contradictions
- GraphClient.create_typed_edge with confidence parameter
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.update_memory_metadata.return_value = (True, "Updated")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    return storage


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    graph.ensure_memory_node = AsyncMock()
    graph.create_typed_edge = AsyncMock(return_value=True)
    graph.get_contradictions_for_hashes = AsyncMock(return_value={})
    graph.get_all_contradictions = AsyncMock(return_value=[])
    return graph


@pytest.fixture
def memory_service(mock_storage, mock_graph):
    return MemoryService(storage=mock_storage, graph_client=mock_graph)


@pytest.fixture
def memory_service_no_graph(mock_storage):
    return MemoryService(storage=mock_storage, graph_client=None)


def make_memory(content_hash: str, content: str = "test content", metadata: dict | None = None) -> Memory:
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=[],
        metadata=metadata or {},
        created_at=1700000000.0,
        updated_at=1700000000.0,
    )


# =============================================================================
# supersede_memory tests
# =============================================================================


class TestSupersede:
    @pytest.mark.asyncio
    async def test_supersede_success(self, memory_service, mock_storage, mock_graph):
        old = make_memory("old_hash", "old content about Redis")
        new = make_memory("new_hash", "new content about Valkey replacing Redis")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new

        result = await memory_service.supersede_memory("old_hash", "new_hash", "Switched from Redis to Valkey")

        assert result["success"] is True
        assert result["superseded"] == "old_hash"
        assert result["superseded_by"] == "new_hash"
        assert result["reason"] == "Switched from Redis to Valkey"

    @pytest.mark.asyncio
    async def test_supersede_marks_metadata(self, memory_service, mock_storage):
        old = make_memory("old_hash", "old content")
        new = make_memory("new_hash", "new content")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new

        await memory_service.supersede_memory("old_hash", "new_hash", "outdated info")

        # Verify update_memory_metadata was called with superseded_by key
        mock_storage.update_memory_metadata.assert_called_once()
        call_args = mock_storage.update_memory_metadata.call_args
        assert call_args[0][0] == "old_hash"
        metadata_update = call_args[0][1]
        assert metadata_update["metadata"]["superseded_by"] == "new_hash"
        assert metadata_update["metadata"]["supersession_reason"] == "outdated info"

    @pytest.mark.asyncio
    async def test_supersede_creates_graph_edge(self, memory_service, mock_graph, mock_storage):
        old = make_memory("old_hash")
        new = make_memory("new_hash")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new

        await memory_service.supersede_memory("old_hash", "new_hash", "reason")

        mock_graph.create_typed_edge.assert_called_once_with(
            source_hash="new_hash",
            target_hash="old_hash",
            relation_type="SUPERSEDES",
        )

    @pytest.mark.asyncio
    async def test_supersede_same_hash_rejected(self, memory_service):
        result = await memory_service.supersede_memory("same_hash", "same_hash", "reason")
        assert result["success"] is False
        assert "different" in result["error"]

    @pytest.mark.asyncio
    async def test_supersede_old_not_found(self, memory_service, mock_storage):
        mock_storage.get_memory_by_hash.return_value = None

        result = await memory_service.supersede_memory("missing_hash", "new_hash", "reason")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_supersede_new_not_found(self, memory_service, mock_storage):
        old = make_memory("old_hash")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else None

        result = await memory_service.supersede_memory("old_hash", "missing_new", "reason")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_supersede_storage_failure(self, memory_service, mock_storage):
        old = make_memory("old_hash")
        new = make_memory("new_hash")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new
        mock_storage.update_memory_metadata.return_value = (False, "storage error")

        result = await memory_service.supersede_memory("old_hash", "new_hash", "reason")
        assert result["success"] is False
        assert "superseded" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_supersede_graph_failure_is_non_fatal(self, memory_service, mock_storage, mock_graph):
        """Graph edge creation failure should not fail the supersession."""
        old = make_memory("old_hash")
        new = make_memory("new_hash")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new
        mock_graph.create_typed_edge.side_effect = Exception("graph unavailable")

        result = await memory_service.supersede_memory("old_hash", "new_hash", "reason")
        # Metadata was updated even though graph failed
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_supersede_no_graph_layer(self, memory_service_no_graph, mock_storage):
        """Supersession works without graph layer (metadata-only)."""
        old = make_memory("old_hash")
        new = make_memory("new_hash")
        mock_storage.get_memory_by_hash.side_effect = lambda h: old if h == "old_hash" else new

        result = await memory_service_no_graph.supersede_memory("old_hash", "new_hash", "reason")
        assert result["success"] is True


# =============================================================================
# _filter_superseded tests
# =============================================================================


class TestFilterSuperseded:
    def test_filters_superseded_memories(self, memory_service):
        results = [
            {"content_hash": "a", "metadata": {"superseded_by": "b"}, "content": "old"},
            {"content_hash": "b", "metadata": {}, "content": "new"},
            {"content_hash": "c", "metadata": None, "content": "other"},
        ]
        filtered = memory_service._filter_superseded(results)
        assert len(filtered) == 2
        assert all(r["content_hash"] != "a" for r in filtered)

    def test_empty_results_unchanged(self, memory_service):
        assert memory_service._filter_superseded([]) == []

    def test_none_metadata_treated_as_not_superseded(self, memory_service):
        results = [{"content_hash": "x", "metadata": None, "content": "test"}]
        filtered = memory_service._filter_superseded(results)
        assert len(filtered) == 1

    def test_no_superseded_in_batch(self, memory_service):
        results = [
            {"content_hash": "a", "metadata": {"foo": "bar"}, "content": "ok"},
            {"content_hash": "b", "metadata": {}, "content": "ok"},
        ]
        assert memory_service._filter_superseded(results) == results


# =============================================================================
# get_contradictions_dashboard tests
# =============================================================================


class TestContradictionsDashboard:
    @pytest.mark.asyncio
    async def test_returns_pairs_with_content(self, memory_service, mock_graph, mock_storage):
        mock_graph.get_all_contradictions.return_value = [
            {"memory_a_hash": "hash_a", "memory_b_hash": "hash_b", "confidence": 0.74, "created_at": 1700000000.0}
        ]
        mem_a = make_memory("hash_a", "CacheKit uses Stripe for billing")
        mem_b = make_memory("hash_b", "CacheKit billing is lemon_squeezy")
        mock_storage.get_memory_by_hash.side_effect = lambda h: mem_a if h == "hash_a" else mem_b

        result = await memory_service.get_contradictions_dashboard(limit=20)

        assert result["success"] is True
        assert result["total"] == 1
        pair = result["pairs"][0]
        assert pair["memory_a_hash"] == "hash_a"
        assert pair["memory_b_hash"] == "hash_b"
        assert pair["confidence"] == 0.74
        assert "Stripe" in pair["memory_a_content"]
        assert "lemon_squeezy" in pair["memory_b_content"]

    @pytest.mark.asyncio
    async def test_marks_superseded_in_pairs(self, memory_service, mock_graph, mock_storage):
        mock_graph.get_all_contradictions.return_value = [
            {"memory_a_hash": "hash_a", "memory_b_hash": "hash_b", "confidence": 0.7, "created_at": 1700000000.0}
        ]
        mem_a = make_memory("hash_a", "old content", metadata={"superseded_by": "hash_b"})
        mem_b = make_memory("hash_b", "new content")
        mock_storage.get_memory_by_hash.side_effect = lambda h: mem_a if h == "hash_a" else mem_b

        result = await memory_service.get_contradictions_dashboard()

        pair = result["pairs"][0]
        assert pair["memory_a_superseded"] is True
        assert pair["memory_b_superseded"] is False

    @pytest.mark.asyncio
    async def test_no_graph_returns_error(self, memory_service_no_graph):
        result = await memory_service_no_graph.get_contradictions_dashboard()
        assert result["success"] is False
        assert "graph layer" in result["error"].lower()
        assert result["pairs"] == []

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_pairs(self, memory_service, mock_graph):
        mock_graph.get_all_contradictions.return_value = []
        result = await memory_service.get_contradictions_dashboard()
        assert result["success"] is True
        assert result["pairs"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_graph_failure_returns_error(self, memory_service, mock_graph):
        mock_graph.get_all_contradictions.side_effect = Exception("connection error")
        result = await memory_service.get_contradictions_dashboard()
        assert result["success"] is False
        assert "connection error" in result["error"]


# =============================================================================
# _get_contradiction_warnings tests
# =============================================================================


class TestContradictionWarnings:
    @pytest.mark.asyncio
    async def test_returns_enriched_warnings(self, memory_service, mock_graph, mock_storage):
        mock_graph.get_contradictions_for_hashes.return_value = {"hash_a": [{"contradicts_hash": "hash_x", "confidence": 0.8}]}
        mem_x = make_memory("hash_x", "contradicting content here")
        mock_storage.get_memory_by_hash.return_value = mem_x

        warnings = await memory_service._get_contradiction_warnings(["hash_a"])

        assert "hash_a" in warnings
        w = warnings["hash_a"][0]
        assert w["memory_id"] == "hash_x"
        assert w["content"] == "contradicting content here"
        assert w["edge_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_no_graph_returns_empty(self, memory_service_no_graph):
        result = await memory_service_no_graph._get_contradiction_warnings(["hash_a"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_hashes_returns_empty(self, memory_service, mock_graph):
        result = await memory_service._get_contradiction_warnings([])
        assert result == {}
        mock_graph.get_contradictions_for_hashes.assert_not_called()

    @pytest.mark.asyncio
    async def test_graph_failure_is_non_fatal(self, memory_service, mock_graph):
        mock_graph.get_contradictions_for_hashes.side_effect = Exception("graph down")
        result = await memory_service._get_contradiction_warnings(["hash_a"])
        assert result == {}


# =============================================================================
# GraphClient.create_typed_edge with confidence tests
# =============================================================================


class TestCreateTypedEdgeWithConfidence:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_confidence_stored_on_contradicts_edge(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[1]]
        mock_graph.query = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()

        await client.create_typed_edge("src_hash", "dst_hash", "CONTRADICTS", confidence=0.75)

        # Verify the query includes confidence param
        calls = [str(call) for call in mock_graph.query.call_args_list]
        confidence_call = [c for c in calls if "confidence" in c]
        assert len(confidence_call) >= 1

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_no_confidence_uses_original_query(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[1]]
        mock_graph.query = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()

        await client.create_typed_edge("src_hash", "dst_hash", "RELATES_TO")

        # Last call should not include confidence param
        last_call = mock_graph.query.call_args_list[-1]
        params = last_call[1].get("params", last_call[0][1] if len(last_call[0]) > 1 else {})
        assert "conf" not in params

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_supersedes_is_valid_relation_type(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        mock_graph = AsyncMock()
        mock_result = MagicMock()
        mock_result.result_set = [[1]]
        mock_graph.query = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()

        # Should not raise ValueError
        result = await client.create_typed_edge("src_hash", "dst_hash", "SUPERSEDES")
        assert result is True


# =============================================================================
# GraphClient.get_contradictions_for_hashes tests
# =============================================================================


class TestGetContradictionsForHashes:
    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_returns_empty_for_empty_hashes(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool
        mock_graph = AsyncMock()
        mock_graph.query = AsyncMock()
        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()
        mock_graph.query.reset_mock()  # Reset after schema initialization calls

        result = await client.get_contradictions_for_hashes([])
        assert result == {}
        mock_graph.query.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_maps_edges_to_both_endpoints(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        mock_result = MagicMock()
        mock_result.result_set = [["hash_a", "hash_b", 0.74]]
        mock_graph = AsyncMock()
        mock_graph.query = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()

        result = await client.get_contradictions_for_hashes(["hash_a", "hash_b"])

        # Both endpoints in our set should get mapped
        assert "hash_a" in result
        assert "hash_b" in result
        assert result["hash_a"][0]["contradicts_hash"] == "hash_b"
        assert result["hash_b"][0]["contradicts_hash"] == "hash_a"
        assert result["hash_a"][0]["confidence"] == 0.74

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.client.BlockingConnectionPool")
    @patch("mcp_memory_service.graph.client.FalkorDB")
    async def test_handles_null_confidence(self, mock_falkordb_cls, mock_pool_cls):
        from mcp_memory_service.graph.client import GraphClient

        mock_pool = MagicMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        mock_result = MagicMock()
        mock_result.result_set = [["hash_a", "hash_b", None]]  # Old edge without confidence
        mock_graph = AsyncMock()
        mock_graph.query = AsyncMock(return_value=mock_result)

        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkordb_cls.return_value = mock_db

        client = GraphClient()
        await client.initialize()

        result = await client.get_contradictions_for_hashes(["hash_a"])
        assert result["hash_a"][0]["confidence"] is None


# =============================================================================
# SUPERSEDES in RELATION_TYPES
# =============================================================================


class TestSchemaIncludesSupersedesType:
    def test_supersedes_in_relation_types(self):
        from mcp_memory_service.graph.schema import RELATION_TYPES

        assert "SUPERSEDES" in RELATION_TYPES

    def test_original_types_still_present(self):
        from mcp_memory_service.graph.schema import RELATION_TYPES

        assert "CONTRADICTS" in RELATION_TYPES
        assert "RELATES_TO" in RELATION_TYPES
        assert "PRECEDES" in RELATION_TYPES
