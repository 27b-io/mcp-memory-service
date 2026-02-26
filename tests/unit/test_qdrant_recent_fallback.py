"""Tests for Qdrant get_all_memories / get_recent_memories order_by fallback.

Verifies that when Qdrant's server-side order_by returns zero results
(missing payload index or incompatible created_at types), the methods
fall back to unordered scroll + Python sort.

Regression test for https://github.com/27b-io/mcp-memory-service/issues/123
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.storage.qdrant_storage import QdrantStorage


def _make_point(point_id: str, created_at: float, content: str = "test") -> MagicMock:
    """Create a mock Qdrant point with payload."""
    point = MagicMock()
    point.id = point_id
    point.payload = {
        "content": content,
        "content_hash": point_id,
        "tags": [],
        "memory_type": "note",
        "metadata": {},
        "created_at": created_at,
        "updated_at": created_at,
        "emotional_valence": None,
        "salience_score": 0.5,
        "access_count": 0,
        "access_timestamps": [],
        "summary": None,
    }
    return point


@pytest.fixture
def qdrant_storage():
    """Create a QdrantMemoryStorage with mocked client."""
    with patch.object(QdrantStorage, "__init__", lambda self: None):
        storage = QdrantStorage()
        storage.client = MagicMock()
        storage.collection_name = "test_collection"
        storage.METADATA_POINT_ID = 1
        storage._failure_count = 0
        storage._circuit_open_until = None
        storage._failure_threshold = 5
        storage._circuit_timeout = 60
        return storage


class TestGetAllMemoriesFallback:
    """Test get_all_memories falls back when order_by returns empty."""

    @pytest.mark.asyncio
    async def test_ordered_scroll_works_normally(self, qdrant_storage):
        """When order_by returns results, no fallback needed."""
        points = [_make_point("p1", 200.0), _make_point("p2", 100.0)]
        # First call returns points, second returns empty (exhausted)
        qdrant_storage.client.scroll.side_effect = [(points, None), ([], None)]

        memories = await qdrant_storage.get_all_memories(limit=10)

        assert len(memories) == 2
        assert memories[0].content_hash == "p1"
        # order_by was used (scroll called with order_by kwarg)
        call_kwargs = qdrant_storage.client.scroll.call_args[1]
        assert "order_by" in call_kwargs

    @pytest.mark.asyncio
    async def test_fallback_when_ordered_returns_empty(self, qdrant_storage):
        """When order_by returns empty but points exist, fall back to unordered."""
        points = [_make_point("p1", 200.0), _make_point("p2", 100.0), _make_point("p3", 300.0)]

        # First call (ordered) returns empty; second call (unordered) returns points
        call_count = 0

        def scroll_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if "order_by" in kwargs:
                return ([], None)  # order_by fails
            return (points, None)  # unordered works

        qdrant_storage.client.scroll.side_effect = scroll_side_effect

        # Mock count to report points exist (unfiltered: uses get_collection fast path)
        collection_info = MagicMock()
        collection_info.points_count = 4  # 3 points + 1 metadata
        qdrant_storage.client.get_collection.return_value = collection_info

        memories = await qdrant_storage.get_all_memories(limit=10)

        assert len(memories) == 3
        # Verify sorted by created_at DESC
        assert memories[0].created_at == 300.0
        assert memories[1].created_at == 200.0
        assert memories[2].created_at == 100.0

    @pytest.mark.asyncio
    async def test_no_fallback_when_collection_truly_empty(self, qdrant_storage):
        """When order_by returns empty AND count is 0, no fallback."""
        qdrant_storage.client.scroll.return_value = ([], None)

        qdrant_storage.client.get_collection.return_value = MagicMock(points_count=1)  # only metadata

        memories = await qdrant_storage.get_all_memories(limit=10)

        assert len(memories) == 0
        # scroll should only be called once (the ordered attempt)
        assert qdrant_storage.client.scroll.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_respects_offset_and_limit(self, qdrant_storage):
        """Fallback correctly applies offset and limit."""
        points = [_make_point(f"p{i}", float(i * 100)) for i in range(5)]

        def scroll_side_effect(**kwargs):
            if "order_by" in kwargs:
                return ([], None)
            return (points, None)

        qdrant_storage.client.scroll.side_effect = scroll_side_effect

        qdrant_storage.client.get_collection.return_value = MagicMock(points_count=6)  # 5 + metadata

        memories = await qdrant_storage.get_all_memories(limit=2, offset=1)

        # 5 points sorted desc: p4(400), p3(300), p2(200), p1(100), p0(0)
        # offset=1, limit=2 → p3, p2
        assert len(memories) == 2
        assert memories[0].created_at == 300.0
        assert memories[1].created_at == 200.0

    @pytest.mark.asyncio
    async def test_filtered_fallback_no_recursion(self, qdrant_storage):
        """Fallback with memory_type filter must not recurse into count_all_memories → get_all_memories."""
        points = [_make_point("p1", 200.0)]
        points[0].payload["memory_type"] = "note"

        def scroll_side_effect(**kwargs):
            if "order_by" in kwargs:
                return ([], None)
            return (points, None)

        qdrant_storage.client.scroll.side_effect = scroll_side_effect

        # count() uses Qdrant's native server-side count API (no get_all_memories call)
        count_result = MagicMock()
        count_result.count = 1
        qdrant_storage.client.count.return_value = count_result

        memories = await qdrant_storage.get_all_memories(limit=10, memory_type="note")

        assert len(memories) == 1
        # Verify count was called with a filter (not get_collection)
        qdrant_storage.client.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_filtered_fallback_with_tags(self, qdrant_storage):
        """Fallback with tags filter uses server-side count, no recursion."""
        points = [_make_point("p1", 200.0)]
        points[0].payload["tags"] = ["important"]

        def scroll_side_effect(**kwargs):
            if "order_by" in kwargs:
                return ([], None)
            return (points, None)

        qdrant_storage.client.scroll.side_effect = scroll_side_effect

        count_result = MagicMock()
        count_result.count = 1
        qdrant_storage.client.count.return_value = count_result

        memories = await qdrant_storage.get_all_memories(limit=10, tags=["important"])

        assert len(memories) == 1


class TestCountAllMemories:
    """Test count_all_memories uses native Qdrant count API."""

    @pytest.mark.asyncio
    async def test_unfiltered_count_excludes_metadata(self, qdrant_storage):
        """Unfiltered count uses fast path (get_collection) and subtracts 1 for __metadata__."""
        collection_info = MagicMock()
        collection_info.points_count = 101  # 100 memories + 1 metadata
        qdrant_storage.client.get_collection.return_value = collection_info

        result = await qdrant_storage.count_all_memories()

        assert result == 100
        qdrant_storage.client.get_collection.assert_called_once()
        qdrant_storage.client.count.assert_not_called()

    @pytest.mark.asyncio
    async def test_filtered_count_does_not_subtract_metadata(self, qdrant_storage):
        """Filtered count doesn't subtract metadata (metadata has no memory_type/tags)."""
        count_result = MagicMock()
        count_result.count = 5
        qdrant_storage.client.count.return_value = count_result

        result = await qdrant_storage.count_all_memories(memory_type="note")

        assert result == 5


class TestGetRecentMemoriesFallback:
    """Test get_recent_memories falls back when order_by returns empty."""

    @pytest.mark.asyncio
    async def test_ordered_scroll_works_normally(self, qdrant_storage):
        """When order_by returns results, no fallback needed."""
        points = [_make_point("p1", 200.0), _make_point("p2", 100.0)]
        qdrant_storage.client.scroll.return_value = (points, None)

        memories = await qdrant_storage.get_recent_memories(n=5)

        assert len(memories) == 2
        assert memories[0].content_hash == "p1"

    @pytest.mark.asyncio
    async def test_fallback_when_ordered_returns_empty(self, qdrant_storage):
        """When order_by returns empty but points exist, fall back to unordered."""
        points = [_make_point("p1", 200.0), _make_point("p2", 100.0)]

        def scroll_side_effect(**kwargs):
            if "order_by" in kwargs:
                return ([], None)
            return (points, None)

        qdrant_storage.client.scroll.side_effect = scroll_side_effect

        qdrant_storage.client.get_collection.return_value = MagicMock(points_count=3)  # 2 + metadata

        memories = await qdrant_storage.get_recent_memories(n=5)

        assert len(memories) == 2
        assert memories[0].created_at == 200.0
        assert memories[1].created_at == 100.0


class TestEnsurePayloadIndexes:
    """Test _ensure_payload_indexes is called on initialization."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_called_for_existing_collection(self, qdrant_storage):
        """Payload indexes should be created even for existing collections."""
        qdrant_storage._initialized = False
        qdrant_storage.url = "http://localhost:6333"
        qdrant_storage.storage_path = None
        qdrant_storage.embedding_model = "test-model"
        qdrant_storage._vector_size = 768
        qdrant_storage.quantization_enabled = False
        qdrant_storage.config = MagicMock()

        with (
            patch.object(qdrant_storage, "_ensure_model_loaded"),
            patch.object(qdrant_storage, "_collection_exists", new_callable=AsyncMock, return_value=True),
            patch.object(qdrant_storage, "_verify_model_compatibility", new_callable=AsyncMock),
            patch.object(qdrant_storage, "_ensure_payload_indexes", new_callable=AsyncMock) as mock_ensure,
            patch.object(qdrant_storage, "_ensure_tag_collection", new_callable=AsyncMock),
        ):
            # Mock the client creation and model instance
            qdrant_storage.client = MagicMock()
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            qdrant_storage._embedding_model_instance = mock_model

            await qdrant_storage.initialize()

            mock_ensure.assert_called_once()
