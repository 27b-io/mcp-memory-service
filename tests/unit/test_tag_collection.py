"""Tests for persistent tag embedding collection in QdrantStorage."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.storage.qdrant_storage import QdrantStorage


def _tag_to_uuid(tag: str) -> str:
    hex_digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(hex_digest))


@pytest.fixture
def qdrant_storage():
    """Create QdrantStorage with mocked client for tag collection tests."""
    with patch.object(QdrantStorage, "__init__", lambda self: None):
        storage = QdrantStorage()
        storage.client = MagicMock()
        storage.collection_name = "memories"
        storage.tag_collection_name = "memories_tags"
        storage._vector_size = 3
        storage._known_tags = set()
        storage.embedding_model = "test-model"
        storage.METADATA_POINT_ID = 1
        storage._failure_count = 0
        storage._circuit_open_until = None
        storage._failure_threshold = 5
        storage._circuit_timeout = 60
        return storage


class TestTagCollectionInit:
    @pytest.mark.asyncio
    async def test_creates_tag_collection_when_missing(self, qdrant_storage):
        """Should create tag collection and migrate tags on first run."""
        collections_mock = MagicMock()
        col = MagicMock()
        col.name = "memories"
        collections_mock.collections = [col]
        qdrant_storage.client.get_collections.return_value = collections_mock

        # get_all_tags returns existing tags
        with patch.object(qdrant_storage, "get_all_tags", new_callable=AsyncMock, return_value=["python", "docker"]):
            with patch.object(qdrant_storage, "_upsert_tag_embeddings", new_callable=AsyncMock) as mock_upsert:
                await qdrant_storage._ensure_tag_collection()
                qdrant_storage.client.create_collection.assert_called_once()
                mock_upsert.assert_called_once_with(["python", "docker"])

    @pytest.mark.asyncio
    async def test_creates_collection_no_migration_when_no_tags(self, qdrant_storage):
        """Should create tag collection but skip migration when no tags exist."""
        collections_mock = MagicMock()
        col = MagicMock()
        col.name = "memories"
        collections_mock.collections = [col]
        qdrant_storage.client.get_collections.return_value = collections_mock

        with patch.object(qdrant_storage, "get_all_tags", new_callable=AsyncMock, return_value=[]):
            with patch.object(qdrant_storage, "_upsert_tag_embeddings", new_callable=AsyncMock) as mock_upsert:
                await qdrant_storage._ensure_tag_collection()
                qdrant_storage.client.create_collection.assert_called_once()
                mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_loads_known_tags_when_collection_exists(self, qdrant_storage):
        """Should populate _known_tags from existing tag collection."""
        collections_mock = MagicMock()
        col1 = MagicMock()
        col1.name = "memories"
        col2 = MagicMock()
        col2.name = "memories_tags"
        collections_mock.collections = [col1, col2]
        qdrant_storage.client.get_collections.return_value = collections_mock

        # Simulate two tags in the tag collection
        p1 = MagicMock()
        p1.payload = {"tag": "python"}
        p2 = MagicMock()
        p2.payload = {"tag": "docker"}
        qdrant_storage.client.scroll.return_value = ([p1, p2], None)

        await qdrant_storage._ensure_tag_collection()
        assert qdrant_storage._known_tags == {"python", "docker"}


class TestIndexNewTags:
    @pytest.mark.asyncio
    async def test_indexes_only_new_tags(self, qdrant_storage):
        """Should only encode tags not in _known_tags."""
        qdrant_storage._known_tags = {"python"}

        with patch.object(qdrant_storage, "_upsert_tag_embeddings", new_callable=AsyncMock) as mock_upsert:
            await qdrant_storage.index_new_tags(["python", "docker", "redis"])
            mock_upsert.assert_called_once_with(["docker", "redis"])

    @pytest.mark.asyncio
    async def test_noop_when_all_tags_known(self, qdrant_storage):
        """Should not encode anything when all tags are already known."""
        qdrant_storage._known_tags = {"python", "docker"}

        with patch.object(qdrant_storage, "_upsert_tag_embeddings", new_callable=AsyncMock) as mock_upsert:
            await qdrant_storage.index_new_tags(["python", "docker"])
            mock_upsert.assert_not_called()


class TestSearchSimilarTags:
    @pytest.mark.asyncio
    async def test_returns_tags_from_qdrant_search(self, qdrant_storage):
        """Should return tag strings from Qdrant search results."""
        qdrant_storage._known_tags = {"python", "docker", "redis"}

        hit1 = MagicMock()
        hit1.payload = {"tag": "python"}
        hit1.score = 0.95
        hit2 = MagicMock()
        hit2.payload = {"tag": "docker"}
        hit2.score = 0.7
        qdrant_storage.client.search.return_value = [hit1, hit2]

        result = await qdrant_storage.search_similar_tags(
            query_embedding=[1.0, 0.0, 0.0],
            threshold=0.5,
            max_tags=10,
        )
        assert result == ["python", "docker"]
        qdrant_storage.client.search.assert_called_once()
        call_kwargs = qdrant_storage.client.search.call_args.kwargs
        assert call_kwargs["collection_name"] == "memories_tags"
        assert call_kwargs["score_threshold"] == 0.5
        assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_known_tags(self, qdrant_storage):
        """Should short-circuit when _known_tags is empty."""
        qdrant_storage._known_tags = set()
        result = await qdrant_storage.search_similar_tags([1.0, 0.0, 0.0])
        assert result == []
        qdrant_storage.client.search.assert_not_called()


class TestUpsertTagEmbeddings:
    @pytest.mark.asyncio
    async def test_upserts_points_with_passage_prompt(self, qdrant_storage):
        """Should encode with 'passage' prompt and upsert to tag collection."""
        with patch.object(
            qdrant_storage,
            "generate_embeddings_batch",
            new_callable=AsyncMock,
            return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ) as mock_embed:
            await qdrant_storage._upsert_tag_embeddings(["python", "docker"])
            mock_embed.assert_called_once_with(["python", "docker"], prompt_name="passage")
            qdrant_storage.client.upsert.assert_called_once()
            call_kwargs = qdrant_storage.client.upsert.call_args.kwargs
            assert call_kwargs["collection_name"] == "memories_tags"
            points = call_kwargs["points"]
            assert len(points) == 2
            assert points[0].id == _tag_to_uuid("python")
            assert points[0].payload["tag"] == "python"
            assert points[1].payload["tag"] == "docker"

        assert qdrant_storage._known_tags == {"python", "docker"}

    @pytest.mark.asyncio
    async def test_noop_for_empty_list(self, qdrant_storage):
        """Should not call anything for empty tag list."""
        await qdrant_storage._upsert_tag_embeddings([])
        qdrant_storage.client.upsert.assert_not_called()
