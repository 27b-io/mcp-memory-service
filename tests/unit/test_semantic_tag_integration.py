"""Tests for semantic tag matching integration in search pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture(autouse=True)
async def _clear_cachekit_caches():
    """Clear CacheKit L1 caches before and after tests."""
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags

        await _cached_fetch_all_tags.ainvalidate_cache()
    except Exception:  # noqa: BLE001
        pass
    yield
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags

        await _cached_fetch_all_tags.ainvalidate_cache()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.get_all_tags = AsyncMock(return_value=["proton-bridge", "imap", "email"])
    storage.count = AsyncMock(return_value=100)
    storage.retrieve = AsyncMock(return_value=[])
    storage.search_by_tags = AsyncMock(return_value=[])

    # generate_embeddings_batch returns query embeddings
    def _make_embeddings(tags, prompt_name="query"):
        query_vectors = {
            "proton bridge": [0.95, 0.05, 0.0],
            "test": [0.5, 0.5, 0.0],
        }
        return [query_vectors.get(t, [0.0, 0.0, 0.0]) for t in tags]

    storage.generate_embeddings_batch = AsyncMock(side_effect=_make_embeddings)

    # search_similar_tags: native Qdrant k-NN on tag collection
    storage.search_similar_tags = AsyncMock(return_value=["proton-bridge"])
    return storage


@pytest.fixture
def service(mock_storage):
    return MemoryService(storage=mock_storage)


class TestSearchSemanticTags:
    @pytest.mark.asyncio
    async def test_search_semantic_tags_finds_matching_tags(self, service, mock_storage):
        """Semantic tag search should find tags similar to query embedding."""
        await service._search_semantic_tags("proton bridge", fetch_size=10)
        mock_storage.search_similar_tags.assert_called_once()
        # After finding similar tags, should search memories by those tags
        mock_storage.search_by_tags.assert_called_once()
        call_tags = mock_storage.search_by_tags.call_args.kwargs.get("tags", [])
        assert "proton-bridge" in call_tags

    @pytest.mark.asyncio
    async def test_search_semantic_tags_disabled(self, service):
        """When disabled, should return empty list without calling storage."""
        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_settings.semantic_tag.enabled = False
            result = await service._search_semantic_tags("test", fetch_size=10)
            assert result == []

    @pytest.mark.asyncio
    async def test_search_semantic_tags_non_fatal(self, service, mock_storage):
        """Errors should be caught and return empty list."""
        mock_storage.generate_embeddings_batch.side_effect = RuntimeError("model error")
        result = await service._search_semantic_tags("test", fetch_size=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_semantic_tags_no_matches(self, service, mock_storage):
        """When no similar tags found, should return empty list."""
        mock_storage.search_similar_tags = AsyncMock(return_value=[])
        result = await service._search_semantic_tags("unrelated query", fetch_size=10)
        assert result == []
        mock_storage.search_by_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_vector_search_includes_semantic_tags(self, service, mock_storage):
        """_single_vector_search should include semantic tag matches."""
        semantic_memory = Memory(
            content="Proton Bridge config",
            content_hash="semantic_hash",
            tags=["proton-bridge"],
            memory_type="note",
            created_at=1.0,
            updated_at=1.0,
        )

        with patch.object(service, "_search_semantic_tags", return_value=[semantic_memory]):
            result = await service._single_vector_search(
                query="proton bridge",
                keywords=[],
                fetch_size=10,
                memory_type=None,
                alpha=0.5,
            )
            result_hashes = [m.content_hash for m, _, _ in result]
            assert "semantic_hash" in result_hashes
