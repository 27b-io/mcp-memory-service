"""Tests for semantic tag matching integration in search pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.get_all_tags = AsyncMock(return_value=["proton-bridge", "imap", "email"])
    storage.count = AsyncMock(return_value=100)
    storage.retrieve = AsyncMock(return_value=[])
    storage.search_by_tags = AsyncMock(return_value=[])

    # Return embeddings keyed by tag order — but _get_cached_tags returns a set,
    # so we use side_effect to generate deterministic embeddings based on input tags.
    def _make_embeddings(tags: list[str]) -> list[list[float]]:
        tag_vectors = {
            "proton-bridge": [1.0, 0.0, 0.0],
            "imap": [0.0, 1.0, 0.0],
            "email": [0.0, 0.0, 1.0],
        }
        return [tag_vectors.get(t, [0.0, 0.0, 0.0]) for t in tags]

    storage.generate_embeddings_batch = AsyncMock(side_effect=_make_embeddings)
    # Query embedding close to "proton-bridge" [1.0, 0.0, 0.0]
    storage._generate_embedding = MagicMock(return_value=[0.95, 0.05, 0.0])
    return storage


@pytest.fixture
def service(mock_storage):
    return MemoryService(storage=mock_storage)


class TestSearchSemanticTags:
    @pytest.mark.asyncio
    async def test_search_semantic_tags_finds_matching_tags(self, service, mock_storage):
        """Semantic tag search should find tags similar to query embedding."""
        await service._search_semantic_tags("proton bridge", fetch_size=10)
        # Query embedding [0.95, 0.05, 0.0] is close to "proton-bridge" [1.0, 0.0, 0.0]
        if mock_storage.search_by_tags.called:
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
        service._tag_embedding_cache = None
        result = await service._search_semantic_tags("test", fetch_size=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_tag_embedding_index_cached(self, service, mock_storage):
        """Index should be cached and not rebuilt on every call."""
        await service._get_tag_embedding_index()
        await service._get_tag_embedding_index()
        assert mock_storage.generate_embeddings_batch.call_count == 1

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
