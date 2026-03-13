"""Tests for CachedEmbeddingProvider wrapper."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
from mcp_memory_service.embedding.protocol import EmbeddingProvider


@pytest.fixture
def inner_provider():
    provider = AsyncMock(spec=EmbeddingProvider)
    provider.embed_batch = AsyncMock(return_value=[[0.1] * 768])
    type(provider).dimensions = PropertyMock(return_value=768)
    type(provider).model_name = PropertyMock(return_value="test-model")
    return provider


class TestCachedEmbeddingProvider:
    def test_satisfies_protocol(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert isinstance(cached, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_first_call_delegates_to_inner(self, inner_provider):
        """Cache miss calls inner provider once per text."""
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch(["hello"])
        assert inner_provider.embed_batch.await_count == 1
        assert result == [[0.1] * 768]

    @pytest.mark.asyncio
    async def test_cache_hit_skips_inner(self, inner_provider):
        """Second call for same text+prompt_name doesn't call inner."""
        cached = CachedEmbeddingProvider(inner_provider)
        await cached.embed_batch(["hello"])
        await cached.embed_batch(["hello"])
        # CacheKit caches per-text — second call should be a hit.
        assert inner_provider.embed_batch.await_count == 1

    @pytest.mark.asyncio
    async def test_different_prompt_name_separate_cache(self, inner_provider):
        """'query' and 'passage' for same text produce separate cache entries."""
        inner_provider.embed_batch = AsyncMock(side_effect=[[[0.1] * 768], [[0.9] * 768]])
        cached = CachedEmbeddingProvider(inner_provider)
        r1 = await cached.embed_batch(["hello"], prompt_name="query")
        r2 = await cached.embed_batch(["hello"], prompt_name="passage")
        assert r1 != r2
        assert inner_provider.embed_batch.await_count == 2

    @pytest.mark.asyncio
    async def test_batch_calls_inner_per_text(self, inner_provider):
        """Batch of N texts results in N per-text cache lookups."""
        inner_provider.embed_batch = AsyncMock(side_effect=[[[0.1] * 768], [[0.2] * 768]])
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert inner_provider.embed_batch.await_count == 2

    @pytest.mark.asyncio
    async def test_mixed_hits_and_misses(self, inner_provider):
        """Pre-cached text is not re-embedded."""
        call_count = 0

        async def track_calls(texts, prompt_name="query"):
            nonlocal call_count
            call_count += 1
            return [[float(call_count)] * 768]

        inner_provider.embed_batch = AsyncMock(side_effect=track_calls)
        cached = CachedEmbeddingProvider(inner_provider)

        # Cache "hello"
        await cached.embed_batch(["hello"])
        assert call_count == 1

        # Request ["hello", "world"] — "hello" should be cached
        result = await cached.embed_batch(["hello", "world"])
        assert len(result) == 2
        # Only "world" should trigger a new inner call
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_empty_batch(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch([])
        assert result == []
        inner_provider.embed_batch.assert_not_awaited()

    def test_dimensions_delegates(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert cached.dimensions == 768

    def test_model_name_delegates(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert cached.model_name == "test-model"
