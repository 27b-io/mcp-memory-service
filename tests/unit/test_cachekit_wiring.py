"""Tests for CacheKit wiring and cache infrastructure."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
async def _clear_caches():
    """Clear CacheKit caches before and after each test."""
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags, _cached_corpus_count, _cached_embed, _cached_extract_keywords
        await _cached_fetch_all_tags.ainvalidate_cache()
        await _cached_corpus_count.ainvalidate_cache()
        await _cached_embed.ainvalidate_cache()
        await _cached_extract_keywords.ainvalidate_cache()
    except Exception:
        pass
    yield
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags, _cached_corpus_count, _cached_embed, _cached_extract_keywords
        await _cached_fetch_all_tags.ainvalidate_cache()
        await _cached_corpus_count.ainvalidate_cache()
        await _cached_embed.ainvalidate_cache()
        await _cached_extract_keywords.ainvalidate_cache()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_cachekit_available():
    """CacheKit should be importable and the flag should be True."""
    from mcp_memory_service.services.memory_service import _CACHEKIT_AVAILABLE
    assert _CACHEKIT_AVAILABLE is True


@pytest.mark.asyncio
async def test_cached_fetch_all_tags_calls_storage():
    """_cached_fetch_all_tags should delegate to storage.get_all_tags()."""
    import mcp_memory_service.services.memory_service as mod

    mock_storage = AsyncMock()
    mock_storage.get_all_tags = AsyncMock(return_value=["python", "rust"])

    original = mod._storage_ref
    mod._storage_ref = mock_storage
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags
        await _cached_fetch_all_tags.ainvalidate_cache()
        result = await _cached_fetch_all_tags()
        assert set(result) == {"python", "rust"}
        mock_storage.get_all_tags.assert_awaited_once()
    finally:
        mod._storage_ref = original


@pytest.mark.asyncio
async def test_no_ck_kwargs_in_module():
    """_ck_kwargs should no longer exist — CacheKit uses defaults."""
    import mcp_memory_service.services.memory_service as mod
    assert not hasattr(mod, "_ck_kwargs"), "_ck_kwargs should be removed; CacheKit resolves backend from env"


@pytest.mark.asyncio
async def test_cached_corpus_count_delegates_to_storage():
    """_cached_corpus_count should call storage.count() and cache the result."""
    import mcp_memory_service.services.memory_service as mod

    mock_storage = AsyncMock()
    mock_storage.count = AsyncMock(return_value=42)

    original = mod._storage_ref
    mod._storage_ref = mock_storage
    try:
        from mcp_memory_service.services.memory_service import _cached_corpus_count

        await _cached_corpus_count.ainvalidate_cache()
        result = await _cached_corpus_count()
        assert result == 42
        mock_storage.count.assert_awaited_once()

        # Second call should be cached (no additional storage call)
        result2 = await _cached_corpus_count()
        assert result2 == 42
        assert mock_storage.count.await_count == 1
    finally:
        mod._storage_ref = original


@pytest.mark.asyncio
async def test_cached_embed_delegates_to_embed_fn():
    """_cached_embed should call the embed function and cache the result."""
    import mcp_memory_service.services.memory_service as mod

    mock_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    original_fn = mod._embed_fn
    mod._embed_fn = mock_embed
    try:
        from mcp_memory_service.services.memory_service import _cached_embed

        await _cached_embed.ainvalidate_cache()
        result = await _cached_embed("test query")
        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_awaited_with(["test query"])
    finally:
        mod._embed_fn = original_fn


@pytest.mark.asyncio
async def test_embed_namespace_includes_model_name():
    """Embedding cache namespace should include model name for safety on model changes."""
    import mcp_memory_service.services.memory_service as mod

    assert hasattr(mod, "_model_name"), "_model_name should exist at module level"
    # Model name should be set from config
    assert len(mod._model_name) > 0, "_model_name should be non-empty"


@pytest.mark.asyncio
async def test_get_cache_health_returns_dict():
    """_get_cache_health should return a dict with status and components."""
    from mcp_memory_service.web.api.health import _get_cache_health

    result = await _get_cache_health()
    assert isinstance(result, dict)
    assert "status" in result


@pytest.mark.asyncio
async def test_cached_extract_keywords_returns_keywords():
    """_cached_extract_keywords should extract and cache keywords."""
    import mcp_memory_service.services.memory_service as mod

    mock_storage = AsyncMock()
    mock_storage.get_all_tags = AsyncMock(return_value=["python", "rust", "docker"])

    original = mod._storage_ref
    mod._storage_ref = mock_storage
    try:
        from mcp_memory_service.services.memory_service import (
            _cached_extract_keywords,
            _cached_fetch_all_tags,
        )

        await _cached_fetch_all_tags.ainvalidate_cache()
        await _cached_extract_keywords.ainvalidate_cache()

        result = await _cached_extract_keywords("python docker deployment")
        assert "python" in result
        assert "docker" in result
    finally:
        mod._storage_ref = original
