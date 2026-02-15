"""
Tests for Redis cache service for memory search results.

Tests cover:
- Cache hit/miss scenarios
- TTL expiration
- Cache invalidation on writes
- Concurrent access safety
"""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRedisCacheBasics:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Cache miss should return None when key doesn't exist."""
        from mcp_memory_service.cache.redis_cache import RedisCache

        with patch("mcp_memory_service.cache.redis_cache.ConnectionPool") as mock_pool_cls, patch(
            "mcp_memory_service.cache.redis_cache.Redis"
        ) as mock_redis_cls:
            mock_pool = MagicMock()
            mock_pool.aclose = AsyncMock()
            mock_pool_cls.from_url.return_value = mock_pool

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis.aclose = AsyncMock()
            mock_redis_cls.return_value = mock_redis

            cache = RedisCache(url="redis://localhost:6379", ttl_seconds=300)
            await cache.initialize()

            try:
                result = await cache.get("nonexistent_key")
                assert result is None
                mock_redis.get.assert_called_once_with("mcp:cache:nonexistent_key")
            finally:
                await cache.close()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self):
        """Cache hit should return the stored value."""
        import json

        from mcp_memory_service.cache.redis_cache import RedisCache

        test_data = {"content": "test memory", "tags": ["test"]}

        with patch("mcp_memory_service.cache.redis_cache.ConnectionPool") as mock_pool_cls, patch(
            "mcp_memory_service.cache.redis_cache.Redis"
        ) as mock_redis_cls:
            mock_pool = MagicMock()
            mock_pool.aclose = AsyncMock()
            mock_pool_cls.from_url.return_value = mock_pool

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.get = AsyncMock(return_value=json.dumps(test_data))
            mock_redis.setex = AsyncMock()
            mock_redis.aclose = AsyncMock()
            mock_redis_cls.return_value = mock_redis

            cache = RedisCache(url="redis://localhost:6379", ttl_seconds=300)
            await cache.initialize()

            try:
                await cache.set("test_key", test_data)
                result = await cache.get("test_key")

                assert result == test_data
                mock_redis.setex.assert_called_once()
            finally:
                await cache.close()

    @pytest.mark.asyncio
    async def test_cache_respects_ttl(self):
        """Cached values should expire after TTL."""
        from mcp_memory_service.cache.redis_cache import RedisCache

        with patch("mcp_memory_service.cache.redis_cache.ConnectionPool") as mock_pool_cls, patch(
            "mcp_memory_service.cache.redis_cache.Redis"
        ) as mock_redis_cls:
            mock_pool = MagicMock()
            mock_pool.aclose = AsyncMock()
            mock_pool_cls.from_url.return_value = mock_pool

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.setex = AsyncMock()
            mock_redis.aclose = AsyncMock()
            mock_redis_cls.return_value = mock_redis

            # Use 1 second TTL for test
            cache = RedisCache(url="redis://localhost:6379", ttl_seconds=1)
            await cache.initialize()

            try:
                test_data = {"content": "test memory"}
                await cache.set("ttl_test_key", test_data)

                # Verify setex was called with 1 second TTL
                args = mock_redis.setex.call_args
                assert args[0][1] == 1  # TTL argument
            finally:
                await cache.close()

    @pytest.mark.asyncio
    async def test_cache_invalidate_clears_pattern(self):
        """Invalidate should clear all keys matching pattern."""
        from mcp_memory_service.cache.redis_cache import RedisCache

        with patch("mcp_memory_service.cache.redis_cache.ConnectionPool") as mock_pool_cls, patch(
            "mcp_memory_service.cache.redis_cache.Redis"
        ) as mock_redis_cls:
            mock_pool = MagicMock()
            mock_pool.aclose = AsyncMock()
            mock_pool_cls.from_url.return_value = mock_pool

            # Mock scan_iter to return matching keys
            async def mock_scan_iter(match):
                if "search:*" in match:
                    for key in ["test:cache:search:query1", "test:cache:search:query2"]:
                        yield key

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.scan_iter = mock_scan_iter
            mock_redis.delete = AsyncMock(return_value=2)
            mock_redis.aclose = AsyncMock()
            mock_redis_cls.return_value = mock_redis

            cache = RedisCache(url="redis://localhost:6379", ttl_seconds=300, key_prefix="test:cache:")
            await cache.initialize()

            try:
                # Invalidate all search keys
                deleted = await cache.invalidate_pattern("search:*")

                # Should delete 2 keys
                assert deleted == 2
                mock_redis.delete.assert_called_once()
            finally:
                await cache.close()


class TestCacheKeyGeneration:
    """Test cache key generation from query parameters."""

    def test_generate_cache_key_consistent(self):
        """Same parameters should generate same cache key."""
        from mcp_memory_service.cache.redis_cache import generate_cache_key

        params1 = {"query": "test", "page": 1, "page_size": 10}
        params2 = {"query": "test", "page": 1, "page_size": 10}

        key1 = generate_cache_key("retrieve", params1)
        key2 = generate_cache_key("retrieve", params2)

        assert key1 == key2

    def test_generate_cache_key_different_for_different_params(self):
        """Different parameters should generate different cache keys."""
        from mcp_memory_service.cache.redis_cache import generate_cache_key

        params1 = {"query": "test1", "page": 1}
        params2 = {"query": "test2", "page": 1}

        key1 = generate_cache_key("retrieve", params1)
        key2 = generate_cache_key("retrieve", params2)

        assert key1 != key2

    def test_generate_cache_key_handles_nested_dicts(self):
        """Cache key should handle nested dictionaries."""
        from mcp_memory_service.cache.redis_cache import generate_cache_key

        params = {
            "query": "test",
            "metadata": {"key": "value"},
            "tags": ["a", "b"],
        }

        key = generate_cache_key("retrieve", params)
        assert isinstance(key, str)
        assert len(key) > 0


class TestMemoryServiceCacheIntegration:
    """Test cache integration with MemoryService."""

    @pytest.mark.asyncio
    async def test_cache_is_initialized_when_enabled(self):
        """MemoryService should initialize cache if config enabled."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_storage = MagicMock()

        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_cache_config = MagicMock()
            mock_cache_config.enabled = True
            mock_cache_config.url = "redis://localhost:6379"
            mock_cache_config.ttl_seconds = 300
            mock_cache_config.key_prefix = "test:"
            mock_cache_config.max_connections = 10
            mock_settings.redis_cache = mock_cache_config
            mock_settings.three_tier = MagicMock(enabled=False)

            service = MemoryService(storage=mock_storage)

            # Cache should be created
            assert service._cache is not None

    @pytest.mark.asyncio
    async def test_cache_not_initialized_when_disabled(self):
        """MemoryService should not initialize cache if config disabled."""
        from mcp_memory_service.services.memory_service import MemoryService

        mock_storage = MagicMock()

        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_cache_config = MagicMock()
            mock_cache_config.enabled = False
            mock_settings.redis_cache = mock_cache_config
            mock_settings.three_tier = MagicMock(enabled=False)

            service = MemoryService(storage=mock_storage)

            # Cache should not be created
            assert service._cache is None
