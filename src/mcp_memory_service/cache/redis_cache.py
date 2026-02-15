"""
Redis cache implementation for memory search results.

Provides caching layer for expensive search operations with:
- Configurable TTL (default 5 minutes)
- Pattern-based invalidation
- JSON serialization for complex objects
"""

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)


def generate_cache_key(operation: str, params: dict[str, Any]) -> str:
    """
    Generate a consistent cache key from operation and parameters.

    Args:
        operation: The operation type (e.g., 'retrieve', 'search', 'scan')
        params: Dictionary of query parameters

    Returns:
        Cache key string in format: operation:param1=value1:param2=value2:...
    """
    # Sort params for consistent key generation
    sorted_params = sorted(params.items())

    # Serialize to JSON for complex types (lists, dicts)
    param_strs = []
    for key, value in sorted_params:
        if isinstance(value, (dict, list)):
            # Use JSON for complex types
            value_str = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            value_str = str(value)
        param_strs.append(f"{key}={value_str}")

    key_base = f"{operation}:" + ":".join(param_strs)

    # Use hash for very long keys to keep Redis keys reasonable
    if len(key_base) > 200:
        key_hash = hashlib.sha256(key_base.encode()).hexdigest()[:16]
        return f"{operation}:{key_hash}"

    return key_base


class RedisCache:
    """
    Redis-based cache for memory search results.

    Provides get/set operations with TTL and pattern-based invalidation.
    Uses JSON serialization for storing complex Python objects.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl_seconds: int = 300,
        key_prefix: str = "mcp:cache:",
        max_connections: int = 10,
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
            ttl_seconds: Default TTL for cache entries (default 300 = 5 minutes)
            key_prefix: Prefix for all cache keys
            max_connections: Maximum Redis connections in pool
        """
        self.url = url
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.max_connections = max_connections

        self._pool: ConnectionPool | None = None
        self._redis: Redis | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self._initialized:
            return

        self._pool = ConnectionPool.from_url(
            self.url,
            max_connections=self.max_connections,
            decode_responses=True,  # Auto-decode bytes to str
        )

        self._redis = Redis(connection_pool=self._pool)

        # Test connection
        try:
            await self._redis.ping()
            self._initialized = True
            logger.info(f"RedisCache initialized: {self.url} (TTL={self.ttl_seconds}s)")
        except Exception as e:
            logger.error(f"RedisCache initialization failed: {e}")
            # Close on failure
            if self._redis:
                await self._redis.aclose()
            if self._pool:
                await self._pool.aclose()
            self._redis = None
            self._pool = None
            raise

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

        if self._pool:
            await self._pool.aclose()
            self._pool = None

        self._initialized = False

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """
        Get cached value by key.

        Args:
            key: Cache key (without prefix)

        Returns:
            Cached value as dict, or None if not found or error
        """
        if not self._initialized or not self._redis:
            return None

        try:
            full_key = self._make_key(key)
            value = await self._redis.get(full_key)

            if value is None:
                return None

            # Deserialize JSON
            return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: dict[str, Any]) -> bool:
        """
        Set cached value with TTL.

        Args:
            key: Cache key (without prefix)
            value: Value to cache (must be JSON-serializable)

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or not self._redis:
            return False

        try:
            full_key = self._make_key(key)
            # Serialize to JSON
            json_value = json.dumps(value)

            # Set with TTL
            await self._redis.setex(full_key, self.ttl_seconds, json_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete cached value.

        Args:
            key: Cache key (without prefix)

        Returns:
            True if deleted, False otherwise
        """
        if not self._initialized or not self._redis:
            return False

        try:
            full_key = self._make_key(key)
            await self._redis.delete(full_key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "search:*")

        Returns:
            Number of keys deleted
        """
        if not self._initialized or not self._redis:
            return 0

        try:
            full_pattern = self._make_key(pattern)

            # Find all matching keys
            keys = []
            async for key in self._redis.scan_iter(match=full_pattern):
                keys.append(key)

            if not keys:
                return 0

            # Delete all matching keys
            deleted = await self._redis.delete(*keys)
            logger.debug(f"Cache invalidated {deleted} keys matching pattern: {pattern}")
            return deleted
        except Exception as e:
            logger.warning(f"Cache invalidation failed for pattern {pattern}: {e}")
            return 0

    async def invalidate_all(self) -> int:
        """
        Invalidate all cache keys with our prefix.

        Returns:
            Number of keys deleted
        """
        return await self.invalidate_pattern("*")
