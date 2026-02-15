"""Quota service for enforcing per-client storage limits."""

import logging
import time

from ..config import QuotaSettings
from ..storage.base import MemoryStorage
from ..utils.quota import QuotaExceededError, QuotaStatus

logger = logging.getLogger(__name__)


class QuotaService:
    """Manages quota enforcement for memory storage."""

    def __init__(
        self,
        storage: MemoryStorage,
        settings: QuotaSettings,
    ):
        self.storage = storage
        self.settings = settings

    async def check_quota(self, client_id: str) -> QuotaStatus:
        """
        Check all quotas and raise QuotaExceededError if any exceeded.
        Returns QuotaStatus with warning flags if approaching limits.
        """
        status = await self._compute_quota_status(client_id)

        # Check hard limits
        if status.memory_usage_pct >= 1.0:
            raise QuotaExceededError(
                quota_type="memory_count",
                current=status.memory_count,
                limit=status.memory_limit,
                client_id=client_id,
            )

        if status.storage_usage_pct >= 1.0:
            raise QuotaExceededError(
                quota_type="storage_size",
                current=status.storage_bytes,
                limit=status.storage_limit,
                client_id=client_id,
            )

        if status.rate_usage_pct >= 1.0:
            retry_after = self.settings.rate_limit_window_seconds
            raise QuotaExceededError(
                quota_type="rate_limit",
                current=status.memories_last_hour,
                limit=status.rate_limit,
                client_id=client_id,
                retry_after=retry_after,
            )

        return status

    async def get_quota_status(self, client_id: str) -> QuotaStatus:
        """Get current quota usage without throwing exceptions."""
        return await self._compute_quota_status(client_id)

    async def _compute_quota_status(self, client_id: str) -> QuotaStatus:
        """Compute current quota usage for a client."""
        # Fetch all memories for this client (use large limit to get all)
        memories = await self.storage.search_by_tags(
            tags=[f"source:{client_id}"],
            match_all=False,
            limit=1000000,  # Very large limit to get all memories
            offset=0,
        )

        # Compute memory count
        memory_count = len(memories)
        memory_usage_pct = memory_count / self.settings.max_memories

        # Compute storage size
        storage_bytes = sum(len(m.content.encode("utf-8")) for m in memories)
        storage_usage_pct = storage_bytes / self.settings.max_storage_bytes

        # Compute rate limit
        now = time.time()
        cutoff = now - self.settings.rate_limit_window_seconds
        recent_memories = [m for m in memories if m.created_at >= cutoff]
        memories_last_hour = len(recent_memories)
        rate_usage_pct = memories_last_hour / self.settings.max_memories_per_hour

        # Determine warning level
        max_usage = max(memory_usage_pct, storage_usage_pct, rate_usage_pct)
        has_warning = False
        warning_level = None

        if max_usage >= self.settings.warning_threshold_high:
            has_warning = True
            warning_level = "high"
        elif max_usage >= self.settings.warning_threshold_low:
            has_warning = True
            warning_level = "low"

        return QuotaStatus(
            client_id=client_id,
            memory_count=memory_count,
            memory_limit=self.settings.max_memories,
            memory_usage_pct=memory_usage_pct,
            storage_bytes=storage_bytes,
            storage_limit=self.settings.max_storage_bytes,
            storage_usage_pct=storage_usage_pct,
            memories_last_hour=memories_last_hour,
            rate_limit=self.settings.max_memories_per_hour,
            rate_usage_pct=rate_usage_pct,
            has_warning=has_warning,
            warning_level=warning_level,
        )
