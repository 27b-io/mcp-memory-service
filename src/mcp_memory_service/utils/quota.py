"""Quota enforcement models and exceptions."""

from dataclasses import dataclass


class QuotaExceededError(Exception):
    """Raised when a quota limit is exceeded."""

    def __init__(
        self,
        quota_type: str,  # "memory_count", "storage_size", "rate_limit"
        current: int | float,
        limit: int | float,
        client_id: str,
        retry_after: int | None = None,
    ):
        self.quota_type = quota_type
        self.current = current
        self.limit = limit
        self.client_id = client_id
        self.retry_after = retry_after

        message = f"Quota exceeded for client '{client_id}': {quota_type} limit reached ({current}/{limit})"
        if retry_after:
            message += f". Retry after {retry_after} seconds."

        super().__init__(message)


@dataclass
class QuotaStatus:
    """Current quota usage for a client."""

    client_id: str

    # Memory count
    memory_count: int
    memory_limit: int
    memory_usage_pct: float

    # Storage size
    storage_bytes: int
    storage_limit: int
    storage_usage_pct: float

    # Rate limit
    memories_last_hour: int
    rate_limit: int
    rate_usage_pct: float

    # Warning flags
    has_warning: bool
    warning_level: str | None  # "low" (80%), "high" (90%), None
