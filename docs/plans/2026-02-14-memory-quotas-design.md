# Memory Usage Quotas Design

**Date:** 2026-02-14
**Feature:** Per-client memory storage quotas
**Status:** Design approved, ready for implementation

## Overview

Implement per-client storage quotas to prevent unbounded growth with three enforced limits:
- Total memories: 10,000 default
- Total storage: 1GB default
- Rate limit: 100 memories/hour default

Soft warnings at 80% and 90% usage, hard limits at 100%.

## Requirements

- **Scope**: Per `client_id` (from authentication)
- **Limits**: Memory count, storage size, rate limiting
- **Enforcement**: Soft warnings (80%, 90%) + hard limits (100%)
- **Storage**: In-memory computation from existing data
- **Response**: HTTP 429 with quota details when exceeded
- **Protocols**: Works for both MCP and HTTP interfaces

## Architecture

### Service-Layer Enforcement (Chosen Approach)

```
┌─────────────────────────────────────────────────────────┐
│ HTTP API / MCP Protocol                                 │
│  └─> extract client_id from auth                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ MemoryService.store_memory(content, client_id, ...)    │
│  1. QuotaService.check_quota(client_id) ───┐           │
│  2. If OK: proceed with storage            │           │
│  3. If exceeded: raise QuotaExceededError  │           │
└────────────────┬───────────────────────────┼───────────┘
                 │                           │
                 ▼                           ▼
        ┌────────────────┐        ┌─────────────────────┐
        │ MemoryStorage  │        │   QuotaService      │
        │ (Qdrant, etc.) │        │  - count_memories() │
        └────────────────┘        │  - calc_storage()   │
                                  │  - check_rate()     │
                                  └─────────────────────┘
```

**Why service-layer?**
- ✅ Works for both MCP and HTTP protocols
- ✅ Clean separation of concerns
- ✅ Only runs on write operations
- ✅ Easy to test independently
- ✅ Can inject into existing MemoryService

## Components

### QuotaSettings (src/mcp_memory_service/config.py)

```python
class QuotaSettings(BaseSettings):
    """Quota configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_QUOTA_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    enabled: bool = Field(default=False)

    # Memory count limits
    max_memories: int = Field(default=10_000, ge=1)

    # Storage size limits (bytes)
    max_storage_bytes: int = Field(default=1_073_741_824, ge=1)  # 1GB

    # Rate limits
    max_memories_per_hour: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=3600, ge=60)

    # Warning thresholds (percentage)
    warning_threshold_low: float = Field(default=0.8, ge=0.0, le=1.0)
    warning_threshold_high: float = Field(default=0.9, ge=0.0, le=1.0)
```

### QuotaExceededError (src/mcp_memory_service/utils/quota.py)

```python
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
```

### QuotaStatus (src/mcp_memory_service/utils/quota.py)

```python
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
```

### QuotaService (src/mcp_memory_service/services/quota_service.py)

```python
class QuotaService:
    """Manages quota enforcement for memory storage."""

    def __init__(
        self,
        storage: MemoryStorage,
        settings: QuotaSettings,
    ):
        self.storage = storage
        self.settings = settings
        self._rate_limit_cache: dict[str, list[float]] = {}

    async def check_quota(self, client_id: str) -> QuotaStatus:
        """
        Check all quotas and raise QuotaExceededError if any exceeded.
        Returns QuotaStatus with warning flags if approaching limits.
        """

    async def get_quota_status(self, client_id: str) -> QuotaStatus:
        """Get current quota usage without throwing exceptions."""
```

## Data Flow

### Store Memory Flow

1. **HTTP/MCP** → extract `client_id` from authentication
2. **MemoryService.store_memory()** → call `QuotaService.check_quota(client_id)`
3. **QuotaService** queries storage backend for:
   - Memory count (filter by `source:{client_id}` tag)
   - Storage size (sum of content lengths)
   - Rate limit (count memories created in last hour)
4. **If exceeded** → raise `QuotaExceededError`
5. **If 80%+** → return `QuotaStatus` with warning flag
6. **If OK** → proceed with storage

### HTTP Error Handling

```python
# In web/api/memories.py
try:
    result = await memory_service.store_memory(...)
except QuotaExceededError as e:
    return JSONResponse(
        status_code=429,
        headers={
            "X-RateLimit-Limit": str(e.limit),
            "X-RateLimit-Remaining": "0",
            "Retry-After": str(e.retry_after or 3600),
        },
        content={
            "error": "quota_exceeded",
            "quota_type": e.quota_type,
            "current": e.current,
            "limit": e.limit,
            "message": str(e),
        }
    )
```

## Quota Computation

### Memory Count
```python
# Query storage for memories tagged with client_id
memories = await storage.list_memories(tags=[f"source:{client_id}"])
count = len(memories)
```

### Storage Size
```python
# Sum content lengths for client's memories
total_bytes = sum(len(m.content.encode('utf-8')) for m in memories)
```

### Rate Limit
```python
# Count memories created in last hour
now = time.time()
cutoff = now - settings.rate_limit_window_seconds
recent = [m for m in memories if m.created_at >= cutoff]
rate_count = len(recent)
```

## Integration Points

### MemoryService Changes

```python
class MemoryService:
    def __init__(
        self,
        storage: MemoryStorage,
        graph_client: GraphClient | None = None,
        write_queue: HebbianWriteQueue | None = None,
        quota_service: QuotaService | None = None,  # NEW
    ):
        self.storage = storage
        self._graph = graph_client
        self._write_queue = write_queue
        self._quota_service = quota_service  # NEW

    async def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        client_id: str | None = None,  # NEW (from auth)
        ...
    ) -> dict[str, Any]:
        # Check quota before storing
        if self._quota_service and client_id:
            quota_status = await self._quota_service.check_quota(client_id)
            # Add warning to response if approaching limits
            if quota_status.has_warning:
                logger.warning(f"Client {client_id} at {quota_status.warning_level} quota warning")

        # Proceed with normal storage...
```

### Factory/Initialization

```python
# In shared_storage.py or factory
quota_service = None
if settings.quota.enabled:
    quota_service = QuotaService(
        storage=storage_backend,
        settings=settings.quota,
    )

memory_service = MemoryService(
    storage=storage_backend,
    quota_service=quota_service,
)
```

## Configuration

### Environment Variables

```bash
# Enable quotas
MCP_QUOTA_ENABLED=true

# Memory count limit
MCP_QUOTA_MAX_MEMORIES=10000

# Storage size limit (1GB in bytes)
MCP_QUOTA_MAX_STORAGE_BYTES=1073741824

# Rate limit (memories per hour)
MCP_QUOTA_MAX_MEMORIES_PER_HOUR=100
MCP_QUOTA_RATE_LIMIT_WINDOW_SECONDS=3600

# Warning thresholds
MCP_QUOTA_WARNING_THRESHOLD_LOW=0.8   # 80%
MCP_QUOTA_WARNING_THRESHOLD_HIGH=0.9  # 90%
```

## Testing Strategy

### Unit Tests

1. **QuotaService Tests**
   - Test quota calculation for each limit type
   - Test warning thresholds (80%, 90%)
   - Test hard limits (100%)
   - Test rate limit window sliding

2. **Integration Tests**
   - Test MemoryService integration
   - Test HTTP 429 responses
   - Test quota headers in responses
   - Test warning flags in responses

3. **Edge Cases**
   - Multiple clients (isolation)
   - Concurrent requests (race conditions)
   - Clock changes (rate limit window)
   - Anonymous clients (client_id="anonymous")

## Open Questions

- [ ] Should quotas apply to anonymous clients? (Recommend: yes, shared quota)
- [ ] Should admins have unlimited quotas? (Recommend: yes, check scope)
- [ ] Should quota status be exposed in dashboard? (Recommend: yes, new API endpoint)
- [ ] Should we track quota events for analytics? (Recommend: optional, log warnings)

## Implementation Checklist

- [ ] Create QuotaSettings in config.py
- [ ] Create QuotaService in services/quota_service.py
- [ ] Create QuotaExceededError and QuotaStatus
- [ ] Integrate QuotaService into MemoryService
- [ ] Add HTTP 429 handling in web/api/memories.py
- [ ] Add quota status endpoint (GET /api/quota)
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Add configuration examples

## Security Considerations

- **DoS Prevention**: Rate limits prevent abuse
- **Resource Protection**: Storage limits prevent disk exhaustion
- **Fair Usage**: Per-client quotas ensure fair resource allocation
- **Timing Attacks**: Use constant-time operations where possible
- **Cache Poisoning**: Don't cache quota status indefinitely (compute on demand)

## Performance Considerations

- **Query Optimization**: Use storage backend's native filtering (tags)
- **Caching**: Consider short-lived cache (30s) for quota status
- **Async Operations**: All quota checks are async (non-blocking)
- **Minimal Overhead**: Only runs on write operations, not reads
- **Batch Operations**: Handle store_batch() efficiently

## Future Enhancements

- [ ] Per-client custom quotas (override defaults)
- [ ] Quota increase requests (workflow)
- [ ] Usage analytics dashboard
- [ ] Quota alerts/notifications
- [ ] Grace period after limit (e.g., 24h at 110%)
- [ ] Automatic cleanup of old memories (LRU eviction)
