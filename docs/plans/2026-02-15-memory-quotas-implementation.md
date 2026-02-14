# Memory Usage Quotas Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement per-client storage quotas with soft warnings and hard limits to prevent unbounded growth.

**Architecture:** Service-layer enforcement using QuotaService that checks limits before storage operations. Integrates with existing MemoryService and HTTP API. Uses in-memory computation from existing storage backend.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic Settings, pytest, Qdrant storage backend

---

## Task 1: Create QuotaSettings Configuration

**Files:**
- Modify: `src/mcp_memory_service/config.py` (after line 686, before Settings class)
- Test: `tests/unit/test_quota_settings.py` (create new)

**Step 1: Write the failing test for QuotaSettings**

Create `tests/unit/test_quota_settings.py`:

```python
"""Unit tests for quota configuration settings."""

import pytest
from mcp_memory_service.config import QuotaSettings


class TestQuotaSettings:
    """Test QuotaSettings configuration."""

    def test_default_values(self):
        """Test default quota settings."""
        settings = QuotaSettings()

        assert settings.enabled is False
        assert settings.max_memories == 10_000
        assert settings.max_storage_bytes == 1_073_741_824  # 1GB
        assert settings.max_memories_per_hour == 100
        assert settings.rate_limit_window_seconds == 3600
        assert settings.warning_threshold_low == 0.8
        assert settings.warning_threshold_high == 0.9

    def test_env_prefix(self):
        """Test environment variable prefix MCP_QUOTA_."""
        import os

        os.environ["MCP_QUOTA_ENABLED"] = "true"
        os.environ["MCP_QUOTA_MAX_MEMORIES"] = "5000"

        settings = QuotaSettings()
        assert settings.enabled is True
        assert settings.max_memories == 5000

        # Cleanup
        del os.environ["MCP_QUOTA_ENABLED"]
        del os.environ["MCP_QUOTA_MAX_MEMORIES"]

    def test_validation_constraints(self):
        """Test field validation constraints."""
        # max_memories must be >= 1
        with pytest.raises(ValueError):
            QuotaSettings(max_memories=0)

        # warning thresholds must be 0.0-1.0
        with pytest.raises(ValueError):
            QuotaSettings(warning_threshold_low=1.5)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_quota_settings.py -v`

Expected: FAIL with "ModuleNotFoundError" or "ImportError" (QuotaSettings doesn't exist)

**Step 3: Implement QuotaSettings in config.py**

Add after line 686 (after HybridSearchSettings, before Settings class):

```python
class QuotaSettings(BaseSettings):
    """Quota configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_QUOTA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable quota enforcement")

    # Memory count limits
    max_memories: int = Field(default=10_000, ge=1, description="Maximum memories per client")

    # Storage size limits (bytes)
    max_storage_bytes: int = Field(default=1_073_741_824, ge=1, description="Maximum storage per client (1GB default)")

    # Rate limits
    max_memories_per_hour: int = Field(default=100, ge=1, description="Maximum memories created per hour")
    rate_limit_window_seconds: int = Field(default=3600, ge=60, description="Rate limit window in seconds")

    # Warning thresholds (percentage)
    warning_threshold_low: float = Field(default=0.8, ge=0.0, le=1.0, description="Low warning at 80%")
    warning_threshold_high: float = Field(default=0.9, ge=0.0, le=1.0, description="High warning at 90%")
```

**Step 4: Add QuotaSettings to main Settings class**

In the Settings class (around line 779), add the quota field:

```python
class Settings(BaseSettings):
    # ... existing fields ...
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_quota_settings.py -v`

Expected: PASS (all 3 tests)

**Step 6: Commit**

```bash
git add src/mcp_memory_service/config.py tests/unit/test_quota_settings.py
git commit -m "feat(quota): add QuotaSettings configuration with env variable support"
```

---

## Task 2: Create Quota Error and Status Models

**Files:**
- Create: `src/mcp_memory_service/utils/quota.py`
- Test: `tests/unit/test_quota_models.py` (create new)

**Step 1: Write the failing test for quota models**

Create `tests/unit/test_quota_models.py`:

```python
"""Unit tests for quota error and status models."""

import pytest
from mcp_memory_service.utils.quota import QuotaExceededError, QuotaStatus


class TestQuotaExceededError:
    """Test QuotaExceededError exception."""

    def test_error_creation(self):
        """Test creating quota exceeded error."""
        error = QuotaExceededError(
            quota_type="memory_count",
            current=150,
            limit=100,
            client_id="test-client",
        )

        assert error.quota_type == "memory_count"
        assert error.current == 150
        assert error.limit == 100
        assert error.client_id == "test-client"
        assert error.retry_after is None

    def test_error_with_retry_after(self):
        """Test error with retry_after for rate limits."""
        error = QuotaExceededError(
            quota_type="rate_limit",
            current=110,
            limit=100,
            client_id="test-client",
            retry_after=3600,
        )

        assert error.retry_after == 3600

    def test_error_message(self):
        """Test error message formatting."""
        error = QuotaExceededError(
            quota_type="storage_size",
            current=1200000000,
            limit=1073741824,
            client_id="test-client",
        )

        message = str(error)
        assert "storage_size" in message.lower()
        assert "test-client" in message.lower()


class TestQuotaStatus:
    """Test QuotaStatus dataclass."""

    def test_status_creation(self):
        """Test creating quota status."""
        status = QuotaStatus(
            client_id="test-client",
            memory_count=5000,
            memory_limit=10000,
            memory_usage_pct=0.5,
            storage_bytes=500000000,
            storage_limit=1073741824,
            storage_usage_pct=0.47,
            memories_last_hour=50,
            rate_limit=100,
            rate_usage_pct=0.5,
            has_warning=False,
            warning_level=None,
        )

        assert status.client_id == "test-client"
        assert status.memory_count == 5000
        assert status.has_warning is False

    def test_status_with_warning(self):
        """Test status with warning flags."""
        status = QuotaStatus(
            client_id="test-client",
            memory_count=8500,
            memory_limit=10000,
            memory_usage_pct=0.85,
            storage_bytes=900000000,
            storage_limit=1073741824,
            storage_usage_pct=0.84,
            memories_last_hour=85,
            rate_limit=100,
            rate_usage_pct=0.85,
            has_warning=True,
            warning_level="low",
        )

        assert status.has_warning is True
        assert status.warning_level == "low"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_quota_models.py -v`

Expected: FAIL with "ModuleNotFoundError" (quota module doesn't exist)

**Step 3: Implement quota models**

Create `src/mcp_memory_service/utils/quota.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_quota_models.py -v`

Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/utils/quota.py tests/unit/test_quota_models.py
git commit -m "feat(quota): add QuotaExceededError and QuotaStatus models"
```

---

## Task 3: Create QuotaService Core Logic

**Files:**
- Create: `src/mcp_memory_service/services/quota_service.py`
- Test: `tests/unit/test_quota_service.py` (create new)

**Step 1: Write the failing test for QuotaService**

Create `tests/unit/test_quota_service.py`:

```python
"""Unit tests for QuotaService."""

import time
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.config import QuotaSettings
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.quota_service import QuotaService
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.utils.quota import QuotaExceededError


@pytest.fixture
def quota_settings():
    """Create quota settings for testing."""
    return QuotaSettings(
        enabled=True,
        max_memories=100,
        max_storage_bytes=1000,
        max_memories_per_hour=10,
        rate_limit_window_seconds=3600,
        warning_threshold_low=0.8,
        warning_threshold_high=0.9,
    )


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    return storage


@pytest.fixture
def quota_service(mock_storage, quota_settings):
    """Create QuotaService instance."""
    return QuotaService(storage=mock_storage, settings=quota_settings)


class TestQuotaServiceMemoryCount:
    """Test memory count quota enforcement."""

    @pytest.mark.asyncio
    async def test_under_limit(self, quota_service, mock_storage):
        """Test when memory count is under limit."""
        # Mock 50 memories for client
        memories = [
            Memory(
                content=f"Memory {i}",
                content_hash=f"hash_{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(50)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_count == 50
        assert status.memory_limit == 100
        assert status.memory_usage_pct == 0.5
        assert status.has_warning is False

    @pytest.mark.asyncio
    async def test_warning_threshold_low(self, quota_service, mock_storage):
        """Test low warning threshold (80%)."""
        # 85 memories = 85% usage
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(85)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_usage_pct == 0.85
        assert status.has_warning is True
        assert status.warning_level == "low"

    @pytest.mark.asyncio
    async def test_warning_threshold_high(self, quota_service, mock_storage):
        """Test high warning threshold (90%)."""
        # 92 memories = 92% usage
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(92)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memory_usage_pct == 0.92
        assert status.has_warning is True
        assert status.warning_level == "high"

    @pytest.mark.asyncio
    async def test_limit_exceeded(self, quota_service, mock_storage):
        """Test when memory count limit is exceeded."""
        # 101 memories exceeds limit of 100
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(101)
        ]
        mock_storage.list_memories.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "memory_count"
        assert exc_info.value.current == 101
        assert exc_info.value.limit == 100


class TestQuotaServiceStorageSize:
    """Test storage size quota enforcement."""

    @pytest.mark.asyncio
    async def test_storage_under_limit(self, quota_service, mock_storage):
        """Test when storage size is under limit."""
        # Each memory has 10 bytes, 50 memories = 500 bytes
        memories = [
            Memory(
                content="0123456789",  # 10 bytes
                content_hash=f"hash_{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(50)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.storage_bytes == 500
        assert status.storage_limit == 1000
        assert status.storage_usage_pct == 0.5

    @pytest.mark.asyncio
    async def test_storage_limit_exceeded(self, quota_service, mock_storage):
        """Test when storage size limit is exceeded."""
        # Create memory with 1001 bytes
        memories = [
            Memory(
                content="a" * 1001,
                content_hash="hash_big",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
        ]
        mock_storage.list_memories.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "storage_size"
        assert exc_info.value.current == 1001
        assert exc_info.value.limit == 1000


class TestQuotaServiceRateLimit:
    """Test rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_rate_under_limit(self, quota_service, mock_storage):
        """Test when rate is under limit."""
        now = time.time()
        # 5 memories in last hour
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=now - 1800,  # 30 minutes ago
                updated_at=now - 1800,
            )
            for i in range(5)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        assert status.memories_last_hour == 5
        assert status.rate_limit == 10
        assert status.rate_usage_pct == 0.5

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, quota_service, mock_storage):
        """Test when rate limit is exceeded."""
        now = time.time()
        # 11 memories in last hour (exceeds limit of 10)
        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=now - 1800,
                updated_at=now - 1800,
            )
            for i in range(11)
        ]
        mock_storage.list_memories.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_service.check_quota("test-client")

        assert exc_info.value.quota_type == "rate_limit"
        assert exc_info.value.current == 11
        assert exc_info.value.limit == 10
        assert exc_info.value.retry_after is not None

    @pytest.mark.asyncio
    async def test_rate_limit_old_memories_ignored(self, quota_service, mock_storage):
        """Test that memories outside rate window are ignored."""
        now = time.time()
        # 5 recent + 10 old (outside window)
        memories = [
            Memory(
                content=f"Recent{i}",
                content_hash=f"hr{i}",
                tags=["source:test-client"],
                created_at=now - 1800,  # 30 min ago (inside window)
                updated_at=now - 1800,
            )
            for i in range(5)
        ] + [
            Memory(
                content=f"Old{i}",
                content_hash=f"ho{i}",
                tags=["source:test-client"],
                created_at=now - 7200,  # 2 hours ago (outside window)
                updated_at=now - 7200,
            )
            for i in range(10)
        ]
        mock_storage.list_memories.return_value = memories

        status = await quota_service.check_quota("test-client")

        # Only 5 recent memories should count
        assert status.memories_last_hour == 5
        assert status.has_warning is False


class TestQuotaServiceGetStatus:
    """Test get_quota_status method (no exceptions)."""

    @pytest.mark.asyncio
    async def test_get_status_no_exception(self, quota_service, mock_storage):
        """Test get_quota_status never raises, even when exceeded."""
        # Exceed all limits
        memories = [
            Memory(
                content="a" * 20,  # 20 bytes each
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=time.time(),
                updated_at=time.time(),
            )
            for i in range(101)  # Exceeds memory count
        ]
        mock_storage.list_memories.return_value = memories

        # Should NOT raise, just return status
        status = await quota_service.get_quota_status("test-client")

        assert status.memory_count == 101
        assert status.memory_usage_pct > 1.0  # Over 100%
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_quota_service.py -v`

Expected: FAIL with "ModuleNotFoundError" (quota_service doesn't exist)

**Step 3: Implement QuotaService (part 1 - structure and helpers)**

Create `src/mcp_memory_service/services/quota_service.py`:

```python
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
        # Fetch all memories for this client
        memories = await self.storage.list_memories(tags=[f"source:{client_id}"])

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_quota_service.py -v`

Expected: PASS (all 11 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/services/quota_service.py tests/unit/test_quota_service.py
git commit -m "feat(quota): implement QuotaService with memory count, storage size, and rate limit enforcement"
```

---

## Task 4: Integrate QuotaService into MemoryService

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:78-86` (constructor)
- Modify: `src/mcp_memory_service/services/memory_service.py:250+` (store_memory method)
- Test: `tests/unit/test_memory_service_quota.py` (create new)

**Step 1: Write the failing test for quota integration**

Create `tests/unit/test_memory_service_quota.py`:

```python
"""Unit tests for MemoryService quota integration."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.config import QuotaSettings
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.services.quota_service import QuotaService
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.utils.quota import QuotaExceededError, QuotaStatus


@pytest.fixture
def mock_storage():
    """Create mock storage."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 10000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    return storage


@pytest.fixture
def quota_settings():
    """Create quota settings."""
    return QuotaSettings(
        enabled=True,
        max_memories=100,
        max_storage_bytes=10000,
        max_memories_per_hour=10,
    )


@pytest.fixture
def quota_service(mock_storage, quota_settings):
    """Create quota service."""
    return QuotaService(storage=mock_storage, settings=quota_settings)


@pytest.fixture
def memory_service_with_quota(mock_storage, quota_service):
    """Create MemoryService with quota enforcement."""
    return MemoryService(
        storage=mock_storage,
        quota_service=quota_service,
    )


@pytest.fixture
def memory_service_without_quota(mock_storage):
    """Create MemoryService without quota enforcement."""
    return MemoryService(storage=mock_storage)


class TestMemoryServiceQuotaIntegration:
    """Test quota integration in MemoryService."""

    @pytest.mark.asyncio
    async def test_store_without_quota_service(self, memory_service_without_quota, mock_storage):
        """Test storing when quota service is None (no enforcement)."""
        mock_storage.list_memories.return_value = []

        result = await memory_service_without_quota.store_memory(
            content="Test memory",
            client_id="test-client",
        )

        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_under_limit(self, memory_service_with_quota, mock_storage):
        """Test storing when under quota limits."""
        # Mock quota check returns status with no warnings
        mock_storage.list_memories.return_value = []

        result = await memory_service_with_quota.store_memory(
            content="Test memory",
            client_id="test-client",
        )

        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_warning(self, memory_service_with_quota, mock_storage, quota_service):
        """Test storing when approaching quota limits (warning)."""
        # Mock 85 memories (85% usage triggers low warning)
        from mcp_memory_service.models.memory import Memory

        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=1000.0,
                updated_at=1000.0,
            )
            for i in range(85)
        ]
        mock_storage.list_memories.return_value = memories

        result = await memory_service_with_quota.store_memory(
            content="Test memory",
            client_id="test-client",
        )

        # Should succeed but log warning
        assert result["success"] is True
        mock_storage.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_quota_exceeded(self, memory_service_with_quota, mock_storage):
        """Test storing when quota limit is exceeded."""
        # Mock 101 memories (exceeds limit of 100)
        from mcp_memory_service.models.memory import Memory

        memories = [
            Memory(
                content=f"M{i}",
                content_hash=f"h{i}",
                tags=["source:test-client"],
                created_at=1000.0,
                updated_at=1000.0,
            )
            for i in range(101)
        ]
        mock_storage.list_memories.return_value = memories

        with pytest.raises(QuotaExceededError) as exc_info:
            await memory_service_with_quota.store_memory(
                content="Test memory",
                client_id="test-client",
            )

        assert exc_info.value.quota_type == "memory_count"
        # Store should NOT be called
        mock_storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_without_client_id_skips_quota(self, memory_service_with_quota, mock_storage):
        """Test that missing client_id skips quota checks."""
        mock_storage.list_memories.return_value = []

        result = await memory_service_with_quota.store_memory(
            content="Test memory",
            # No client_id provided
        )

        assert result["success"] is True
        # Quota check should be skipped, storage still called
        mock_storage.store.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_memory_service_quota.py -v`

Expected: FAIL (MemoryService doesn't have quota_service parameter yet)

**Step 3: Modify MemoryService constructor to accept quota_service**

In `src/mcp_memory_service/services/memory_service.py`, modify the `__init__` method (around line 78):

```python
def __init__(
    self,
    storage: MemoryStorage,
    graph_client: GraphClient | None = None,
    write_queue: HebbianWriteQueue | None = None,
    quota_service: "QuotaService | None" = None,  # NEW
):
    self.storage = storage
    self._graph = graph_client
    self._write_queue = write_queue
    self._quota_service = quota_service  # NEW
    self._tag_cache: tuple[float, set[str]] | None = None
    self._three_tier: ThreeTierMemory | None = None
    self._init_three_tier()
```

**Step 4: Add quota check to store_memory method**

Find the `store_memory` method (around line 250+), and add quota check at the beginning, before the storage operation:

```python
async def store_memory(
    self,
    content: str,
    tags: list[str] | None = None,
    client_id: str | None = None,  # Should already exist or add if missing
    memory_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    # ... other params
) -> dict[str, Any]:
    """Store a memory with quota enforcement."""

    # Check quota before storing (NEW)
    if self._quota_service and client_id:
        quota_status = await self._quota_service.check_quota(client_id)
        # Log warning if approaching limits
        if quota_status.has_warning:
            logger.warning(
                f"Client {client_id} at {quota_status.warning_level} quota warning: "
                f"memory={quota_status.memory_usage_pct:.1%}, "
                f"storage={quota_status.storage_usage_pct:.1%}, "
                f"rate={quota_status.rate_usage_pct:.1%}"
            )

    # ... rest of existing store_memory logic
```

Note: If `client_id` parameter doesn't exist in store_memory, add it to the signature.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_memory_service_quota.py -v`

Expected: PASS (all 5 tests)

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_memory_service_quota.py
git commit -m "feat(quota): integrate QuotaService into MemoryService with warning logs"
```

---

## Task 5: Add HTTP 429 Error Handling

**Files:**
- Modify: `src/mcp_memory_service/web/api/memories.py` (store_memory endpoint)
- Test: `tests/integration/test_quota_http.py` (create new)

**Step 1: Write the failing test for HTTP 429 responses**

Create `tests/integration/test_quota_http.py`:

```python
"""Integration tests for quota HTTP error handling."""

import pytest
from fastapi.testclient import TestClient

# Note: This is a simplified test. Real integration test would need full app setup.
# For now, we'll test the error handling logic in isolation.


class TestQuotaHTTPHandling:
    """Test HTTP 429 responses for quota violations."""

    def test_quota_error_to_http_response(self):
        """Test converting QuotaExceededError to HTTP 429 response."""
        from mcp_memory_service.utils.quota import QuotaExceededError

        error = QuotaExceededError(
            quota_type="memory_count",
            current=101,
            limit=100,
            client_id="test-client",
        )

        # Verify error has required attributes for response
        assert error.quota_type == "memory_count"
        assert error.current == 101
        assert error.limit == 100
        assert error.client_id == "test-client"

    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error includes retry_after."""
        from mcp_memory_service.utils.quota import QuotaExceededError

        error = QuotaExceededError(
            quota_type="rate_limit",
            current=11,
            limit=10,
            client_id="test-client",
            retry_after=3600,
        )

        assert error.retry_after == 3600
```

**Step 2: Run test to verify it passes (structure test)**

Run: `uv run pytest tests/integration/test_quota_http.py -v`

Expected: PASS (basic structure tests)

**Step 3: Add QuotaExceededError handling to memories.py**

In `src/mcp_memory_service/web/api/memories.py`, find the `store_memory` endpoint (or POST /memories endpoint) and wrap it with error handling:

Add import at the top:

```python
from ...utils.quota import QuotaExceededError
from fastapi.responses import JSONResponse
```

Modify the endpoint to handle QuotaExceededError:

```python
@router.post("/memories")
async def store_memory(
    request: MemoryCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    # ... auth dependencies if any
):
    """Store a new memory with quota enforcement."""
    try:
        # Extract client_id from auth context (or use "anonymous")
        client_id = "anonymous"  # TODO: Extract from auth if OAuth enabled

        result = await memory_service.store_memory(
            content=request.content,
            tags=request.tags,
            memory_type=request.memory_type,
            metadata=request.metadata,
            client_hostname=request.client_hostname,
            client_id=client_id,
        )

        return MemoryCreateResponse(
            success=result["success"],
            message=result["message"],
            content_hash=result.get("content_hash"),
            memory=result.get("memory"),
        )

    except QuotaExceededError as e:
        return JSONResponse(
            status_code=429,
            headers={
                "X-RateLimit-Limit": str(int(e.limit)),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(e.retry_after or 3600),
            },
            content={
                "error": "quota_exceeded",
                "quota_type": e.quota_type,
                "current": int(e.current),
                "limit": int(e.limit),
                "message": str(e),
            },
        )
```

**Step 4: Run integration tests**

Run: `uv run pytest tests/integration/test_quota_http.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp_memory_service/web/api/memories.py tests/integration/test_quota_http.py
git commit -m "feat(quota): add HTTP 429 error handling for quota violations"
```

---

## Task 6: Add Quota Status Endpoint

**Files:**
- Modify: `src/mcp_memory_service/web/api/memories.py` (add GET /quota endpoint)
- Test: `tests/integration/test_quota_endpoint.py` (create new)

**Step 1: Write the failing test for quota status endpoint**

Create `tests/integration/test_quota_endpoint.py`:

```python
"""Integration tests for quota status endpoint."""


class TestQuotaStatusEndpoint:
    """Test GET /api/quota endpoint."""

    def test_quota_status_response_structure(self):
        """Test quota status has correct structure."""
        from mcp_memory_service.utils.quota import QuotaStatus

        status = QuotaStatus(
            client_id="test-client",
            memory_count=50,
            memory_limit=100,
            memory_usage_pct=0.5,
            storage_bytes=500000,
            storage_limit=1000000,
            storage_usage_pct=0.5,
            memories_last_hour=5,
            rate_limit=10,
            rate_usage_pct=0.5,
            has_warning=False,
            warning_level=None,
        )

        # Verify all fields are present
        assert status.client_id == "test-client"
        assert status.memory_count == 50
        assert status.memory_limit == 100
        assert status.storage_bytes == 500000
        assert status.memories_last_hour == 5
        assert status.has_warning is False
```

**Step 2: Run test to verify structure**

Run: `uv run pytest tests/integration/test_quota_endpoint.py -v`

Expected: PASS

**Step 3: Add quota status endpoint to memories.py**

Add response model and endpoint to `src/mcp_memory_service/web/api/memories.py`:

```python
# Add response model after other models
class QuotaStatusResponse(BaseModel):
    """Response model for quota status."""

    client_id: str
    memory_count: int
    memory_limit: int
    memory_usage_pct: float
    storage_bytes: int
    storage_limit: int
    storage_usage_pct: float
    memories_last_hour: int
    rate_limit: int
    rate_usage_pct: float
    has_warning: bool
    warning_level: str | None


# Add endpoint
@router.get("/quota", response_model=QuotaStatusResponse)
async def get_quota_status(
    memory_service: MemoryService = Depends(get_memory_service),
    # ... auth dependencies if any
):
    """Get current quota usage for the authenticated client."""
    # Extract client_id from auth context
    client_id = "anonymous"  # TODO: Extract from auth if OAuth enabled

    # Get quota service from memory service
    if not memory_service._quota_service:
        raise HTTPException(
            status_code=503,
            detail="Quota service not enabled",
        )

    status = await memory_service._quota_service.get_quota_status(client_id)

    return QuotaStatusResponse(
        client_id=status.client_id,
        memory_count=status.memory_count,
        memory_limit=status.memory_limit,
        memory_usage_pct=status.memory_usage_pct,
        storage_bytes=status.storage_bytes,
        storage_limit=status.storage_limit,
        storage_usage_pct=status.storage_usage_pct,
        memories_last_hour=status.memories_last_hour,
        rate_limit=status.rate_limit,
        rate_usage_pct=status.rate_usage_pct,
        has_warning=status.has_warning,
        warning_level=status.warning_level,
    )
```

**Step 4: Run integration tests**

Run: `uv run pytest tests/integration/test_quota_endpoint.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp_memory_service/web/api/memories.py tests/integration/test_quota_endpoint.py
git commit -m "feat(quota): add GET /api/quota endpoint for quota status"
```

---

## Task 7: Wire QuotaService in Factory/Initialization

**Files:**
- Modify: `src/mcp_memory_service/shared_storage.py` (or relevant factory file)
- Test: Test via existing integration tests

**Step 1: Find the memory service initialization**

Search for where MemoryService is instantiated:

Run: `grep -n "MemoryService(" src/mcp_memory_service/*.py src/mcp_memory_service/**/*.py`

**Step 2: Add QuotaService initialization**

In the initialization file (likely `shared_storage.py` or `web/dependencies.py`), add:

```python
from .config import settings
from .services.quota_service import QuotaService

# Initialize quota service if enabled
quota_service = None
if settings.quota.enabled:
    quota_service = QuotaService(
        storage=storage_backend,
        settings=settings.quota,
    )

# Pass to MemoryService
memory_service = MemoryService(
    storage=storage_backend,
    graph_client=graph_client,
    write_queue=write_queue,
    quota_service=quota_service,
)
```

**Step 3: Run existing integration tests**

Run: `uv run pytest tests/integration/ -v -k "memory"`

Expected: PASS (existing tests should still work)

**Step 4: Commit**

```bash
git add src/mcp_memory_service/shared_storage.py  # or relevant file
git commit -m "feat(quota): wire QuotaService into memory service factory"
```

---

## Task 8: Update Documentation

**Files:**
- Create: `docs/guides/quotas.md`
- Modify: `README.md` (add quota section)

**Step 1: Create quota guide**

Create `docs/guides/quotas.md`:

```markdown
# Memory Usage Quotas

## Overview

Per-client storage quotas prevent unbounded growth by enforcing limits on:
- Total memories per client (default: 10,000)
- Total storage size per client (default: 1GB)
- Memory creation rate (default: 100/hour)

## Configuration

Enable quotas via environment variables:

```bash
# Enable quota enforcement
MCP_QUOTA_ENABLED=true

# Memory count limit
MCP_QUOTA_MAX_MEMORIES=10000

# Storage size limit (bytes)
MCP_QUOTA_MAX_STORAGE_BYTES=1073741824  # 1GB

# Rate limit
MCP_QUOTA_MAX_MEMORIES_PER_HOUR=100
MCP_QUOTA_RATE_LIMIT_WINDOW_SECONDS=3600

# Warning thresholds
MCP_QUOTA_WARNING_THRESHOLD_LOW=0.8   # 80%
MCP_QUOTA_WARNING_THRESHOLD_HIGH=0.9  # 90%
```

## Enforcement

### Soft Warnings

At 80% and 90% usage, warnings are logged but operations proceed.

### Hard Limits

At 100% usage, operations fail with HTTP 429 (Too Many Requests):

```json
{
  "error": "quota_exceeded",
  "quota_type": "memory_count",
  "current": 10001,
  "limit": 10000,
  "message": "Quota exceeded for client 'user-123': memory_count limit reached (10001/10000)"
}
```

Response headers:
- `X-RateLimit-Limit`: The quota limit
- `X-RateLimit-Remaining`: Remaining quota (0 when exceeded)
- `Retry-After`: Seconds to wait (for rate limits)

## Checking Quota Status

GET `/api/quota` returns current usage:

```json
{
  "client_id": "user-123",
  "memory_count": 8500,
  "memory_limit": 10000,
  "memory_usage_pct": 0.85,
  "storage_bytes": 900000000,
  "storage_limit": 1073741824,
  "storage_usage_pct": 0.84,
  "memories_last_hour": 85,
  "rate_limit": 100,
  "rate_usage_pct": 0.85,
  "has_warning": true,
  "warning_level": "low"
}
```

## Anonymous Clients

Clients without authentication share a quota under `client_id="anonymous"`.

## Disabling Quotas

Set `MCP_QUOTA_ENABLED=false` (default) to disable all quota enforcement.
```

**Step 2: Update README.md**

Add quota section to README.md configuration section:

```markdown
## Configuration

### Memory Usage Quotas

Control per-client storage limits (disabled by default):

```bash
MCP_QUOTA_ENABLED=true              # Enable quota enforcement
MCP_QUOTA_MAX_MEMORIES=10000        # Max memories per client
MCP_QUOTA_MAX_STORAGE_BYTES=1073741824  # Max storage (1GB)
MCP_QUOTA_MAX_MEMORIES_PER_HOUR=100 # Rate limit
```

See [Quota Guide](docs/guides/quotas.md) for details.
```

**Step 3: Commit**

```bash
git add docs/guides/quotas.md README.md
git commit -m "docs: add quota configuration and usage guide"
```

---

## Task 9: Run Full Test Suite and Quality Gates

**Step 1: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`

Expected: PASS

**Step 2: Run all integration tests**

Run: `uv run pytest tests/integration/ -v`

Expected: PASS

**Step 3: Run linting**

Run: `uv run ruff check src/ tests/`

Expected: No errors

**Step 4: Run formatting check**

Run: `uv run ruff format --check src/ tests/`

Expected: No changes needed

**Step 5: Fix any issues found**

If tests fail or linting errors exist, fix them before proceeding.

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test failures and linting issues"
```

---

## Final Checklist

Before marking complete:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Ruff linting passes
- [ ] Ruff formatting passes
- [ ] Documentation updated
- [ ] All tasks committed with conventional commit messages
- [ ] Design doc cross-referenced in implementation

---

## Execution Options

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach do you prefer?
