"""Unit tests for quota error and status models."""

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
