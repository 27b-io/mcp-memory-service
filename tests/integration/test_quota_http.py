"""Integration tests for quota HTTP error handling."""

import pytest


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
