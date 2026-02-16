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
