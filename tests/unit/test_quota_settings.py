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
