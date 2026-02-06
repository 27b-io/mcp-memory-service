"""
Unit tests for FalkorDB configuration.

Validates defaults, env var loading, and SecretStr handling.
"""

import os
from unittest.mock import patch

import pytest


class TestFalkorDBSettings:
    """Test FalkorDBSettings pydantic model."""

    def test_defaults(self):
        from mcp_memory_service.config import FalkorDBSettings

        cfg = FalkorDBSettings()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.password is None
        assert cfg.graph_name == "memory_graph"
        assert cfg.write_queue_key == "mcp:graph:write_queue"
        assert cfg.write_queue_batch_size == 50
        assert cfg.max_connections == 16
        assert cfg.enabled is False

    def test_env_override(self):
        from mcp_memory_service.config import FalkorDBSettings

        env = {
            "MCP_FALKORDB_HOST": "graphhost",
            "MCP_FALKORDB_PORT": "6380",
            "MCP_FALKORDB_PASSWORD": "s3cret",
            "MCP_FALKORDB_GRAPH_NAME": "custom_graph",
            "MCP_FALKORDB_ENABLED": "true",
            "MCP_FALKORDB_MAX_CONNECTIONS": "32",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = FalkorDBSettings()

        assert cfg.host == "graphhost"
        assert cfg.port == 6380
        assert cfg.password is not None
        assert cfg.password.get_secret_value() == "s3cret"
        assert cfg.graph_name == "custom_graph"
        assert cfg.enabled is True
        assert cfg.max_connections == 32

    def test_password_is_secretstr(self):
        """Password must not appear in repr/str."""
        from mcp_memory_service.config import FalkorDBSettings

        with patch.dict(os.environ, {"MCP_FALKORDB_PASSWORD": "hunter2"}, clear=False):
            cfg = FalkorDBSettings()

        assert "hunter2" not in repr(cfg)
        assert "hunter2" not in str(cfg.password)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_port_validation(self):
        from pydantic import ValidationError

        from mcp_memory_service.config import FalkorDBSettings

        with pytest.raises(ValidationError):
            FalkorDBSettings(port=0)

        with pytest.raises(ValidationError):
            FalkorDBSettings(port=70000)

    def test_batch_size_validation(self):
        from pydantic import ValidationError

        from mcp_memory_service.config import FalkorDBSettings

        with pytest.raises(ValidationError):
            FalkorDBSettings(write_queue_batch_size=0)

        with pytest.raises(ValidationError):
            FalkorDBSettings(write_queue_batch_size=501)
