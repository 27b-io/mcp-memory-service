"""Tests for mcp_server module."""

from unittest.mock import patch


class TestExposeToolsFlag:
    """Test that three-tier tools are registered only when expose_tools=True."""

    @patch("src.mcp_memory_service.mcp_server._register_three_tier_tools")
    @patch("src.mcp_memory_service.config.settings")
    def test_registers_when_expose_tools_true(self, mock_settings, mock_register):
        """expose_tools=True must call _register_three_tier_tools."""
        mock_settings.three_tier.expose_tools = True

        from src.mcp_memory_service.mcp_server import _maybe_register_three_tier_tools, mcp

        _maybe_register_three_tier_tools()

        mock_register.assert_called_once_with(mcp)

    @patch("src.mcp_memory_service.mcp_server._register_three_tier_tools")
    @patch("src.mcp_memory_service.config.settings")
    def test_skips_when_expose_tools_false(self, mock_settings, mock_register):
        """expose_tools=False must NOT call _register_three_tier_tools."""
        mock_settings.three_tier.expose_tools = False

        from src.mcp_memory_service.mcp_server import _maybe_register_three_tier_tools

        _maybe_register_three_tier_tools()

        mock_register.assert_not_called()
