"""Tests verifying stateless MCP transport configuration."""

import inspect
import os
from unittest.mock import patch


def test_unified_server_uses_stateless_http():
    """The unified server must pass stateless_http=True for HTTP transport."""
    from mcp_memory_service.unified_server import UnifiedServer

    with patch.dict(os.environ, {"MCP_TRANSPORT_MODE": "http"}):
        server = UnifiedServer()

    assert server.mcp_enabled is True
    assert server.mcp_transport == "http"

    # Verify source contains stateless_http=True (the call is inside run_mcp_server)
    source = inspect.getsource(server.run_mcp_server)
    assert "stateless_http=True" in source


def test_standalone_main_uses_stateless_http():
    """The standalone mcp_server.main() must pass stateless_http=True."""
    from mcp_memory_service.mcp_server import main

    source = inspect.getsource(main)
    assert "stateless_http=True" in source
