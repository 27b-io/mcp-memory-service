"""
Integration tests for conflicts API endpoints.

Tests the conflict resolution system that detects and resolves
memory contradictions stored as CONTRADICTS edges in the graph.
"""

import pytest
from fastapi.testclient import TestClient


def test_conflicts_module_importable():
    """Test that conflicts module can be imported without errors."""
    import mcp_memory_service.web.api.conflicts

    assert hasattr(mcp_memory_service.web.api.conflicts, "router")


def test_conflicts_routes_registered():
    """Test that conflicts API routes are registered in the app."""
    from mcp_memory_service.web.app import app

    routes = [route.path for route in app.routes]

    # Check that conflicts routes are registered
    assert any("/api/conflicts" in r for r in routes), "Conflicts routes not registered"


def test_list_conflicts_endpoint_exists():
    """Test that the list conflicts endpoint responds (even if graph is unavailable)."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)
    response = client.get("/api/conflicts/")

    # Should get either 503 (graph unavailable) or 200 (success)
    assert response.status_code in [200, 503]

    if response.status_code == 503:
        # Graph unavailable is expected in tests without FalkorDB
        data = response.json()
        assert "graph" in data["detail"].lower() or "falkordb" in data["detail"].lower()
    else:
        # If graph is available, should return valid response structure
        data = response.json()
        assert "conflicts" in data
        assert "total" in data
        assert isinstance(data["conflicts"], list)


def test_get_conflict_detail_endpoint_exists():
    """Test that the get conflict detail endpoint responds."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    # Use dummy hashes
    source_hash = "a" * 64
    target_hash = "b" * 64

    response = client.get(f"/api/conflicts/{source_hash}/{target_hash}")

    # Should get 404 (not found), 503 (graph unavailable), or 200 (success)
    assert response.status_code in [200, 404, 503]


def test_resolve_conflict_endpoint_exists():
    """Test that the resolve conflict endpoint responds."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    # Use dummy hashes
    source_hash = "a" * 64
    target_hash = "b" * 64

    response = client.post(
        f"/api/conflicts/{source_hash}/{target_hash}/resolve",
        json={"strategy": "keep_new"},
    )

    # Should get 404 (not found), 503 (graph unavailable), or 200 (success)
    assert response.status_code in [200, 404, 503]


def test_resolution_strategies_enum():
    """Test that ResolutionStrategy enum has expected values."""
    from mcp_memory_service.web.api.conflicts import ResolutionStrategy

    # Verify all expected strategies exist
    assert hasattr(ResolutionStrategy, "KEEP_NEW")
    assert hasattr(ResolutionStrategy, "KEEP_OLD")
    assert hasattr(ResolutionStrategy, "DELETE_BOTH")
    assert hasattr(ResolutionStrategy, "KEEP_BOTH")

    # Verify values
    assert ResolutionStrategy.KEEP_NEW == "keep_new"
    assert ResolutionStrategy.KEEP_OLD == "keep_old"
    assert ResolutionStrategy.DELETE_BOTH == "delete_both"
    assert ResolutionStrategy.KEEP_BOTH == "keep_both"


def test_invalid_resolution_strategy_rejected():
    """Test that invalid resolution strategies are rejected."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    source_hash = "a" * 64
    target_hash = "b" * 64

    response = client.post(
        f"/api/conflicts/{source_hash}/{target_hash}/resolve",
        json={"strategy": "invalid_strategy"},
    )

    # Should get 422 (validation error) or 503 (graph unavailable in test env)
    assert response.status_code in [422, 503]
    data = response.json()
    assert "detail" in data
