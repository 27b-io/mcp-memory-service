"""
Unit tests for tag suggestion/autocomplete endpoint.

Tests the /api/tags/suggest endpoint logic with mocked storage.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.web.app import app
from mcp_memory_service.web.dependencies import get_storage


# Mock storage fixture
@pytest.fixture
def mock_storage():
    """Create a mock storage backend for testing."""
    storage = AsyncMock()
    storage.get_all_tags = AsyncMock()
    return storage


@pytest.fixture
def client_with_mock_storage(mock_storage):
    """Create test client with mocked storage dependency."""

    def override_get_storage():
        return mock_storage

    app.dependency_overrides[get_storage] = override_get_storage
    client = TestClient(app)
    yield client, mock_storage
    app.dependency_overrides.clear()


def test_tag_suggest_returns_all_tags_no_query(client_with_mock_storage):
    """Test that without query, all tags are returned (up to limit)."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["python", "javascript", "rust", "go", "java"]

    response = client.get("/api/tags/suggest?limit=10")

    assert response.status_code == 200
    data = response.json()

    assert data["query"] is None
    assert data["total"] == 5
    assert len(data["suggestions"]) == 5
    # Should be sorted alphabetically
    assert [s["tag"] for s in data["suggestions"]] == ["go", "java", "javascript", "python", "rust"]


def test_tag_suggest_filters_by_prefix(client_with_mock_storage):
    """Test that query parameter filters tags by prefix."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["python", "pytorch", "javascript", "java", "pandas"]

    response = client.get("/api/tags/suggest?q=py")

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "py"
    assert data["total"] == 2
    # Only tags starting with "py"
    tags = [s["tag"] for s in data["suggestions"]]
    assert tags == ["python", "pytorch"]


def test_tag_suggest_case_insensitive_match(client_with_mock_storage):
    """Test that query matching is case-insensitive."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["Python", "PyTorch", "javascript", "PYTHON_TOOLS"]

    response = client.get("/api/tags/suggest?q=PY")

    assert response.status_code == 200
    data = response.json()

    # Should match Python, PyTorch, PYTHON_TOOLS (case-insensitive)
    tags = [s["tag"] for s in data["suggestions"]]
    assert len(tags) == 3
    assert "Python" in tags
    assert "PyTorch" in tags
    assert "PYTHON_TOOLS" in tags


def test_tag_suggest_respects_limit(client_with_mock_storage):
    """Test that limit parameter restricts number of results."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["a", "b", "c", "d", "e", "f", "g"]

    response = client.get("/api/tags/suggest?limit=3")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 3  # Limited to 3
    assert len(data["suggestions"]) == 3
    # Should get first 3 alphabetically
    assert [s["tag"] for s in data["suggestions"]] == ["a", "b", "c"]


def test_tag_suggest_empty_result_with_no_matches(client_with_mock_storage):
    """Test that empty result is returned when no tags match."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["python", "javascript", "rust"]

    response = client.get("/api/tags/suggest?q=nonexistent")

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "nonexistent"
    assert data["total"] == 0
    assert data["suggestions"] == []


def test_tag_suggest_empty_storage(client_with_mock_storage):
    """Test behavior when storage has no tags."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = []

    response = client.get("/api/tags/suggest")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 0
    assert data["suggestions"] == []


def test_tag_suggest_validates_limit_too_low(client_with_mock_storage):
    """Test that limit < 1 returns validation error."""
    client, _ = client_with_mock_storage

    response = client.get("/api/tags/suggest?limit=0")

    assert response.status_code == 422  # Validation error


def test_tag_suggest_validates_limit_too_high(client_with_mock_storage):
    """Test that limit > 100 returns validation error."""
    client, _ = client_with_mock_storage

    response = client.get("/api/tags/suggest?limit=999")

    assert response.status_code == 422  # Validation error


def test_tag_suggest_handles_storage_error(client_with_mock_storage):
    """Test that storage errors are handled gracefully."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.side_effect = Exception("Database connection failed")

    response = client.get("/api/tags/suggest")

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


def test_tag_suggest_suggestion_structure(client_with_mock_storage):
    """Test that each suggestion has correct structure."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["test-tag"]

    response = client.get("/api/tags/suggest")

    assert response.status_code == 200
    data = response.json()

    suggestion = data["suggestions"][0]
    assert "tag" in suggestion
    assert suggestion["tag"] == "test-tag"
    assert "count" in suggestion
    # Count is None in current implementation (can be enhanced later)
    assert suggestion["count"] is None


def test_tag_suggest_whitespace_query(client_with_mock_storage):
    """Test that whitespace in query is handled correctly."""
    client, mock_storage = client_with_mock_storage
    mock_storage.get_all_tags.return_value = ["python", "javascript"]

    # Query with leading/trailing whitespace
    response = client.get("/api/tags/suggest?q=  py  ")

    assert response.status_code == 200
    data = response.json()

    # Whitespace should be stripped
    tags = [s["tag"] for s in data["suggestions"]]
    assert tags == ["python"]


def test_tag_suggest_default_limit(client_with_mock_storage):
    """Test that default limit is 10."""
    client, mock_storage = client_with_mock_storage
    # Create 15 tags
    many_tags = [f"tag{i:02d}" for i in range(15)]
    mock_storage.get_all_tags.return_value = many_tags

    response = client.get("/api/tags/suggest")

    assert response.status_code == 200
    data = response.json()

    # Should return only 10 (default limit)
    assert len(data["suggestions"]) == 10
    assert data["total"] == 10
