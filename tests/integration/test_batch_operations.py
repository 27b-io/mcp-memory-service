"""Integration tests for batch memory operations API."""

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.web.app import app


def test_batch_create_memories():
    """Test batch memory creation endpoint."""
    client = TestClient(app)

    # Create batch request
    batch_request = {
        "memories": [
            {"content": "Test memory 1", "tags": ["test", "batch"], "memory_type": "note"},
            {"content": "Test memory 2", "tags": ["test"], "memory_type": "fact"},
            {"content": "Test memory 3", "tags": ["batch"]},
        ]
    }

    response = client.post("/api/memories/batch", json=batch_request)

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "total" in data
    assert "successful" in data
    assert "failed" in data
    assert "results" in data

    # All should succeed
    assert data["total"] == 3
    assert data["successful"] == 3
    assert data["failed"] == 0
    assert len(data["results"]) == 3

    # Check each result
    for result in data["results"]:
        assert result["success"] is True
        assert "content_hash" in result
        assert result["content_hash"] is not None
        assert "memory" in result


def test_batch_create_with_size_limit():
    """Test batch create rejects requests over 100 items."""
    client = TestClient(app)

    # Create request with 101 memories
    batch_request = {"memories": [{"content": f"Memory {i}"} for i in range(101)]}

    response = client.post("/api/memories/batch", json=batch_request)

    assert response.status_code == 400
    assert "maximum of 100" in response.json()["detail"].lower()


def test_batch_delete_memories():
    """Test batch memory deletion endpoint."""
    client = TestClient(app)

    # First create some memories
    create_batch = {"memories": [{"content": f"Delete me {i}"} for i in range(3)]}

    create_response = client.post("/api/memories/batch", json=create_batch)
    assert create_response.status_code == 200

    # Extract content hashes
    content_hashes = [r["content_hash"] for r in create_response.json()["results"]]

    # Now delete them in batch
    delete_request = {"content_hashes": content_hashes}
    delete_response = client.delete("/api/memories/batch", json=delete_request)

    assert delete_response.status_code == 200
    data = delete_response.json()

    assert data["total"] == 3
    assert data["successful"] == 3
    assert data["failed"] == 0


def test_batch_update_memories():
    """Test batch memory update endpoint."""
    client = TestClient(app)

    # First create some memories
    create_batch = {"memories": [{"content": f"Update me {i}", "tags": ["old"]} for i in range(3)]}

    create_response = client.post("/api/memories/batch", json=create_batch)
    assert create_response.status_code == 200

    # Extract content hashes
    content_hashes = [r["content_hash"] for r in create_response.json()["results"]]

    # Now update them in batch
    update_request = {"updates": [{"content_hash": h, "tags": ["updated"], "memory_type": "note"} for h in content_hashes]}

    update_response = client.put("/api/memories/batch", json=update_request)

    assert update_response.status_code == 200
    data = update_response.json()

    assert data["total"] == 3
    assert data["successful"] == 3
    assert data["failed"] == 0

    # Verify tags were updated
    for result in data["results"]:
        assert result["success"] is True
        assert result["memory"]["tags"] == ["updated"]
        assert result["memory"]["memory_type"] == "note"


def test_batch_partial_failure():
    """Test that batch operations handle partial failures correctly."""
    client = TestClient(app)

    # Try to update non-existent memories mixed with real ones
    # First create one real memory
    create_response = client.post("/api/memories", json={"content": "Real memory", "tags": ["test"], "memory_type": "note"})
    assert create_response.status_code == 200
    real_hash = create_response.json()["content_hash"]

    # Try to update mix of real and fake hashes
    update_request = {
        "updates": [
            {"content_hash": real_hash, "tags": ["updated"]},
            {"content_hash": "fake_hash_1", "tags": ["fail"]},
            {"content_hash": "fake_hash_2", "tags": ["fail"]},
        ]
    }

    update_response = client.put("/api/memories/batch", json=update_request)

    assert update_response.status_code == 200
    data = update_response.json()

    # Should have 1 success, 2 failures
    assert data["total"] == 3
    assert data["successful"] == 1
    assert data["failed"] == 2

    # Check individual results
    assert data["results"][0]["success"] is True
    assert data["results"][1]["success"] is False
    assert data["results"][2]["success"] is False


def test_batch_empty_request():
    """Test that empty batch requests are rejected."""
    client = TestClient(app)

    # Empty create
    response = client.post("/api/memories/batch", json={"memories": []})
    assert response.status_code == 400
    assert "at least one" in response.json()["detail"].lower()

    # Empty delete
    response = client.delete("/api/memories/batch", json={"content_hashes": []})
    assert response.status_code == 400

    # Empty update
    response = client.put("/api/memories/batch", json={"updates": []})
    assert response.status_code == 400
