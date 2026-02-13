"""Tests for MemoryVersion model."""

import time

from mcp_memory_service.models.memory import MemoryVersion


def test_memory_version_creation():
    """MemoryVersion should store version snapshot data."""
    version = MemoryVersion(
        content_hash="abc123",
        version=1,
        content="test content",
        timestamp=1234567890.0,
    )

    assert version.content_hash == "abc123"
    assert version.version == 1
    assert version.content == "test content"
    assert version.timestamp == 1234567890.0


def test_memory_version_to_dict():
    """MemoryVersion.to_dict() should return all fields."""
    version = MemoryVersion(
        content_hash="abc123",
        version=2,
        content="version 2 content",
        timestamp=1234567890.0,
    )

    data = version.to_dict()
    assert data["content_hash"] == "abc123"
    assert data["version"] == 2
    assert data["content"] == "version 2 content"
    assert data["timestamp"] == 1234567890.0


def test_memory_version_from_dict():
    """MemoryVersion.from_dict() should parse payload data."""
    data = {
        "content_hash": "abc123",
        "version": 3,
        "content": "version 3 content",
        "timestamp": 1234567890.0,
    }

    version = MemoryVersion.from_dict(data)
    assert version.content_hash == "abc123"
    assert version.version == 3
    assert version.content == "version 3 content"
    assert version.timestamp == 1234567890.0


def test_memory_version_timestamp_defaults_to_now():
    """MemoryVersion timestamp should default to current time."""
    before = time.time()
    version = MemoryVersion(
        content_hash="abc123",
        version=1,
        content="test",
    )
    after = time.time()

    assert before <= version.timestamp <= after
