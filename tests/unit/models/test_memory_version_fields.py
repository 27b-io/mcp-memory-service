"""Tests for memory version fields."""

from mcp_memory_service.models.memory import Memory


def test_memory_has_version_fields():
    """Memory model should have current_version and version_count fields."""
    memory = Memory(
        content="test content",
        content_hash="abc123",
        current_version=1,
        version_count=1,
    )

    assert memory.current_version == 1
    assert memory.version_count == 1


def test_memory_version_fields_default_to_one():
    """Version fields should default to 1 for new memories."""
    memory = Memory(
        content="test content",
        content_hash="abc123",
    )

    assert memory.current_version == 1
    assert memory.version_count == 1


def test_memory_to_dict_includes_version_fields():
    """to_dict() should include version fields."""
    memory = Memory(
        content="test",
        content_hash="abc123",
        current_version=2,
        version_count=3,
    )

    data = memory.to_dict()
    assert data["current_version"] == 2
    assert data["version_count"] == 3


def test_memory_from_dict_parses_version_fields():
    """from_dict() should parse version fields."""
    data = {
        "content": "test",
        "content_hash": "abc123",
        "current_version": 2,
        "version_count": 3,
        "tags_str": "",
    }

    memory = Memory.from_dict(data)
    assert memory.current_version == 2
    assert memory.version_count == 3
