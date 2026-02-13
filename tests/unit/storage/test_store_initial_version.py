"""Tests for storing initial version when creating memories."""

import asyncio

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from mcp_memory_service.models.memory import Memory, MemoryVersion
from mcp_memory_service.storage.qdrant_storage import QdrantStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_store_memory_creates_initial_version(tmp_path):
    """store() should create version 1 snapshot in versions collection."""
    storage = QdrantStorage(
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_memories",
        storage_path=str(tmp_path / "qdrant"),
    )

    await storage.initialize()

    # Create and store a new memory
    content = "Test memory content"
    content_hash = generate_content_hash(content)

    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=["test"],
        memory_type="note",
    )

    success, message = await storage.store(memory)
    assert success, f"Failed to store memory: {message}"

    # Retrieve version 1 from versions collection
    loop = asyncio.get_event_loop()
    versions_collection_name = f"{storage.collection_name}_versions"

    # Query versions collection for this content_hash using proper Filter model
    scroll_filter = Filter(
        must=[
            FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
            FieldCondition(key="version", match=MatchValue(value=1)),
        ]
    )

    scroll_results = await loop.run_in_executor(
        None,
        lambda: storage.client.scroll(
            collection_name=versions_collection_name,
            scroll_filter=scroll_filter,
            limit=1,
        ),
    )

    points, _ = scroll_results
    assert len(points) == 1, "Should have created version 1"

    version_data = points[0].payload
    assert version_data["content_hash"] == content_hash
    assert version_data["version"] == 1
    assert version_data["content"] == "Test memory content"
    assert "timestamp" in version_data


@pytest.mark.asyncio
async def test_initial_version_matches_memory_content(tmp_path):
    """Version 1 should be exact snapshot of original content."""
    storage = QdrantStorage(
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_memories",
        storage_path=str(tmp_path / "qdrant"),
    )

    await storage.initialize()

    # Store memory with specific content
    original_content = "Original test content for version tracking"
    content_hash = generate_content_hash(original_content)

    memory = Memory(
        content=original_content,
        content_hash=content_hash,
        tags=["versioning", "test"],
        memory_type="note",
        metadata={"importance": 0.8},
    )

    success, message = await storage.store(memory)
    assert success, f"Failed to store memory: {message}"

    # Verify the memory was stored with correct version fields
    assert memory.current_version == 1
    assert memory.version_count == 1

    # Verify version 1 exists with same content
    loop = asyncio.get_event_loop()
    versions_collection_name = f"{storage.collection_name}_versions"

    scroll_filter = Filter(must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))])

    scroll_results = await loop.run_in_executor(
        None,
        lambda: storage.client.scroll(
            collection_name=versions_collection_name,
            scroll_filter=scroll_filter,
            limit=10,
        ),
    )

    points, _ = scroll_results
    assert len(points) == 1

    version = MemoryVersion.from_dict(points[0].payload)
    assert version.content == original_content
    assert version.version == 1
    assert version.content_hash == content_hash
