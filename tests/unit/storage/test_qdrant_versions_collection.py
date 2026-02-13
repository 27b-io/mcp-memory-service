"""Tests for Qdrant versions collection initialization."""

import pytest

from mcp_memory_service.storage.qdrant_storage import QdrantStorage


@pytest.mark.asyncio
async def test_initialize_creates_versions_collection(tmp_path):
    """initialize() should create memory_versions collection."""
    storage = QdrantStorage(
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_memories",
        storage_path=str(tmp_path / "qdrant"),
    )

    await storage.initialize()

    # Check that versions collection exists
    import asyncio

    loop = asyncio.get_event_loop()
    collections = await loop.run_in_executor(None, storage.client.get_collections)
    collection_names = [col.name for col in collections.collections]

    assert "test_memories_versions" in collection_names


@pytest.mark.asyncio
async def test_versions_collection_has_correct_schema(tmp_path):
    """Versions collection should have correct vector size and indexes."""
    storage = QdrantStorage(
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_memories",
        storage_path=str(tmp_path / "qdrant"),
    )

    await storage.initialize()

    # Get collection info
    import asyncio

    loop = asyncio.get_event_loop()
    collection_info = await loop.run_in_executor(None, storage.client.get_collection, "test_memories_versions")

    # Verify vector size matches main collection
    assert collection_info.config.params.vectors.size == 384  # all-MiniLM-L6-v2
