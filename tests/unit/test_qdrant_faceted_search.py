"""Unit tests for QdrantStorage.faceted_search() implementation."""

import hashlib
import random
import shutil
from datetime import datetime

import pytest
from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage
from src.mcp_memory_service.utils.hashing import generate_content_hash


def create_deterministic_embedding(text: str, vector_size: int = 384) -> list[float]:
    """Create a deterministic embedding based on text content."""
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    seed = hash_val % (2**32)
    rng = random.Random(seed)
    return [rng.random() * 2 - 1 for _ in range(vector_size)]


def create_memory(
    content: str, tags: list[str] | None = None, memory_type: str | None = None, created_at: float | None = None
) -> Memory:
    """Helper to create a Memory object with proper content_hash."""
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags or [],
        memory_type=memory_type,
        created_at=created_at,
    )


class TestQdrantFacetedSearch:
    """Integration tests for faceted search with real Qdrant embedded instance."""

    @pytest.fixture(scope="function")
    async def qdrant_storage(self, tmp_path, monkeypatch):
        """Create real Qdrant storage for integration tests with mocked embeddings."""
        storage_path = tmp_path / "qdrant"
        storage_path.mkdir(exist_ok=True)

        storage = QdrantStorage(
            storage_path=str(storage_path), embedding_model="all-MiniLM-L6-v2", collection_name="test_memories"
        )

        # Mock the embedding generation to avoid downloading models
        def mock_embedding(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)

        async def mock_query_embedding(query: str) -> list[float]:
            return create_deterministic_embedding(query, vector_size=384)

        # Create a mock embedding service
        class MockEmbeddingService:
            def encode(self, texts, convert_to_numpy=False):
                import numpy as np

                result = [create_deterministic_embedding(text, vector_size=384) for text in texts]
                if convert_to_numpy:
                    return np.array(result)
                return result

        monkeypatch.setattr(storage, "_generate_embedding", mock_embedding)
        monkeypatch.setattr(storage, "_generate_query_embedding", mock_query_embedding)
        storage.embedding_service = MockEmbeddingService()

        await storage.initialize()
        yield storage
        await storage.close()
        # Clean up temp directory
        if storage_path.exists():
            shutil.rmtree(storage_path, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_faceted_search_by_tags_or_logic(self, qdrant_storage):
        """Test faceted search with tags using OR logic (any tag matches)."""
        # Arrange: Store memories with different tags
        memories = [
            create_memory("Python testing", tags=["python", "testing"], memory_type="note"),
            create_memory("Python web dev", tags=["python", "web"], memory_type="note"),
            create_memory("JavaScript testing", tags=["javascript", "testing"], memory_type="note"),
            create_memory("Unrelated content", tags=["other"], memory_type="note"),
        ]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Search for memories with tags ["python", "javascript"]
        result = await qdrant_storage.faceted_search(tags=["python", "javascript"], tag_match_all=False, page_size=10)

        # Assert: Should match first 3 memories (any tag matches)
        assert result["total"] == 3
        assert len(result["memories"]) == 3
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert result["has_more"] is False
        assert result["total_pages"] == 1

        # Verify content matches
        contents = {m.content for m in result["memories"]}
        assert "Python testing" in contents
        assert "Python web dev" in contents
        assert "JavaScript testing" in contents
        assert "Unrelated content" not in contents

    @pytest.mark.asyncio
    async def test_faceted_search_by_tags_and_logic(self, qdrant_storage):
        """Test faceted search with tags using AND logic (all tags must match)."""
        # Arrange: Store memories with different tag combinations
        memories = [
            create_memory("Python and testing", tags=["python", "testing"], memory_type="note"),
            create_memory("Python only", tags=["python"], memory_type="note"),
            create_memory("Testing only", tags=["testing"], memory_type="note"),
            create_memory("Python web testing", tags=["python", "web", "testing"], memory_type="note"),
        ]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Search for memories with ALL tags ["python", "testing"]
        result = await qdrant_storage.faceted_search(tags=["python", "testing"], tag_match_all=True, page_size=10)

        # Assert: Should match only memories with both tags
        assert result["total"] == 2
        assert len(result["memories"]) == 2

        # Verify content matches
        contents = {m.content for m in result["memories"]}
        assert "Python and testing" in contents
        assert "Python web testing" in contents
        assert "Python only" not in contents
        assert "Testing only" not in contents

    @pytest.mark.asyncio
    async def test_faceted_search_by_memory_type(self, qdrant_storage):
        """Test faceted search filtering by memory type."""
        # Arrange: Store memories with different types
        memories = [
            create_memory("Note 1", tags=["test"], memory_type="note"),
            create_memory("Decision 1", tags=["test"], memory_type="decision"),
            create_memory("Note 2", tags=["test"], memory_type="note"),
            create_memory("Task 1", tags=["test"], memory_type="task"),
        ]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Search for "note" type only
        result = await qdrant_storage.faceted_search(memory_type="note", page_size=10)

        # Assert: Should match only note type memories
        assert result["total"] == 2
        assert len(result["memories"]) == 2

        # Verify all are note type
        for memory in result["memories"]:
            assert memory.memory_type == "note"

    @pytest.mark.asyncio
    async def test_faceted_search_by_date_range(self, qdrant_storage):
        """Test faceted search filtering by date range."""
        # Arrange: Store memories with different timestamps
        base_time = datetime(2024, 1, 1).timestamp()
        memories = [
            create_memory("Old memory", tags=["test"], created_at=base_time),
            create_memory("Middle memory", tags=["test"], created_at=base_time + 86400 * 5),  # 5 days later
            create_memory("Recent memory", tags=["test"], created_at=base_time + 86400 * 10),  # 10 days later
            create_memory("Newest memory", tags=["test"], created_at=base_time + 86400 * 15),  # 15 days later
        ]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Search for memories from day 4 to day 11
        date_from = base_time + 86400 * 4
        date_to = base_time + 86400 * 11
        result = await qdrant_storage.faceted_search(date_from=date_from, date_to=date_to, page_size=10)

        # Assert: Should match middle and recent memories
        assert result["total"] == 2
        assert len(result["memories"]) == 2

        # Verify content matches
        contents = {m.content for m in result["memories"]}
        assert "Middle memory" in contents
        assert "Recent memory" in contents
        assert "Old memory" not in contents
        assert "Newest memory" not in contents

    @pytest.mark.asyncio
    async def test_faceted_search_combined_filters(self, qdrant_storage):
        """Test faceted search with multiple filters combined."""
        # Arrange: Store memories with various attributes
        base_time = datetime(2024, 1, 1).timestamp()
        memories = [
            create_memory("Python note old", tags=["python"], memory_type="note", created_at=base_time),
            create_memory("Python note new", tags=["python"], memory_type="note", created_at=base_time + 86400 * 5),
            create_memory("Python decision new", tags=["python"], memory_type="decision", created_at=base_time + 86400 * 5),
            create_memory("JavaScript note new", tags=["javascript"], memory_type="note", created_at=base_time + 86400 * 5),
        ]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Search for python notes from day 4 onwards
        date_from = base_time + 86400 * 4
        result = await qdrant_storage.faceted_search(
            tags=["python"], tag_match_all=False, memory_type="note", date_from=date_from, page_size=10
        )

        # Assert: Should match only "Python note new"
        assert result["total"] == 1
        assert len(result["memories"]) == 1
        assert result["memories"][0].content == "Python note new"

    @pytest.mark.asyncio
    async def test_faceted_search_pagination(self, qdrant_storage):
        """Test faceted search pagination works correctly."""
        # Arrange: Store 15 memories with same tag
        memories = [create_memory(f"Memory {i}", tags=["test"], memory_type="note") for i in range(15)]

        for memory in memories:
            await qdrant_storage.store(memory)

        # Act: Page 1 (first 10)
        result_page1 = await qdrant_storage.faceted_search(tags=["test"], page=1, page_size=10)

        # Assert page 1
        assert result_page1["total"] == 15
        assert len(result_page1["memories"]) == 10
        assert result_page1["page"] == 1
        assert result_page1["page_size"] == 10
        assert result_page1["has_more"] is True
        assert result_page1["total_pages"] == 2

        # Act: Page 2 (remaining 5)
        result_page2 = await qdrant_storage.faceted_search(tags=["test"], page=2, page_size=10)

        # Assert page 2
        assert result_page2["total"] == 15
        assert len(result_page2["memories"]) == 5
        assert result_page2["page"] == 2
        assert result_page2["page_size"] == 10
        assert result_page2["has_more"] is False
        assert result_page2["total_pages"] == 2

        # Verify no overlap between pages
        page1_hashes = {m.content_hash for m in result_page1["memories"]}
        page2_hashes = {m.content_hash for m in result_page2["memories"]}
        assert len(page1_hashes & page2_hashes) == 0
