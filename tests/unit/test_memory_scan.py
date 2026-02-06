"""Unit tests for memory_scan service method."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    return storage


@pytest.fixture
def memory_service(mock_storage):
    """Create a MemoryService with mock storage."""
    return MemoryService(storage=mock_storage)


@pytest.fixture
def sample_results():
    """Create sample MemoryQueryResult objects."""
    return [
        MemoryQueryResult(
            memory=Memory(
                content="Full content of memory one about Python debugging.",
                content_hash="hash_001",
                tags=["python", "debugging"],
                memory_type="note",
                created_at=1700000000.0,
                summary="Python debugging techniques and tips.",
            ),
            relevance_score=0.85,
        ),
        MemoryQueryResult(
            memory=Memory(
                content="Second memory with lots of detail about Qdrant vector search setup and configuration.",
                content_hash="hash_002",
                tags=["qdrant", "search"],
                memory_type="reference",
                created_at=1700001000.0,
                summary="Qdrant vector search setup and config.",
            ),
            relevance_score=0.72,
        ),
        MemoryQueryResult(
            memory=Memory(
                content="Third memory without a pre-stored summary. First sentence here.",
                content_hash="hash_003",
                tags=["test"],
                memory_type="note",
                created_at=1700002000.0,
                summary=None,  # No stored summary
            ),
            relevance_score=0.65,
        ),
    ]


class TestScanMemories:
    """Test MemoryService.scan_memories method."""

    @pytest.mark.asyncio
    async def test_scan_summary_format(self, memory_service, mock_storage, sample_results):
        """Summary format returns summary, not content."""
        mock_storage.retrieve.return_value = sample_results

        result = await memory_service.scan_memories(query="python", output_format="summary")

        assert result["count"] == 3
        assert result["format"] == "summary"
        for entry in result["results"]:
            assert "summary" in entry
            assert "content" not in entry
            assert "content_hash" in entry
            assert "relevance" in entry
            assert "tags" in entry

    @pytest.mark.asyncio
    async def test_scan_full_format(self, memory_service, mock_storage, sample_results):
        """Full format returns content, not summary."""
        mock_storage.retrieve.return_value = sample_results

        result = await memory_service.scan_memories(query="python", output_format="full")

        assert result["format"] == "full"
        for entry in result["results"]:
            assert "content" in entry
            assert "summary" not in entry

    @pytest.mark.asyncio
    async def test_scan_both_format(self, memory_service, mock_storage, sample_results):
        """Both format returns summary and content."""
        mock_storage.retrieve.return_value = sample_results

        result = await memory_service.scan_memories(query="python", output_format="both")

        assert result["format"] == "both"
        for entry in result["results"]:
            assert "summary" in entry
            assert "content" in entry

    @pytest.mark.asyncio
    async def test_scan_generates_summary_for_old_memories(self, memory_service, mock_storage, sample_results):
        """Memories without stored summary get on-the-fly extraction."""
        mock_storage.retrieve.return_value = sample_results

        result = await memory_service.scan_memories(query="test", output_format="summary")

        # Third entry has no stored summary — should be auto-generated
        third = result["results"][2]
        assert third["summary"] is not None
        assert len(third["summary"]) > 0

    @pytest.mark.asyncio
    async def test_scan_empty_results(self, memory_service, mock_storage):
        """Empty results handled gracefully."""
        mock_storage.retrieve.return_value = []

        result = await memory_service.scan_memories(query="nonexistent")

        assert result["count"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_scan_passes_parameters(self, memory_service, mock_storage):
        """Parameters are correctly passed to storage.retrieve."""
        mock_storage.retrieve.return_value = []

        await memory_service.scan_memories(
            query="test query",
            n_results=3,
            min_relevance=0.7,
        )

        mock_storage.retrieve.assert_called_once_with(
            query="test query",
            n_results=3,
            min_similarity=0.7,
        )

    @pytest.mark.asyncio
    async def test_scan_error_handling(self, memory_service, mock_storage):
        """Storage errors are caught and returned gracefully."""
        mock_storage.retrieve.side_effect = Exception("Connection failed")

        result = await memory_service.scan_memories(query="test")

        assert result["count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_scan_relevance_rounded(self, memory_service, mock_storage, sample_results):
        """Relevance scores are rounded to 4 decimal places."""
        mock_storage.retrieve.return_value = sample_results

        result = await memory_service.scan_memories(query="test")

        for entry in result["results"]:
            # Check it's a float with at most 4 decimal places
            assert isinstance(entry["relevance"], float)
            assert entry["relevance"] == round(entry["relevance"], 4)

    @pytest.mark.asyncio
    async def test_scan_invalid_format_returns_error(self, memory_service, mock_storage):
        """Invalid output_format returns an error, not silent empty results."""
        result = await memory_service.scan_memories(query="test", output_format="xml")

        assert result["count"] == 0
        assert "error" in result
        assert "Invalid format" in result["error"]
        mock_storage.retrieve.assert_not_called()
