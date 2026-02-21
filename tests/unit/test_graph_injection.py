"""Tests for graph neighbor injection into search results."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=["test"],
        memory_type="note",
        created_at=datetime.now().timestamp(),
    )


@pytest.mark.asyncio
async def test_graph_injection_adds_new_candidates():
    """Graph injection fetches neighbor memories not in result set and injects them."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    mock_graph = AsyncMock()
    mock_graph.spreading_activation = AsyncMock(return_value={"hash_neighbor": 0.8})
    neighbor = _make_memory("hash_neighbor", "neighbor content")
    mock_storage.get_memories_batch = AsyncMock(return_value=[neighbor])

    service = MemoryService(storage=mock_storage, graph_client=mock_graph)

    existing_combined = [(_make_memory("hash_a"), 0.7, {"source": "vector"})]
    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 2
    injected_hashes = {m.content_hash for m, _, _ in result}
    assert "hash_neighbor" in injected_hashes


@pytest.mark.asyncio
async def test_graph_injection_skips_existing_results():
    """Graph injection does not duplicate memories already in results."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    mock_graph = AsyncMock()
    mock_graph.spreading_activation = AsyncMock(return_value={"hash_a": 0.9})

    service = MemoryService(storage=mock_storage, graph_client=mock_graph)

    existing_combined = [(_make_memory("hash_a"), 0.7, {"source": "vector"})]
    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 1
    mock_storage.get_memories_batch.assert_not_called()


@pytest.mark.asyncio
async def test_graph_injection_normalizes_display_scores():
    """Graph-injected entries use max(min_existing_cosine, activation) as display score."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    mock_graph = AsyncMock()
    mock_graph.spreading_activation = AsyncMock(return_value={"neighbor_hash": 0.02})

    neighbor = _make_memory("neighbor_hash", "neighbor content")
    mock_storage.get_memories_batch = AsyncMock(return_value=[neighbor])

    service = MemoryService(storage=mock_storage, graph_client=mock_graph)

    existing_mem1 = _make_memory("existing_1")
    existing_mem2 = _make_memory("existing_2")
    combined = [
        (existing_mem1, 0.8, {"source": "vector"}),
        (existing_mem2, 0.6, {"source": "vector"}),
    ]

    result = await service._inject_graph_neighbors(
        combined=combined,
        seed_hashes=["existing_1"],
        inject_limit=10,
        min_activation=0.01,
    )

    injected = [r for r in result if r[0].content_hash == "neighbor_hash"]
    assert len(injected) == 1
    display_score = injected[0][1]
    # min existing cosine is 0.6, activation is 0.02 -> max(0.6, 0.02) = 0.6
    assert display_score == 0.6, f"Expected normalized score 0.6, got {display_score}"


@pytest.mark.asyncio
async def test_graph_injection_noop_when_no_graph():
    """Graph injection is a no-op when graph layer is disabled."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    service = MemoryService(storage=mock_storage, graph_client=None)

    existing_combined = [(_make_memory("hash_a"), 0.7, {})]
    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 1
