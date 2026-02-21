"""Tests for batch embedding and pre-embedded vector search."""

import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_qdrant_storage():
    """Create a QdrantStorage with mocked internals for testing new methods."""
    from mcp_memory_service.storage.qdrant_storage import QdrantStorage

    storage = QdrantStorage.__new__(QdrantStorage)
    storage.collection_name = "test_collection"
    storage._vector_size = 768
    storage.embedding_model = "test-model"
    storage._model_lock = __import__("threading").Lock()
    storage.client = MagicMock()
    storage._failure_count = 0
    storage._circuit_open_until = None
    storage._failure_threshold = 5
    storage._circuit_timeout = 60
    return storage


def test_generate_embeddings_batch_signature(mock_qdrant_storage):
    """generate_embeddings_batch exists and is async."""
    assert hasattr(mock_qdrant_storage, "generate_embeddings_batch")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.generate_embeddings_batch)


def test_search_by_vector_signature(mock_qdrant_storage):
    """search_by_vector exists and is async."""
    assert hasattr(mock_qdrant_storage, "search_by_vector")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.search_by_vector)


def test_get_memories_batch_signature(mock_qdrant_storage):
    """get_memories_batch exists and is async."""
    assert hasattr(mock_qdrant_storage, "get_memories_batch")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.get_memories_batch)


@pytest.mark.asyncio
async def test_generate_embeddings_batch_calls_encode(mock_qdrant_storage):
    """Batch embedding calls model.encode with all texts at once."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
    mock_model.prompts = {"query": "search_query: "}
    mock_qdrant_storage._embedding_model_instance = mock_model

    result = await mock_qdrant_storage.generate_embeddings_batch(["query one", "query two"])

    assert len(result) == 2
    assert len(result[0]) == 768
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args
    assert len(call_args[0][0]) == 2


@pytest.mark.asyncio
async def test_search_by_vector_skips_embedding(mock_qdrant_storage):
    """search_by_vector uses provided embedding, does not call encode."""
    mock_model = MagicMock()
    mock_qdrant_storage._embedding_model_instance = mock_model
    mock_qdrant_storage.client.query_points.return_value = MagicMock(points=[])

    embedding = [0.1] * 768
    result = await mock_qdrant_storage.search_by_vector(embedding=embedding, n_results=10, min_similarity=0.0)

    assert result == []
    mock_model.encode.assert_not_called()
