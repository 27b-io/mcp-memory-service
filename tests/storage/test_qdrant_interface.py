"""Test QdrantStorage implements the updated MemoryStorage interface correctly."""

import inspect

from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.storage.qdrant_storage import QdrantStorage


def test_qdrant_retrieve_matches_base_signature():
    """Verify QdrantStorage.retrieve() signature matches MemoryStorage.retrieve()."""
    base_sig = inspect.signature(MemoryStorage.retrieve)
    impl_sig = inspect.signature(QdrantStorage.retrieve)

    base_params = set(base_sig.parameters.keys())
    impl_params = set(impl_sig.parameters.keys())

    # Verify implementation has all base parameters
    assert base_params == impl_params, f"Parameter mismatch: base has {base_params}, impl has {impl_params}"

    # Verify date parameters exist
    assert "start_timestamp" in impl_params
    assert "end_timestamp" in impl_params

    # Verify defaults match
    assert impl_sig.parameters["start_timestamp"].default is None
    assert impl_sig.parameters["end_timestamp"].default is None


def test_qdrant_count_semantic_search_matches_base_signature():
    """Verify QdrantStorage.count_semantic_search() matches base signature."""
    base_sig = inspect.signature(MemoryStorage.count_semantic_search)
    impl_sig = inspect.signature(QdrantStorage.count_semantic_search)

    base_params = set(base_sig.parameters.keys())
    impl_params = set(impl_sig.parameters.keys())

    # Verify implementation has all base parameters
    assert base_params == impl_params, f"Parameter mismatch: base has {base_params}, impl has {impl_params}"

    # Verify date parameters exist
    assert "start_timestamp" in impl_params
    assert "end_timestamp" in impl_params

    # Verify defaults match
    assert impl_sig.parameters["start_timestamp"].default is None
    assert impl_sig.parameters["end_timestamp"].default is None
