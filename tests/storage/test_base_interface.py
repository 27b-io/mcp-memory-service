"""Test storage interface signatures."""

import inspect

from mcp_memory_service.storage.base import MemoryStorage


def test_retrieve_signature_has_date_params():
    """Verify retrieve() method signature includes date range parameters."""
    sig = inspect.signature(MemoryStorage.retrieve)
    params = sig.parameters

    # Verify all expected parameters exist
    expected_params = [
        "self",
        "query",
        "n_results",
        "tags",
        "memory_type",
        "min_similarity",
        "offset",
        "start_timestamp",
        "end_timestamp",
    ]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"

    # Verify date parameters are optional (have defaults)
    assert params["start_timestamp"].default is None, "start_timestamp should default to None"
    assert params["end_timestamp"].default is None, "end_timestamp should default to None"

    # Verify they're typed as optional floats
    start_annotation = str(params["start_timestamp"].annotation)
    end_annotation = str(params["end_timestamp"].annotation)

    assert "float" in start_annotation, f"start_timestamp should be float type, got {start_annotation}"
    assert "None" in start_annotation, f"start_timestamp should allow None, got {start_annotation}"
    assert "float" in end_annotation, f"end_timestamp should be float type, got {end_annotation}"
    assert "None" in end_annotation, f"end_timestamp should allow None, got {end_annotation}"


def test_count_semantic_search_signature_has_date_params():
    """Verify count_semantic_search() method signature includes date range parameters."""
    sig = inspect.signature(MemoryStorage.count_semantic_search)
    params = sig.parameters

    # Verify all expected parameters exist
    expected_params = ["self", "query", "tags", "memory_type", "min_similarity", "start_timestamp", "end_timestamp"]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"

    # Verify date parameters are optional (have defaults)
    assert params["start_timestamp"].default is None, "start_timestamp should default to None"
    assert params["end_timestamp"].default is None, "end_timestamp should default to None"

    # Verify they're typed as optional floats
    start_annotation = str(params["start_timestamp"].annotation)
    end_annotation = str(params["end_timestamp"].annotation)

    assert "float" in start_annotation, f"start_timestamp should be float type, got {start_annotation}"
    assert "None" in start_annotation, f"start_timestamp should allow None, got {start_annotation}"
    assert "float" in end_annotation, f"end_timestamp should be float type, got {end_annotation}"
    assert "None" in end_annotation, f"end_timestamp should allow None, got {end_annotation}"


def test_recall_memory_signature_has_date_params():
    """Verify recall_memory() delegates date parameters correctly."""
    sig = inspect.signature(MemoryStorage.recall_memory)
    params = sig.parameters

    # Verify date parameters exist
    assert "start_timestamp" in params, "Missing start_timestamp parameter"
    assert "end_timestamp" in params, "Missing end_timestamp parameter"

    # Verify defaults
    assert params["start_timestamp"].default is None
    assert params["end_timestamp"].default is None


def test_search_signature_has_date_params():
    """Verify search() delegates date parameters correctly."""
    sig = inspect.signature(MemoryStorage.search)
    params = sig.parameters

    # Verify date parameters exist
    assert "start_timestamp" in params, "Missing start_timestamp parameter"
    assert "end_timestamp" in params, "Missing end_timestamp parameter"

    # Verify defaults
    assert params["start_timestamp"].default is None
    assert params["end_timestamp"].default is None
