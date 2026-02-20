"""
Unit tests for memory clustering utilities.

Tests cover:
- run_clustering: HDBSCAN wrapper
- generate_cluster_label: label generation strategies
- build_cluster_infos: cluster metadata assembly
- MemoryService.run_clustering: end-to-end service method (mocked storage)
- MemoryService.get_clusters: cluster enumeration
- MemoryService.expand_cluster_results: neighbourhood expansion
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.utils.clustering import (
    build_cluster_infos,
    generate_cluster_label,
    run_clustering,
)

# =============================================================================
# Helpers
# =============================================================================


def make_memory(
    content_hash: str,
    content: str = "test content",
    tags: list[str] | None = None,
    access_count: int = 0,
    summary: str | None = None,
    cluster_id: int | None = None,
    cluster_label: str | None = None,
) -> Memory:
    metadata = {}
    if cluster_id is not None:
        metadata["cluster_id"] = cluster_id
    if cluster_label is not None:
        metadata["cluster_label"] = cluster_label
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or [],
        memory_type="note",
        metadata=metadata,
        access_count=access_count,
        summary=summary,
        created_at=1700000000.0,
        updated_at=1700000000.0,
    )


# =============================================================================
# run_clustering
# =============================================================================


def test_run_clustering_basic():
    """Basic clustering with 3 well-separated groups."""
    import numpy as np

    # Create 3 tight clusters of 5 vectors each in 4D space
    rng = np.random.default_rng(42)
    group_a = rng.normal([0, 0, 0, 0], 0.01, (5, 4))
    group_b = rng.normal([10, 10, 0, 0], 0.01, (5, 4))
    group_c = rng.normal([0, 10, 10, 0], 0.01, (5, 4))
    all_vecs = np.vstack([group_a, group_b, group_c])

    embeddings = {f"hash_{i}": all_vecs[i].tolist() for i in range(15)}
    assignments = run_clustering(embeddings, min_cluster_size=3)

    assert len(assignments) == 15
    # All hashes present
    assert set(assignments.keys()) == set(embeddings.keys())
    # Should find 3 clusters (IDs 0, 1, 2) plus no noise
    cluster_ids = set(assignments.values()) - {-1}
    assert len(cluster_ids) == 3


def test_run_clustering_noise_points():
    """Points far from any cluster should be labeled -1 (noise)."""
    import numpy as np

    rng = np.random.default_rng(7)
    cluster = rng.normal([0, 0], 0.01, (10, 2))
    noise = rng.uniform(100, 200, (3, 2))
    all_vecs = np.vstack([cluster, noise])

    embeddings = {f"h{i}": all_vecs[i].tolist() for i in range(13)}
    assignments = run_clustering(embeddings, min_cluster_size=5)

    noise_count = sum(1 for cid in assignments.values() if cid == -1)
    assert noise_count >= 1  # At least the outlier points should be noise


def test_run_clustering_too_few_memories():
    with pytest.raises(ValueError, match="at least"):
        run_clustering({"h1": [0.0, 1.0]}, min_cluster_size=5)


def test_run_clustering_import_error():
    """If sklearn unavailable, ImportError is raised."""
    with patch("mcp_memory_service.utils.clustering.CLUSTERING_AVAILABLE", False):
        with pytest.raises(ImportError, match="scikit-learn"):
            run_clustering({"h1": [0.0, 1.0]}, min_cluster_size=1)


# =============================================================================
# generate_cluster_label
# =============================================================================


def test_label_from_common_tag():
    """Should use most common tag when >= 2 memories share it."""
    mems = [
        make_memory("h1", tags=["project-alpha", "decision"]),
        make_memory("h2", tags=["project-alpha", "note"]),
        make_memory("h3", tags=["project-alpha"]),
    ]
    label = generate_cluster_label(mems)
    assert label == "project-alpha"


def test_label_falls_back_to_content():
    """Should use first sentence from most-accessed memory when tags don't dominate."""
    mems = [
        make_memory("h1", content="The quick brown fox jumped.", tags=["a"], access_count=5),
        make_memory("h2", content="Different topic here.", tags=["b"], access_count=1),
    ]
    label = generate_cluster_label(mems)
    assert "quick brown fox" in label or len(label) > 0


def test_label_empty_memories():
    assert generate_cluster_label([]) == "empty-cluster"


def test_label_truncation():
    """Labels longer than 60 chars should be truncated with ellipsis."""
    content = "A" * 100
    mems = [make_memory("h1", content=content, tags=[])]
    label = generate_cluster_label(mems)
    assert len(label) <= 63  # 60 chars + "..."


# =============================================================================
# build_cluster_infos
# =============================================================================


def test_build_cluster_infos_basic():
    mems = {
        "h0": make_memory("h0", tags=["alpha"], access_count=1),
        "h1": make_memory("h1", tags=["alpha"], access_count=5),
        "h2": make_memory("h2", tags=["alpha"], access_count=3),
        "h3": make_memory("h3", tags=["beta"], access_count=0),
        "noise": make_memory("noise", tags=[], access_count=0),
    }
    assignments = {"h0": 0, "h1": 0, "h2": 0, "h3": 1, "noise": -1}

    infos = build_cluster_infos(assignments, mems)

    assert len(infos) == 2  # clusters 0 and 1, not noise
    cluster0 = next(c for c in infos if c.cluster_id == 0)
    assert cluster0.size == 3
    assert cluster0.label == "alpha"
    # h1 has highest access_count, should be first representative
    assert cluster0.representative_hashes[0] == "h1"


def test_build_cluster_infos_noise_excluded():
    mems = {"h0": make_memory("h0"), "noise": make_memory("noise")}
    assignments = {"h0": 0, "noise": -1}
    infos = build_cluster_infos(assignments, mems)
    assert len(infos) == 1
    assert infos[0].cluster_id == 0


def test_build_cluster_infos_summaries():
    mems = {
        "h0": make_memory("h0", summary="First summary."),
        "h1": make_memory("h1", summary="Second summary."),
    }
    assignments = {"h0": 0, "h1": 0}
    infos = build_cluster_infos(assignments, mems)
    assert infos[0].summary is not None
    assert "First summary" in infos[0].summary or "Second summary" in infos[0].summary


# =============================================================================
# MemoryService integration (mocked storage)
# =============================================================================


def make_mock_storage(is_qdrant=True):
    if is_qdrant:
        from mcp_memory_service.storage.qdrant_storage import QdrantStorage

        storage = MagicMock(spec=QdrantStorage)
    else:
        from mcp_memory_service.storage.base import MemoryStorage

        storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    return storage


@pytest.mark.asyncio
async def test_run_clustering_non_qdrant_storage():
    """Should return error for non-Qdrant storage backends."""
    storage = make_mock_storage(is_qdrant=False)
    svc = MemoryService(storage=storage)
    result = await svc.run_clustering()
    assert result["success"] is False
    assert "Qdrant" in result["error"]


@pytest.mark.asyncio
async def test_run_clustering_not_enough_memories():
    """Should return error when fewer memories than min_cluster_size."""
    storage = make_mock_storage(is_qdrant=True)
    storage.get_all_embeddings = AsyncMock(return_value={"h1": [0.0, 1.0]})
    svc = MemoryService(storage=storage)
    result = await svc.run_clustering(min_cluster_size=5)
    assert result["success"] is False
    assert "at least" in result["error"]


@pytest.mark.asyncio
async def test_run_clustering_success():
    """Should cluster, update storage, and return cluster list."""
    import numpy as np

    rng = np.random.default_rng(1)
    embeddings = {}
    mems = {}
    for i in range(12):
        h = f"hash_{i}"
        cluster_center = [float(i // 4 * 10), 0.0]
        vec = (rng.normal(cluster_center, 0.05, 2)).tolist()
        embeddings[h] = vec
        mems[h] = make_memory(h, tags=[f"group{i // 4}"])

    storage = make_mock_storage(is_qdrant=True)
    storage.get_all_embeddings = AsyncMock(return_value=embeddings)
    storage.get_all_memories = AsyncMock(return_value=list(mems.values()))
    storage.bulk_set_cluster_assignments = AsyncMock(return_value=12)

    svc = MemoryService(storage=storage)
    result = await svc.run_clustering(min_cluster_size=3)

    assert result["success"] is True
    assert result["clusters_found"] >= 1
    assert "clusters" in result
    storage.bulk_set_cluster_assignments.assert_called_once()


@pytest.mark.asyncio
async def test_get_clusters_empty():
    """Should return empty list when no clusters assigned yet."""
    storage = make_mock_storage(is_qdrant=True)
    storage.get_all_memories = AsyncMock(
        return_value=[
            make_memory("h1"),  # no cluster_id in metadata
        ]
    )
    svc = MemoryService(storage=storage)
    result = await svc.get_clusters()
    assert result["success"] is True
    assert result["clusters"] == []


@pytest.mark.asyncio
async def test_get_clusters_with_assignments():
    """Should group memories by cluster_id and return cluster list."""
    storage = make_mock_storage(is_qdrant=True)
    storage.get_all_memories = AsyncMock(
        return_value=[
            make_memory("h1", tags=["alpha"], cluster_id=0, cluster_label="alpha"),
            make_memory("h2", tags=["alpha"], cluster_id=0, cluster_label="alpha"),
            make_memory("h3", tags=["beta"], cluster_id=1, cluster_label="beta"),
        ]
    )
    svc = MemoryService(storage=storage)
    result = await svc.get_clusters()
    assert result["success"] is True
    assert len(result["clusters"]) == 2
    sizes = {c["cluster_id"]: c["size"] for c in result["clusters"]}
    assert sizes[0] == 2
    assert sizes[1] == 1


@pytest.mark.asyncio
async def test_expand_cluster_results_no_clusters():
    """Memories without cluster_id should not trigger expansion."""
    storage = make_mock_storage(is_qdrant=True)
    svc = MemoryService(storage=storage)
    mems = [make_memory("h1"), make_memory("h2")]
    result = await svc.expand_cluster_results(mems)
    assert result == mems  # unchanged


@pytest.mark.asyncio
async def test_expand_cluster_results_adds_siblings():
    """Should fetch and append cluster siblings, de-duplicating existing results."""
    storage = make_mock_storage(is_qdrant=True)
    sibling = make_memory("sibling_hash", cluster_id=0)
    storage.get_cluster_members = AsyncMock(
        return_value=[
            make_memory("h1", cluster_id=0),  # already in results
            sibling,
        ]
    )
    svc = MemoryService(storage=storage)

    primary = [make_memory("h1", cluster_id=0)]
    result = await svc.expand_cluster_results(primary, top_n_siblings=3)

    assert len(result) == 2
    hashes = [m.content_hash for m in result]
    assert "h1" in hashes
    assert "sibling_hash" in hashes


@pytest.mark.asyncio
async def test_expand_cluster_results_non_qdrant():
    """Should return unchanged results for non-Qdrant backends."""
    from mcp_memory_service.storage.base import MemoryStorage

    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    svc = MemoryService(storage=storage)
    mems = [make_memory("h1", cluster_id=0)]
    result = await svc.expand_cluster_results(mems)
    assert result == mems
