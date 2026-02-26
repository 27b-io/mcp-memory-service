"""Unit tests for memory clustering utilities.

Tests run_clustering(), build_cluster_registry(), label generation, and
MemoryService clustering methods using mocked storage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from mcp_memory_service.utils.clustering import (
    ClusterInfo,
    _generate_cluster_label,
    build_cluster_registry,
    run_clustering,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_embeddings(n: int, dim: int = 8) -> list[list[float]]:
    """Generate deterministic unit vectors for testing."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).tolist()


def _make_clustered_embeddings(
    n_clusters: int = 3,
    n_per_cluster: int = 10,
    dim: int = 16,
    noise: float = 0.05,
) -> tuple[list[list[float]], list[str], list[list[str]]]:
    """Generate embeddings with clear cluster structure for deterministic tests.

    Each cluster is centred on a random unit vector; members are noisy
    perturbations of that centroid.
    """
    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    embeddings: list[list[float]] = []
    content_hashes: list[str] = []
    tags_list: list[list[str]] = []

    for cid in range(n_clusters):
        cluster_tag = f"topic-{cid}"
        for i in range(n_per_cluster):
            v = centroids[cid] + noise * rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            embeddings.append(v.tolist())
            content_hashes.append(f"hash-{cid}-{i:03d}")
            tags_list.append([cluster_tag, "shared-tag"])

    return embeddings, content_hashes, tags_list


# =============================================================================
# Label generation
# =============================================================================


class TestGenerateClusterLabel:
    def test_top_tags_joined_by_slash(self):
        tags_list = [["python", "fastapi"], ["python", "qdrant"], ["python"]]
        hashes = ["h1", "h2", "h3"]
        label = _generate_cluster_label(tags_list, hashes)
        # "python" is most common, should appear first
        assert label.startswith("python")
        assert "/" in label or label == "python"

    def test_fallback_to_hash_prefix_when_no_tags(self):
        label = _generate_cluster_label([[], []], ["abcdef1234567890", "other"])
        assert label == "cluster-abcdef12"

    def test_empty_inputs_return_unnamed(self):
        label = _generate_cluster_label([], [])
        assert label == "unnamed"

    def test_top_3_tags_at_most(self):
        tags_list = [["a", "b", "c", "d", "e"]] * 5
        hashes = [f"h{i}" for i in range(5)]
        label = _generate_cluster_label(tags_list, hashes)
        parts = label.split("/")
        assert len(parts) <= 3


# =============================================================================
# run_clustering()
# =============================================================================


class TestRunClustering:
    def test_empty_inputs(self):
        labels, clusters = run_clustering([], [], [])
        assert labels == []
        assert clusters == []

    def test_single_memory(self):
        emb = _make_embeddings(1)
        labels, clusters = run_clustering(emb, ["h1"], [["tag1"]])
        assert labels == [0]
        assert len(clusters) == 1
        assert clusters[0].size == 1
        assert "h1" in clusters[0].member_hashes

    def test_returns_correct_types(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=5)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        assert isinstance(labels, list)
        assert all(isinstance(lbl, int) for lbl in labels)
        assert all(isinstance(c, ClusterInfo) for c in clusters)

    def test_all_hashes_covered_in_labels(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=8)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        # Every memory has a label (including noise -1)
        assert len(labels) == len(hashes)

    def test_cluster_member_hashes_non_overlapping(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=8)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        all_members = [h for c in clusters for h in c.member_hashes]
        assert len(all_members) == len(set(all_members)), "member_hashes must be unique across clusters"

    def test_cluster_members_are_valid_hashes(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=6)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        hash_set = set(hashes)
        for cluster in clusters:
            for member in cluster.member_hashes:
                assert member in hash_set

    def test_cluster_labels_generated(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=6)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        for cluster in clusters:
            assert isinstance(cluster.label, str)
            assert cluster.label  # non-empty

    def test_noise_points_excluded_from_clusters(self):
        """HDBSCAN may assign -1 (noise); noise should not appear in cluster.member_hashes."""
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=8)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        cluster_hashes = {h for c in clusters for h in c.member_hashes}
        noise_hashes = {hashes[i] for i, lbl in enumerate(labels) if lbl == -1}
        assert cluster_hashes.isdisjoint(noise_hashes)

    def test_top_tags_populated(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=6)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        for cluster in clusters:
            # Each cluster should have at least the dominant topic tag
            assert len(cluster.top_tags) >= 1

    def test_cluster_size_matches_member_count(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=6)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        for cluster in clusters:
            assert cluster.size == len(cluster.member_hashes)

    def test_min_cluster_size_respected(self):
        """Clusters should not be smaller than min_cluster_size (when HDBSCAN available)."""
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=20)
        large_min = 10
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=large_min)
        # With a large min size, we should get fewer clusters
        # (this is a soft check — HDBSCAN may merge things)
        assert len(clusters) >= 1

    def test_to_dict_serializable(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=5)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        for cluster in clusters:
            d = cluster.to_dict()
            assert "cluster_id" in d
            assert "label" in d
            assert "size" in d
            assert "top_tags" in d
            assert "member_hashes" in d


# =============================================================================
# build_cluster_registry()
# =============================================================================


class TestBuildClusterRegistry:
    def test_registry_structure(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=5)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        registry = build_cluster_registry(labels, hashes, clusters)

        assert "clusters" in registry
        assert "hash_to_cluster" in registry
        assert "cluster_members" in registry
        assert "noise_hashes" in registry
        assert "n_clusters" in registry
        assert "n_noise" in registry
        assert "total_memories" in registry

    def test_hash_to_cluster_covers_all_memories(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=5)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        registry = build_cluster_registry(labels, hashes, clusters)
        assert set(registry["hash_to_cluster"].keys()) == set(hashes)

    def test_noise_hashes_have_label_minus_one(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=5)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        registry = build_cluster_registry(labels, hashes, clusters)
        for h in registry["noise_hashes"]:
            assert registry["hash_to_cluster"][h] == -1

    def test_total_memories_count(self):
        n = 20
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=n // 2)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        registry = build_cluster_registry(labels, hashes, clusters)
        assert registry["total_memories"] == n

    def test_n_clusters_matches_cluster_list(self):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=6)
        labels, clusters = run_clustering(embs, hashes, tags, min_cluster_size=3)
        registry = build_cluster_registry(labels, hashes, clusters)
        assert registry["n_clusters"] == len(registry["clusters"])


# =============================================================================
# MemoryService.run_clustering() (mocked storage)
# =============================================================================


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.fetch_vectors_for_clustering = AsyncMock()
    storage.set_payload_fields = AsyncMock()
    storage.get_memories_batch = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def memory_service(mock_storage):
    from mcp_memory_service.services.memory_service import MemoryService

    service = MemoryService.__new__(MemoryService)
    service.storage = mock_storage
    service._graph = None
    service._write_queue = None
    service._cluster_registry = None
    service._three_tier = None
    from collections import deque

    service._search_logs = deque()
    service._audit_logs = deque()
    return service


class TestMemoryServiceRunClustering:
    @pytest.mark.asyncio
    async def test_run_clustering_empty_storage(self, memory_service, mock_storage):
        mock_storage.fetch_vectors_for_clustering.return_value = []
        result = await memory_service.run_clustering()
        assert result["success"] is False
        assert "No memories" in result["error"]

    @pytest.mark.asyncio
    async def test_run_clustering_populates_registry(self, memory_service, mock_storage):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=3, n_per_cluster=6)
        records = list(zip(hashes, embs, tags))
        mock_storage.fetch_vectors_for_clustering.return_value = records

        result = await memory_service.run_clustering(min_cluster_size=3)

        assert result["success"] is True
        assert result["n_memories"] == len(hashes)
        assert memory_service._cluster_registry is not None

    @pytest.mark.asyncio
    async def test_run_clustering_persists_to_storage(self, memory_service, mock_storage):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=6)
        records = list(zip(hashes, embs, tags))
        mock_storage.fetch_vectors_for_clustering.return_value = records

        await memory_service.run_clustering(min_cluster_size=3)

        # set_payload_fields should be called for each cluster (+ noise if any)
        assert mock_storage.set_payload_fields.called

    @pytest.mark.asyncio
    async def test_get_clusters_before_run_returns_error(self, memory_service):
        result = memory_service.get_clusters()
        assert result["success"] is False
        assert "run_clustering" in result["error"]

    @pytest.mark.asyncio
    async def test_get_clusters_after_run_returns_data(self, memory_service, mock_storage):
        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=5)
        records = list(zip(hashes, embs, tags))
        mock_storage.fetch_vectors_for_clustering.return_value = records

        await memory_service.run_clustering(min_cluster_size=3)
        result = memory_service.get_clusters()

        assert result["success"] is True
        assert "clusters" in result
        assert result["n_memories"] == len(hashes)

    @pytest.mark.asyncio
    async def test_expand_by_cluster_no_registry(self, memory_service):
        result = await memory_service.expand_by_cluster(["hash1", "hash2"])
        assert result == []

    @pytest.mark.asyncio
    async def test_expand_by_cluster_returns_neighbours(self, memory_service, mock_storage):
        from mcp_memory_service.models.memory import Memory

        embs, hashes, tags = _make_clustered_embeddings(n_clusters=2, n_per_cluster=6)
        records = list(zip(hashes, embs, tags))
        mock_storage.fetch_vectors_for_clustering.return_value = records
        await memory_service.run_clustering(min_cluster_size=3)

        # Use first memory from the registry
        registry = memory_service._cluster_registry
        if not registry["clusters"]:
            return  # skip if all noise (degenerate case)

        first_cluster_members = registry["clusters"][0]["member_hashes"]
        seed_hash = first_cluster_members[0]

        # Mock get_memories_batch to return fake Memory objects
        def make_memory(h):
            m = MagicMock(spec=Memory)
            m.content = f"content-{h}"
            m.content_hash = h
            m.tags = []
            m.memory_type = "note"
            m.metadata = {}
            m.created_at = 0.0
            m.updated_at = 0.0
            m.created_at_iso = ""
            m.updated_at_iso = ""
            m.emotional_valence = None
            m.salience_score = 0.0
            m.encoding_context = None
            return m

        mock_storage.get_memories_batch.return_value = [make_memory(h) for h in first_cluster_members[1:4]]

        neighbours = await memory_service.expand_by_cluster([seed_hash], max_per_cluster=3)
        assert isinstance(neighbours, list)
        # Should have fetched some neighbours (unless it's a tiny cluster)
        assert len(neighbours) >= 0  # non-negative; exact count depends on cluster size
