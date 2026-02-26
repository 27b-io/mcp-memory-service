"""Memory clustering using HDBSCAN (sklearn >= 1.3) or KMeans fallback.

Runs on demand to identify natural groupings in the memory vector space.
Cluster assignments are stored as metadata on each memory for fast retrieval.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# HDBSCAN available in sklearn >= 1.3.0 (released June 2023).
# Preferred: doesn't require pre-specifying k, handles noise natively.
try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

# KMeans fallback for older sklearn installations.
try:
    from sklearn.cluster import KMeans as _KMeans

    _KMEANS_AVAILABLE = True
except ImportError:
    _KMEANS_AVAILABLE = False


@dataclass
class ClusterInfo:
    """Information about a single memory cluster."""

    cluster_id: int
    label: str
    size: int
    top_tags: list[str] = field(default_factory=list)
    member_hashes: list[str] = field(default_factory=list)
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "size": self.size,
            "top_tags": self.top_tags,
            "member_hashes": self.member_hashes,
            "summary": self.summary,
        }


def _generate_cluster_label(tags_list: list[list[str]], member_hashes: list[str]) -> str:
    """Generate a human-readable label from the most common tags in a cluster.

    Uses the top 3 most frequent tags joined by '/'. Falls back to a hash
    prefix when no tags are present in the cluster.
    """
    all_tags = [tag for tags in tags_list for tag in tags]
    if all_tags:
        common = Counter(all_tags).most_common(3)
        return "/".join(tag for tag, _ in common)
    return f"cluster-{member_hashes[0][:8]}" if member_hashes else "unnamed"


def run_clustering(
    embeddings: list[list[float]],
    content_hashes: list[str],
    tags_list: list[list[str]],
    min_cluster_size: int = 5,
) -> tuple[list[int], list[ClusterInfo]]:
    """Cluster memory embeddings using HDBSCAN or KMeans fallback.

    HDBSCAN is used when available (sklearn >= 1.3) because it:
    - Discovers cluster count automatically (no k needed)
    - Labels outliers as -1 (noise) instead of forcing them into clusters
    - Handles clusters of varying density

    KMeans fallback is used when HDBSCAN is unavailable. k is estimated
    automatically as max(2, min(n // min_cluster_size, 20)).

    Args:
        embeddings: Embedding vectors, one per memory (must be parallel to content_hashes/tags_list)
        content_hashes: Content hashes for each memory
        tags_list: Tag lists for each memory (used to generate cluster labels)
        min_cluster_size: Minimum members to form a cluster (HDBSCAN). Also
            used to estimate k for KMeans.

    Returns:
        labels: Cluster assignment per memory. -1 = HDBSCAN noise (no cluster).
        clusters: ClusterInfo list (noise points excluded).
    """
    n = len(embeddings)
    if n == 0:
        return [], []

    if n == 1:
        label = _generate_cluster_label(tags_list, content_hashes)
        return [0], [ClusterInfo(cluster_id=0, label=label, size=1, member_hashes=content_hashes)]

    X = np.array(embeddings, dtype=np.float32)

    # Effective min_cluster_size: clamp to [2, n//2] so we always get at least 2 clusters.
    effective_min = max(2, min(min_cluster_size, n // 2))

    labels: list[int]
    if _HDBSCAN_AVAILABLE:
        clusterer = _HDBSCAN(min_cluster_size=effective_min, min_samples=1)
        labels = clusterer.fit_predict(X).tolist()
        logger.info("HDBSCAN produced %d clusters from %d memories", len({lbl for lbl in labels if lbl != -1}), n)
    elif _KMEANS_AVAILABLE:
        n_clusters = max(2, min(n // max(effective_min, 3), 20))
        kmeans = _KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X).tolist()
        logger.info("KMeans produced %d clusters from %d memories", n_clusters, n)
    else:
        logger.warning("sklearn not installed — clustering unavailable; each memory assigned its own label")
        return list(range(n)), [
            ClusterInfo(cluster_id=i, label=f"mem-{h[:8]}", size=1, member_hashes=[h]) for i, h in enumerate(content_hashes)
        ]

    # Build ClusterInfo from label assignments, excluding noise (-1).
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_members[label].append(idx)

    clusters: list[ClusterInfo] = []
    for cluster_id in sorted(k for k in cluster_members if k != -1):
        indices = cluster_members[cluster_id]
        member_hashes = [content_hashes[i] for i in indices]
        member_tags = [tags_list[i] for i in indices]
        all_tags = [tag for tags in member_tags for tag in tags]
        top_tags = [tag for tag, _ in Counter(all_tags).most_common(5)]
        label = _generate_cluster_label(member_tags, member_hashes)
        clusters.append(
            ClusterInfo(
                cluster_id=cluster_id,
                label=label,
                size=len(indices),
                top_tags=top_tags,
                member_hashes=member_hashes,
            )
        )

    logger.info("Clustering complete: %d clusters, %d noise points", len(clusters), labels.count(-1))
    return labels, clusters


def build_cluster_registry(
    labels: list[int],
    content_hashes: list[str],
    clusters: list[ClusterInfo],
) -> dict[str, Any]:
    """Build a registry dict for fast cluster lookups at retrieval time.

    Args:
        labels: Per-memory cluster label (parallel to content_hashes)
        content_hashes: Content hashes in the same order as labels
        clusters: ClusterInfo list from run_clustering()

    Returns:
        Registry dict with:
        - clusters: list of cluster dicts
        - hash_to_cluster: {content_hash: cluster_id} (noise = -1)
        - cluster_members: {cluster_id: [member_hashes]}
        - noise_hashes: list of noise-labelled content hashes
    """
    hash_to_cluster: dict[str, int] = {}
    noise_hashes: list[str] = []

    for h, label in zip(content_hashes, labels):
        hash_to_cluster[h] = label
        if label == -1:
            noise_hashes.append(h)

    cluster_members: dict[int, list[str]] = {c.cluster_id: c.member_hashes for c in clusters}

    return {
        "clusters": [c.to_dict() for c in clusters],
        "hash_to_cluster": hash_to_cluster,
        "cluster_members": cluster_members,
        "noise_hashes": noise_hashes,
        "total_memories": len(content_hashes),
        "n_clusters": len(clusters),
        "n_noise": len(noise_hashes),
    }
