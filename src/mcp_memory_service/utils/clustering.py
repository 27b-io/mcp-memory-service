"""
Memory clustering utilities using HDBSCAN.

Clusters memories in vector space, assigns cluster IDs and labels,
and supports cluster-aware retrieval (neighbourhood expansion).

Design decisions:
- sklearn HDBSCAN: no new dependency (scikit-learn already installed)
- cluster_id / cluster_label stored as top-level Qdrant payload fields
- cluster_id = -1 means noise (not in any cluster)
- Labels auto-generated from most common tags or first-sentence of content
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.memory import Memory

logger = logging.getLogger(__name__)

# Lazy import guard — sklearn is required at runtime but not always installed
try:
    import numpy as np
    from sklearn.cluster import HDBSCAN

    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logger.warning("scikit-learn not available; clustering features disabled")

# How many representative memories to store per cluster
_REPRESENTATIVES_COUNT = 3


@dataclass
class ClusterInfo:
    """Metadata about a single cluster."""

    cluster_id: int
    label: str
    size: int
    representative_hashes: list[str] = field(default_factory=list)
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "size": self.size,
            "representative_hashes": self.representative_hashes,
            "summary": self.summary,
        }


@dataclass
class ClusteringResult:
    """Output of a clustering run."""

    assignments: dict[str, int]  # {content_hash: cluster_id}
    clusters: list[ClusterInfo]
    noise_count: int  # Memories with cluster_id == -1
    total_clustered: int


def run_clustering(
    embeddings: dict[str, list[float]],
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> dict[str, int]:
    """
    Run HDBSCAN clustering on a dict of embeddings.

    Args:
        embeddings: Mapping of content_hash → embedding vector
        min_cluster_size: Minimum cluster size (HDBSCAN param). Smaller = more clusters.
        min_samples: HDBSCAN min_samples. Defaults to min_cluster_size if None.

    Returns:
        Mapping of content_hash → cluster_id. -1 = noise point.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If fewer than min_cluster_size memories provided.
    """
    if not CLUSTERING_AVAILABLE:
        raise ImportError("scikit-learn is required for clustering. Install it with: pip install scikit-learn")

    if len(embeddings) < min_cluster_size:
        raise ValueError(f"Need at least {min_cluster_size} memories to cluster, got {len(embeddings)}")

    hashes = list(embeddings.keys())
    vectors = np.array([embeddings[h] for h in hashes], dtype=np.float32)

    effective_min_samples = min_samples if min_samples is not None else min_cluster_size
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        metric="euclidean",
        store_centers="medoid",
    )
    labels = clusterer.fit_predict(vectors)

    assignments = {h: int(labels[i]) for i, h in enumerate(hashes)}
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points from {len(hashes)} memories")
    return assignments


def generate_cluster_label(memories: list[Memory]) -> str:
    """
    Generate a human-readable label for a cluster.

    Strategy (in order of preference):
    1. Most common tag among cluster members (if >= 2 memories share it)
    2. First non-trivial sentence from the most-accessed member's content
    3. "cluster-N" fallback

    Args:
        memories: Memories belonging to this cluster (non-empty).

    Returns:
        A short descriptive label string.
    """
    if not memories:
        return "empty-cluster"

    # Strategy 1: Most common tag
    tag_counts: dict[str, int] = {}
    for mem in memories:
        for tag in mem.tags:
            tag = tag.strip().lower()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if tag_counts:
        top_tag = max(tag_counts, key=lambda t: tag_counts[t])
        if tag_counts[top_tag] >= max(2, len(memories) // 3):
            return top_tag

    # Strategy 2: First sentence from most-accessed memory
    best_mem = max(memories, key=lambda m: m.access_count)
    content = best_mem.content.strip()
    if content:
        # Take first sentence (up to 60 chars)
        first_line = content.splitlines()[0].strip()
        truncated = first_line[:60]
        if len(first_line) > 60:
            last_space = truncated.rfind(" ")
            if last_space > 20:
                truncated = truncated[:last_space]
            truncated += "..."
        return truncated

    return "unlabeled-cluster"


def build_cluster_infos(
    assignments: dict[str, int],
    memories_by_hash: dict[str, Memory],
) -> list[ClusterInfo]:
    """
    Build ClusterInfo objects from clustering assignments.

    Args:
        assignments: {content_hash: cluster_id}
        memories_by_hash: {content_hash: Memory} for all clustered memories

    Returns:
        List of ClusterInfo (one per cluster, excluding noise cluster -1).
    """
    # Group hashes by cluster_id
    cluster_members: dict[int, list[str]] = {}
    for content_hash, cluster_id in assignments.items():
        cluster_members.setdefault(cluster_id, []).append(content_hash)

    clusters = []
    for cluster_id, member_hashes in sorted(cluster_members.items()):
        if cluster_id == -1:
            continue  # Noise cluster excluded from ClusterInfo list

        member_memories = [memories_by_hash[h] for h in member_hashes if h in memories_by_hash]

        label = generate_cluster_label(member_memories)

        # Pick representatives: memories with highest access_count
        representatives = sorted(member_memories, key=lambda m: m.access_count, reverse=True)
        representative_hashes = [m.content_hash for m in representatives[:_REPRESENTATIVES_COUNT]]

        # Aggregate summaries into a cluster summary
        summaries = [m.summary for m in member_memories if m.summary]
        cluster_summary = "; ".join(summaries[:3]) if summaries else None

        clusters.append(
            ClusterInfo(
                cluster_id=cluster_id,
                label=label,
                size=len(member_hashes),
                representative_hashes=representative_hashes,
                summary=cluster_summary,
            )
        )

    return clusters
