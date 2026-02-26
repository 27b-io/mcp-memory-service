"""
Memory deduplication engine.

Detects near-duplicate memories using embedding cosine similarity and optional
fuzzy text matching, then groups them into clusters using Union-Find for
efficient O(n α(n)) grouping.

Design rationale:
    Exact duplicates are already prevented by Qdrant's content_hash-based upserts.
    This engine targets *semantic* duplicates: memories that express the same
    information in different words. High embedding similarity (≥0.95 by default)
    is the primary signal; difflib fuzzy ratio is a cheap secondary signal that
    requires no extra dependencies.

    Union-Find groups transitive duplicates correctly: if A≈B and B≈C, all three
    land in the same cluster rather than two overlapping pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..models.memory import Memory

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DuplicatePair:
    """A detected near-duplicate pair."""

    hash_a: str
    hash_b: str
    embedding_similarity: float
    fuzzy_similarity: float | None = None  # None if not computed

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "hash_a": self.hash_a,
            "hash_b": self.hash_b,
            "embedding_similarity": round(self.embedding_similarity, 4),
        }
        if self.fuzzy_similarity is not None:
            d["fuzzy_similarity"] = round(self.fuzzy_similarity, 4)
        return d


@dataclass
class DuplicateGroup:
    """A cluster of near-duplicate memories."""

    # All content hashes in this group
    hashes: list[str]
    # The representative pair that triggered the grouping (highest similarity)
    max_similarity: float
    # Recommended canonical hash based on selection strategy
    canonical_hash: str | None = None
    # The Memory objects (populated when requested)
    memories: list[Memory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hashes": self.hashes,
            "max_similarity": round(self.max_similarity, 4),
            "canonical_hash": self.canonical_hash,
            "size": len(self.hashes),
        }


# ---------------------------------------------------------------------------
# Cosine similarity (no numpy dependency — pure Python for portability)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def fuzzy_similarity(text_a: str, text_b: str) -> float:
    """Compute fuzzy text similarity using difflib SequenceMatcher."""
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


# ---------------------------------------------------------------------------
# Union-Find for transitive closure grouping
# ---------------------------------------------------------------------------


class _UnionFind:
    """Path-compressed Union-Find for grouping duplicate clusters."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ---------------------------------------------------------------------------
# Core deduplication logic
# ---------------------------------------------------------------------------


def find_duplicate_pairs(
    memories: list[Memory],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.95,
    fuzzy_threshold: float | None = None,
) -> list[DuplicatePair]:
    """
    Find near-duplicate pairs from a list of memories and their embeddings.

    Args:
        memories: List of Memory objects (order must match embeddings)
        embeddings: Pre-computed embedding vectors (one per memory)
        similarity_threshold: Minimum cosine similarity to flag as duplicate (0.0–1.0)
        fuzzy_threshold: Optional fuzzy text similarity gate (0.0–1.0).
            When set, a pair is only flagged if BOTH embedding and fuzzy
            similarity meet their respective thresholds. Useful for suppressing
            false positives where two short memories happen to be close in
            embedding space but differ in wording.

    Returns:
        List of DuplicatePair objects sorted by descending embedding_similarity
    """
    if len(memories) != len(embeddings):
        raise ValueError("memories and embeddings must have the same length")

    pairs: list[DuplicatePair] = []
    n = len(memories)

    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs with the same content hash (exact duplicates handled elsewhere)
            if memories[i].content_hash == memories[j].content_hash:
                continue

            emb_sim = cosine_similarity(embeddings[i], embeddings[j])
            if emb_sim < similarity_threshold:
                continue

            fuzz_sim: float | None = None
            if fuzzy_threshold is not None:
                fuzz_sim = fuzzy_similarity(memories[i].content, memories[j].content)
                if fuzz_sim < fuzzy_threshold:
                    continue

            pairs.append(
                DuplicatePair(
                    hash_a=memories[i].content_hash,
                    hash_b=memories[j].content_hash,
                    embedding_similarity=emb_sim,
                    fuzzy_similarity=fuzz_sim,
                )
            )

    pairs.sort(key=lambda p: p.embedding_similarity, reverse=True)
    return pairs


def group_duplicate_pairs(pairs: list[DuplicatePair]) -> list[list[str]]:
    """
    Group duplicate pairs into transitive clusters using Union-Find.

    Example: if A≈B and B≈C, returns [[A, B, C]] rather than [[A,B], [B,C]].

    Args:
        pairs: List of DuplicatePair objects

    Returns:
        List of clusters, each cluster is a list of content hashes.
        Only clusters with ≥2 members are returned.
    """
    uf = _UnionFind()
    for pair in pairs:
        uf.union(pair.hash_a, pair.hash_b)

    # Collect groups
    groups: dict[str, list[str]] = {}
    all_hashes = {h for pair in pairs for h in (pair.hash_a, pair.hash_b)}
    for h in all_hashes:
        root = uf.find(h)
        groups.setdefault(root, []).append(h)

    return [sorted(members) for members in groups.values() if len(members) >= 2]


def select_canonical(
    memories: list[Memory],
    strategy: str = "keep_newest",
) -> str:
    """
    Select the canonical memory hash from a duplicate group.

    Args:
        memories: Memory objects in the group
        strategy: One of:
            - "keep_newest": memory with latest created_at wins
            - "keep_oldest": memory with earliest created_at wins
            - "keep_most_accessed": memory with highest access_count wins

    Returns:
        content_hash of the chosen canonical memory

    Raises:
        ValueError: If strategy is unknown or memories list is empty
    """
    if not memories:
        raise ValueError("memories list is empty")

    if strategy == "keep_newest":
        canonical = max(memories, key=lambda m: m.created_at or 0.0)
    elif strategy == "keep_oldest":
        canonical = min(memories, key=lambda m: m.created_at or 0.0)
    elif strategy == "keep_most_accessed":
        canonical = max(memories, key=lambda m: m.access_count)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'keep_newest', 'keep_oldest', or 'keep_most_accessed'")

    return canonical.content_hash


def build_duplicate_groups(
    memories: list[Memory],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.95,
    fuzzy_threshold: float | None = None,
    strategy: str = "keep_newest",
) -> list[DuplicateGroup]:
    """
    Full pipeline: find pairs → cluster → select canonicals → return DuplicateGroups.

    Args:
        memories: Memory objects (must match embeddings order)
        embeddings: Pre-computed embedding vectors
        similarity_threshold: Cosine similarity threshold (default 0.95)
        fuzzy_threshold: Optional fuzzy text gate (default None = disabled)
        strategy: Canonical selection strategy (default "keep_newest")

    Returns:
        List of DuplicateGroup objects, each with canonical_hash set
    """
    pairs = find_duplicate_pairs(memories, embeddings, similarity_threshold, fuzzy_threshold)
    if not pairs:
        return []

    clusters = group_duplicate_pairs(pairs)

    # Index memories by hash for O(1) lookup
    mem_by_hash = {m.content_hash: m for m in memories}

    # Build max_similarity per cluster from pairs
    pair_sim: dict[tuple[str, str], float] = {}
    for p in pairs:
        key = (min(p.hash_a, p.hash_b), max(p.hash_a, p.hash_b))
        pair_sim[key] = max(pair_sim.get(key, 0.0), p.embedding_similarity)

    groups: list[DuplicateGroup] = []
    for cluster in clusters:
        cluster_mems = [mem_by_hash[h] for h in cluster if h in mem_by_hash]

        # Max similarity within cluster
        max_sim = 0.0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                key = (min(cluster[i], cluster[j]), max(cluster[i], cluster[j]))
                max_sim = max(max_sim, pair_sim.get(key, 0.0))

        canonical = select_canonical(cluster_mems, strategy)
        groups.append(
            DuplicateGroup(
                hashes=cluster,
                max_similarity=max_sim,
                canonical_hash=canonical,
                memories=cluster_mems,
            )
        )

    # Sort groups by descending max_similarity
    groups.sort(key=lambda g: g.max_similarity, reverse=True)
    return groups
