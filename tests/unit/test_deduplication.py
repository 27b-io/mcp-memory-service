"""
Unit tests for the memory deduplication engine.

Covers:
- cosine_similarity correctness and edge cases
- fuzzy_similarity wraps difflib correctly
- find_duplicate_pairs detects near-duplicates, respects thresholds
- group_duplicate_pairs (Union-Find) produces transitive clusters
- select_canonical implements each strategy correctly
- build_duplicate_groups full pipeline integration
"""

import math

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.deduplication import (
    DuplicatePair,
    _UnionFind,
    build_duplicate_groups,
    cosine_similarity,
    find_duplicate_pairs,
    fuzzy_similarity,
    group_duplicate_pairs,
    select_canonical,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory(content: str, content_hash: str, created_at: float = 1_000_000.0, access_count: int = 0) -> Memory:
    return Memory(
        content=content,
        content_hash=content_hash,
        created_at=created_at,
        access_count=access_count,
    )


def _unit_vec(n: int, idx: int) -> list[float]:
    """Return a unit vector in dimension n with 1.0 at position idx."""
    v = [0.0] * n
    v[idx] = 1.0
    return v


def _lerp_vec(a: list[float], b: list[float], t: float) -> list[float]:
    """Linearly interpolate between two vectors and normalise."""
    v = [a[i] * (1 - t) + b[i] * t for i in range(len(a))]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_known_similarity(self):
        a = [3.0, 4.0]
        b = [4.0, 3.0]
        # dot = 12+12=24, |a|=5, |b|=5 → 24/25
        assert cosine_similarity(a, b) == pytest.approx(24 / 25)


# ---------------------------------------------------------------------------
# fuzzy_similarity
# ---------------------------------------------------------------------------


class TestFuzzySimilarity:
    def test_identical_strings(self):
        assert fuzzy_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_strings(self):
        assert fuzzy_similarity("", "") == pytest.approx(1.0)

    def test_completely_different(self):
        score = fuzzy_similarity("aaa", "zzz")
        assert score == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert fuzzy_similarity("Hello", "hello") == pytest.approx(1.0)

    def test_partial_overlap(self):
        score = fuzzy_similarity("the cat sat", "the dog sat")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_find_returns_self_for_new_node(self):
        uf = _UnionFind()
        assert uf.find("a") == "a"

    def test_union_merges_groups(self):
        uf = _UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive_union(self):
        uf = _UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_independent_groups_remain_separate(self):
        uf = _UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        assert uf.find("a") != uf.find("c")


# ---------------------------------------------------------------------------
# find_duplicate_pairs
# ---------------------------------------------------------------------------


class TestFindDuplicatePairs:
    def test_identical_embeddings_flagged(self):
        v = [1.0, 0.0, 0.0]
        memories = [_memory("content a", "h1"), _memory("content b", "h2")]
        embeddings = [v, v]
        pairs = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.95)
        assert len(pairs) == 1
        assert pairs[0].embedding_similarity == pytest.approx(1.0)

    def test_orthogonal_embeddings_not_flagged(self):
        memories = [_memory("a", "h1"), _memory("b", "h2")]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        pairs = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.95)
        assert pairs == []

    def test_same_hash_skipped(self):
        v = [1.0, 0.0]
        memories = [_memory("a", "same"), _memory("b", "same")]
        embeddings = [v, v]
        pairs = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.0)
        assert pairs == []

    def test_fuzzy_threshold_gates_pair(self):
        # Embeddings very similar but content very different
        v1 = [1.0, 0.0]
        v2 = _lerp_vec([1.0, 0.0], [0.0, 1.0], 0.01)  # similarity ≈ 0.9999
        memories = [_memory("abc", "h1"), _memory("xyz", "h2")]
        embeddings = [v1, v2]
        # Without fuzzy threshold: flagged
        pairs_no_fuzzy = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.95, fuzzy_threshold=None)
        assert len(pairs_no_fuzzy) == 1
        # With strict fuzzy threshold: filtered out
        pairs_with_fuzzy = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.95, fuzzy_threshold=0.8)
        assert len(pairs_with_fuzzy) == 0

    def test_results_sorted_descending(self):
        v0 = [1.0, 0.0, 0.0]
        v1 = [1.0, 0.0, 0.0]  # identical to v0
        v2 = _lerp_vec([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1)  # slightly less similar
        memories = [_memory("a", "h1"), _memory("b", "h2"), _memory("c", "h3")]
        embeddings = [v0, v1, v2]
        pairs = find_duplicate_pairs(memories, embeddings, similarity_threshold=0.9)
        sims = [p.embedding_similarity for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            find_duplicate_pairs([_memory("a", "h1")], [[1.0], [2.0]], similarity_threshold=0.9)


# ---------------------------------------------------------------------------
# group_duplicate_pairs
# ---------------------------------------------------------------------------


class TestGroupDuplicatePairs:
    def test_single_pair_becomes_two_member_group(self):
        pairs = [DuplicatePair("h1", "h2", 0.97)]
        groups = group_duplicate_pairs(pairs)
        assert len(groups) == 1
        assert set(groups[0]) == {"h1", "h2"}

    def test_transitive_grouping(self):
        # A-B and B-C should produce one group {A, B, C}
        pairs = [DuplicatePair("A", "B", 0.98), DuplicatePair("B", "C", 0.97)]
        groups = group_duplicate_pairs(pairs)
        assert len(groups) == 1
        assert set(groups[0]) == {"A", "B", "C"}

    def test_two_independent_clusters(self):
        pairs = [DuplicatePair("A", "B", 0.98), DuplicatePair("C", "D", 0.97)]
        groups = group_duplicate_pairs(pairs)
        assert len(groups) == 2

    def test_empty_pairs_returns_empty(self):
        assert group_duplicate_pairs([]) == []


# ---------------------------------------------------------------------------
# select_canonical
# ---------------------------------------------------------------------------


class TestSelectCanonical:
    def _make_group(self):
        return [
            _memory("oldest content", "h_oldest", created_at=1000.0, access_count=5),
            _memory("newest content", "h_newest", created_at=3000.0, access_count=2),
            _memory("most accessed", "h_accessed", created_at=2000.0, access_count=10),
        ]

    def test_keep_newest(self):
        assert select_canonical(self._make_group(), "keep_newest") == "h_newest"

    def test_keep_oldest(self):
        assert select_canonical(self._make_group(), "keep_oldest") == "h_oldest"

    def test_keep_most_accessed(self):
        assert select_canonical(self._make_group(), "keep_most_accessed") == "h_accessed"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_canonical(self._make_group(), "unknown")

    def test_empty_memories_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_canonical([], "keep_newest")


# ---------------------------------------------------------------------------
# build_duplicate_groups (integration)
# ---------------------------------------------------------------------------


class TestBuildDuplicateGroups:
    def test_no_duplicates_returns_empty(self):
        n = 4
        memories = [_memory(f"content {i}", f"h{i}") for i in range(n)]
        # Orthogonal embeddings — no duplicates
        embeddings = [_unit_vec(n, i) for i in range(n)]
        groups = build_duplicate_groups(memories, embeddings, similarity_threshold=0.95)
        assert groups == []

    def test_duplicate_pair_detected(self):
        v = [1.0, 0.0]
        memories = [
            _memory("The project uses Python", "h1", created_at=1000.0),
            _memory("The project uses Python.", "h2", created_at=2000.0),
        ]
        embeddings = [v, v]
        groups = build_duplicate_groups(memories, embeddings, similarity_threshold=0.95, strategy="keep_newest")
        assert len(groups) == 1
        assert groups[0].canonical_hash == "h2"  # newer
        assert set(groups[0].hashes) == {"h1", "h2"}

    def test_three_way_cluster(self):
        v = [1.0, 0.0, 0.0]
        memories = [
            _memory("a", "h1", created_at=1000.0),
            _memory("b", "h2", created_at=2000.0),
            _memory("c", "h3", created_at=3000.0),
        ]
        embeddings = [v, v, v]
        groups = build_duplicate_groups(memories, embeddings, similarity_threshold=0.95, strategy="keep_newest")
        assert len(groups) == 1
        assert groups[0].canonical_hash == "h3"
        assert len(groups[0].hashes) == 3

    def test_groups_sorted_by_similarity(self):
        # Two separate pairs: cluster A (dimensions 0,1), cluster B (dimensions 2,3)
        # Each pair is identical within the cluster; zero overlap between clusters.
        memories = [
            _memory("a", "h1"),
            _memory("b", "h2"),
            _memory("c", "h3"),
            _memory("d", "h4"),
        ]
        v_a = [1.0, 0.0, 0.0, 0.0]
        v_b = [0.0, 0.0, 1.0, 0.0]
        embeddings = [v_a, v_a, v_b, v_b]
        groups = build_duplicate_groups(memories, embeddings, similarity_threshold=0.9)
        assert len(groups) == 2
        assert groups[0].max_similarity >= groups[1].max_similarity

    def test_to_dict_serializable(self):
        v = [1.0, 0.0]
        memories = [_memory("a", "h1", created_at=1000.0), _memory("b", "h2", created_at=2000.0)]
        embeddings = [v, v]
        groups = build_duplicate_groups(memories, embeddings, similarity_threshold=0.9)
        d = groups[0].to_dict()
        assert "hashes" in d
        assert "canonical_hash" in d
        assert "max_similarity" in d
        assert "size" in d
        assert isinstance(d["size"], int)
