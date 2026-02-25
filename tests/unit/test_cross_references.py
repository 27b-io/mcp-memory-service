"""
Unit tests for memory cross-referencing utilities.

Covers:
- Candidate filtering: exclude same hash, exclude known contradictions
- Empty input handling
- All-excluded scenario
- Order preservation
"""

from unittest.mock import MagicMock

from mcp_memory_service.utils.cross_references import filter_cross_reference_candidates


def _make_result(content_hash: str) -> MagicMock:
    """Build a minimal MemoryQueryResult-compatible mock."""
    result = MagicMock()
    result.memory.content_hash = content_hash
    return result


class TestFilterCrossReferenceCandidates:
    def test_returns_distinct_related_hashes(self):
        similar = [_make_result("aaa"), _make_result("bbb"), _make_result("ccc")]
        result = filter_cross_reference_candidates("new_hash", similar)
        assert result == ["aaa", "bbb", "ccc"]

    def test_excludes_same_hash(self):
        """The new memory itself should never appear as a cross-reference."""
        similar = [_make_result("self_hash"), _make_result("other")]
        result = filter_cross_reference_candidates("self_hash", similar)
        assert result == ["other"]

    def test_excludes_known_contradictions(self):
        """Hashes already tagged as contradictions should not get RELATES_TO edges."""
        similar = [_make_result("contradiction"), _make_result("related")]
        result = filter_cross_reference_candidates("new", similar, exclude_hashes={"contradiction"})
        assert result == ["related"]

    def test_excludes_same_hash_and_contradictions(self):
        similar = [_make_result("self"), _make_result("bad"), _make_result("good")]
        result = filter_cross_reference_candidates("self", similar, exclude_hashes={"bad"})
        assert result == ["good"]

    def test_empty_similar_returns_empty(self):
        result = filter_cross_reference_candidates("new", [])
        assert result == []

    def test_all_excluded_returns_empty(self):
        similar = [_make_result("a"), _make_result("b")]
        result = filter_cross_reference_candidates("new", similar, exclude_hashes={"a", "b"})
        assert result == []

    def test_none_exclude_hashes_treated_as_empty_set(self):
        """None exclude_hashes should not raise."""
        similar = [_make_result("x")]
        result = filter_cross_reference_candidates("new", similar, exclude_hashes=None)
        assert result == ["x"]

    def test_preserves_order(self):
        """Results should come back in the same order as input."""
        hashes = ["c", "a", "b"]
        similar = [_make_result(h) for h in hashes]
        result = filter_cross_reference_candidates("new", similar)
        assert result == hashes

    def test_duplicate_hashes_in_similar_deduplicated(self):
        """If storage returns the same hash twice, deduplicate in output."""
        similar = [_make_result("dup"), _make_result("dup"), _make_result("unique")]
        result = filter_cross_reference_candidates("new", similar)
        assert result == ["dup", "unique"]
