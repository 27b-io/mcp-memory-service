"""Tests for multi-set RRF merge."""

from datetime import datetime

from mcp_memory_service.models.memory import Memory, MemoryQueryResult


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=[],
        memory_type="note",
        created_at=datetime.now().timestamp(),
    )


def _make_result(content_hash: str, score: float) -> MemoryQueryResult:
    return MemoryQueryResult(
        memory=_make_memory(content_hash),
        relevance_score=score,
        debug_info={},
    )


def test_combine_results_rrf_multi_exists():
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    assert callable(combine_results_rrf_multi)


def test_single_result_set_matches_original():
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    results = [_make_result("hash1", 0.9), _make_result("hash2", 0.7)]
    combined = combine_results_rrf_multi(result_sets=[results], weights=[1.0], tag_matches=[])
    assert len(combined) == 2
    assert combined[0][0].content_hash == "hash1"
    assert combined[0][1] == 0.9


def test_overlapping_results_ranked_higher():
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    set_a = [_make_result("hash_overlap", 0.8), _make_result("hash_a", 0.7)]
    set_b = [_make_result("hash_overlap", 0.75), _make_result("hash_b", 0.6)]
    combined = combine_results_rrf_multi(result_sets=[set_a, set_b], weights=[1.0, 1.0], tag_matches=[])
    hashes = [m.content_hash for m, _, _ in combined]
    assert hashes[0] == "hash_overlap"
    assert len(combined) == 3


def test_cosine_score_is_max_across_sets():
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    set_a = [_make_result("hash1", 0.6)]
    set_b = [_make_result("hash1", 0.8)]
    combined = combine_results_rrf_multi(result_sets=[set_a, set_b], weights=[1.0, 1.0], tag_matches=[])
    assert combined[0][1] == 0.8


def test_weights_affect_rrf_ordering():
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    set_a = [_make_result("hash_a", 0.9)]
    set_b = [_make_result("hash_b", 0.5)]
    combined = combine_results_rrf_multi(result_sets=[set_a, set_b], weights=[0.1, 2.0], tag_matches=[])
    hashes = [m.content_hash for m, _, _ in combined]
    assert hashes[0] == "hash_b"
