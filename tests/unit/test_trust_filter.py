"""
Unit tests for trust score filtering in MemoryService.

Covers:
- _filter_by_trust_score: shared helper that replaces duplicated inline filter logic
- Provenance resolution from both storage locations (metadata vs top-level)
- Regression test for critical OR-condition bug (low trust rescued by missing location)
"""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.utils.provenance import DEFAULT_TRUST_SCORE

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.update_memory_metadata.return_value = (True, "Updated")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    return storage


@pytest.fixture
def memory_service(mock_storage):
    return MemoryService(storage=mock_storage, graph_client=None)


def _make_result(*, content_hash: str, trust_score: float | None = None, in_metadata: bool = False) -> dict:
    """Build a minimal search result dict with provenance at the specified location.

    Args:
        content_hash: Unique identifier.
        trust_score: If set, creates provenance with this score. None = no provenance.
        in_metadata: If True, provenance goes in metadata.provenance (pre-roundtrip).
                     If False, provenance goes at top level (post-roundtrip canonical).
    """
    result: dict = {"content_hash": content_hash, "content": "test", "metadata": {}}
    if trust_score is not None:
        prov = {"trust_score": trust_score, "source": "test"}
        if in_metadata:
            result["metadata"]["provenance"] = prov
        else:
            result["provenance"] = prov
    return result


# =============================================================================
# TestFilterByTrustScore
# =============================================================================


class TestFilterByTrustScore:
    """Tests for MemoryService._filter_by_trust_score()."""

    def test_filters_low_trust_memories(self, memory_service):
        results = [_make_result(content_hash="low", trust_score=0.3)]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.5)
        assert len(filtered) == 0

    def test_keeps_high_trust_memories(self, memory_service):
        results = [_make_result(content_hash="high", trust_score=0.8)]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.5)
        assert len(filtered) == 1
        assert filtered[0]["content_hash"] == "high"

    def test_no_filter_when_none(self, memory_service):
        results = [
            _make_result(content_hash="a", trust_score=0.1),
            _make_result(content_hash="b", trust_score=0.9),
        ]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=None)
        assert len(filtered) == 2

    def test_no_filter_when_zero(self, memory_service):
        results = [
            _make_result(content_hash="a", trust_score=0.1),
            _make_result(content_hash="b", trust_score=0.9),
        ]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.0)
        assert len(filtered) == 2

    def test_legacy_no_provenance_gets_default(self, memory_service):
        """Memories without provenance should be treated as DEFAULT_TRUST_SCORE."""
        results = [_make_result(content_hash="legacy", trust_score=None)]
        # Default is 0.5, so min_trust_score=0.5 should keep it
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=DEFAULT_TRUST_SCORE)
        assert len(filtered) == 1
        # But 0.6 should filter it out
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.6)
        assert len(filtered) == 0

    def test_provenance_in_metadata_only(self, memory_service):
        """Pre-roundtrip: provenance lives in metadata.provenance."""
        results = [_make_result(content_hash="meta", trust_score=0.8, in_metadata=True)]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.5)
        assert len(filtered) == 1

    def test_post_roundtrip_provenance_at_top_level(self, memory_service):
        """Post-roundtrip: provenance lives at result top level."""
        results = [_make_result(content_hash="top", trust_score=0.8, in_metadata=False)]
        filtered = memory_service._filter_by_trust_score(results, min_trust_score=0.5)
        assert len(filtered) == 1

    def test_low_trust_not_rescued_by_missing_second_location(self, memory_service):
        """CRITICAL REGRESSION: A low-trust memory must NOT pass the filter just because
        the other provenance location is missing (which would default to 0.5).

        This is the bug: the old OR condition meant that if provenance was in metadata
        with trust=0.3, the top-level check returned default 0.5, and 0.5 >= 0.5 passed.
        """
        # Provenance in metadata only, low trust
        result_in_meta = _make_result(content_hash="sneaky_meta", trust_score=0.3, in_metadata=True)
        filtered = memory_service._filter_by_trust_score([result_in_meta], min_trust_score=0.5)
        assert len(filtered) == 0, "Low-trust memory in metadata must not pass filter"

        # Provenance at top level only, low trust
        result_at_top = _make_result(content_hash="sneaky_top", trust_score=0.3, in_metadata=False)
        filtered = memory_service._filter_by_trust_score([result_at_top], min_trust_score=0.5)
        assert len(filtered) == 0, "Low-trust memory at top level must not pass filter"
