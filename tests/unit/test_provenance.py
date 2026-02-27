"""
Unit tests for memory provenance tracking utilities.

Tests:
1. Trust score computation per source
2. Provenance record building
3. Modification history recording
4. get_trust_score helper for memory metadata
"""

import time

from mcp_memory_service.utils.provenance import (
    DEFAULT_SOURCE_TRUST,
    MAX_MODIFICATION_HISTORY,
    build_provenance,
    compute_trust_score,
    get_trust_score,
    record_modification,
)

# =============================================================================
# compute_trust_score
# =============================================================================


class TestComputeTrustScore:
    def test_known_sources_return_configured_values(self):
        assert compute_trust_score("api") == DEFAULT_SOURCE_TRUST["api"]
        assert compute_trust_score("working_memory_consolidation") == DEFAULT_SOURCE_TRUST["working_memory_consolidation"]
        assert compute_trust_score("batch") == DEFAULT_SOURCE_TRUST["batch"]

    def test_unknown_source_returns_neutral(self):
        assert compute_trust_score("some_random_source") == 0.5
        assert compute_trust_score("") == 0.5

    def test_source_prefix_is_stripped(self):
        """source: prefix (from hostname tags) should be stripped before lookup."""
        score_with_prefix = compute_trust_score("source:api")
        score_direct = compute_trust_score("api")
        assert score_with_prefix == score_direct

    def test_unknown_prefixed_source_returns_neutral(self):
        assert compute_trust_score("source:unknown_host") == 0.5


# =============================================================================
# build_provenance
# =============================================================================


class TestBuildProvenance:
    def test_returns_required_fields(self):
        prov = build_provenance(source="api", creation_method="direct")
        assert "source" in prov
        assert "creation_method" in prov
        assert "trust_score" in prov
        assert "created_at" in prov
        assert "modification_history" in prov

    def test_source_and_method_preserved(self):
        prov = build_provenance(source="batch", creation_method="auto_split")
        assert prov["source"] == "batch"
        assert prov["creation_method"] == "auto_split"

    def test_trust_score_matches_compute(self):
        prov = build_provenance(source="api", creation_method="direct")
        assert prov["trust_score"] == compute_trust_score("api")

    def test_actor_included_when_provided(self):
        prov = build_provenance(source="api", creation_method="direct", actor="my-agent")
        assert prov["actor"] == "my-agent"

    def test_actor_absent_when_not_provided(self):
        prov = build_provenance(source="api", creation_method="direct")
        assert "actor" not in prov

    def test_extra_fields_merged(self):
        prov = build_provenance(
            source="api",
            creation_method="auto_split",
            extra={"chunk_index": 2, "total_chunks": 5},
        )
        assert prov["chunk_index"] == 2
        assert prov["total_chunks"] == 5

    def test_modification_history_starts_empty(self):
        prov = build_provenance(source="api", creation_method="direct")
        assert prov["modification_history"] == []

    def test_created_at_is_recent(self):
        before = time.time()
        prov = build_provenance(source="api", creation_method="direct")
        after = time.time()
        assert before <= prov["created_at"] <= after


# =============================================================================
# record_modification
# =============================================================================


class TestRecordModification:
    def test_appends_entry_to_history(self):
        prov = build_provenance(source="api", creation_method="direct")
        updated = record_modification(prov, operation="supersede")
        assert len(updated["modification_history"]) == 1
        assert updated["modification_history"][0]["operation"] == "supersede"

    def test_does_not_mutate_original(self):
        prov = build_provenance(source="api", creation_method="direct")
        record_modification(prov, operation="supersede")
        assert prov["modification_history"] == []

    def test_actor_included_when_provided(self):
        prov = build_provenance(source="api", creation_method="direct")
        updated = record_modification(prov, operation="update", actor="agent-42")
        assert updated["modification_history"][0]["actor"] == "agent-42"

    def test_timestamp_is_recent(self):
        prov = build_provenance(source="api", creation_method="direct")
        before = time.time()
        updated = record_modification(prov, operation="delete")
        after = time.time()
        ts = updated["modification_history"][0]["timestamp"]
        assert before <= ts <= after

    def test_multiple_modifications_accumulate(self):
        prov = build_provenance(source="api", creation_method="direct")
        prov = record_modification(prov, operation="update")
        prov = record_modification(prov, operation="supersede")
        assert len(prov["modification_history"]) == 2
        assert prov["modification_history"][1]["operation"] == "supersede"

    def test_history_capped_at_max(self):
        prov = build_provenance(source="api", creation_method="direct")
        for i in range(MAX_MODIFICATION_HISTORY + 10):
            prov = record_modification(prov, operation=f"op_{i}")
        assert len(prov["modification_history"]) == MAX_MODIFICATION_HISTORY
        # Oldest entries are dropped — last entry should be the final operation
        assert prov["modification_history"][-1]["operation"] == f"op_{MAX_MODIFICATION_HISTORY + 9}"


# =============================================================================
# get_trust_score
# =============================================================================


class TestGetTrustScore:
    def test_extracts_score_from_provenance_in_metadata(self):
        metadata = {"provenance": {"trust_score": 0.9, "source": "api"}}
        assert get_trust_score(metadata) == 0.9

    def test_returns_neutral_when_no_provenance(self):
        assert get_trust_score({}) == 0.5
        assert get_trust_score({"other_key": "value"}) == 0.5

    def test_returns_neutral_when_provenance_is_not_dict(self):
        assert get_trust_score({"provenance": None}) == 0.5
        assert get_trust_score({"provenance": "invalid"}) == 0.5

    def test_returns_neutral_when_trust_score_missing_from_provenance(self):
        assert get_trust_score({"provenance": {"source": "api"}}) == 0.5

    def test_custom_trust_score_preserved(self):
        metadata = {"provenance": {"trust_score": 0.3}}
        assert get_trust_score(metadata) == 0.3
