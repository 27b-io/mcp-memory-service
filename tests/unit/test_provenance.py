"""
Unit tests for memory provenance tracking utilities.

Tests:
1. Trust score computation per source
2. Provenance record building (including reserved-key protection)
3. Trust score resolution from search results
4. get_trust_score helper for memory metadata
"""

import time

from mcp_memory_service.utils.provenance import (
    _RESERVED_PROVENANCE_KEYS,
    DEFAULT_SOURCE_TRUST,
    DEFAULT_TRUST_SCORE,
    build_provenance,
    compute_trust_score,
    get_trust_score,
    resolve_trust_score,
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
        assert compute_trust_score("some_random_source") == DEFAULT_TRUST_SCORE
        assert compute_trust_score("") == DEFAULT_TRUST_SCORE

    def test_source_prefix_is_stripped(self):
        """source: prefix (from hostname tags) should be stripped before lookup."""
        score_with_prefix = compute_trust_score("source:api")
        score_direct = compute_trust_score("api")
        assert score_with_prefix == score_direct

    def test_unknown_prefixed_source_returns_neutral(self):
        assert compute_trust_score("source:unknown_host") == DEFAULT_TRUST_SCORE


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

    # -- Reserved-key protection tests --

    def test_extra_cannot_overwrite_trust_score(self):
        """extra dict must not be able to clobber the computed trust_score."""
        prov = build_provenance(source="api", creation_method="direct", extra={"trust_score": 0.0})
        assert prov["trust_score"] == compute_trust_score("api")

    def test_extra_cannot_overwrite_source(self):
        """extra dict must not be able to clobber the source field."""
        prov = build_provenance(source="api", creation_method="direct", extra={"source": "evil"})
        assert prov["source"] == "api"

    def test_extra_cannot_overwrite_reserved_keys(self):
        """All reserved provenance keys must be immune to extra overwrite."""
        overrides = dict.fromkeys(_RESERVED_PROVENANCE_KEYS, "HACKED")
        prov = build_provenance(source="api", creation_method="direct", extra=overrides)
        # None of the reserved fields should contain the injected value
        for key in _RESERVED_PROVENANCE_KEYS:
            assert prov.get(key) != "HACKED", f"Reserved key '{key}' was overwritten by extra"

    def test_extra_cannot_overwrite_actor(self):
        """extra dict must not clobber the explicit actor parameter."""
        prov = build_provenance(source="api", creation_method="direct", actor="legit-agent", extra={"actor": "evil"})
        assert prov["actor"] == "legit-agent"

    def test_extra_non_reserved_keys_still_work(self):
        """Non-reserved keys in extra must still be merged normally."""
        prov = build_provenance(
            source="api",
            creation_method="direct",
            extra={"chunk_index": 2, "custom_field": "hello"},
        )
        assert prov["chunk_index"] == 2
        assert prov["custom_field"] == "hello"


# =============================================================================
# resolve_trust_score
# =============================================================================


class TestResolveTrustScore:
    """Tests for resolve_trust_score() — extracts canonical trust from a search result dict."""

    def test_top_level_provenance_used(self):
        """When provenance is at top level (post-roundtrip), use that score."""
        result = {"provenance": {"trust_score": 0.9, "source": "api"}}
        assert resolve_trust_score(result) == 0.9

    def test_metadata_provenance_fallback(self):
        """When provenance only in metadata (pre-roundtrip), use that score."""
        result = {"metadata": {"provenance": {"trust_score": 0.7, "source": "batch"}}}
        assert resolve_trust_score(result) == 0.7

    def test_both_locations_uses_top_level(self):
        """When provenance in both locations, top-level (canonical) wins."""
        result = {
            "provenance": {"trust_score": 0.9, "source": "api"},
            "metadata": {"provenance": {"trust_score": 0.3, "source": "import"}},
        }
        assert resolve_trust_score(result) == 0.9

    def test_no_provenance_returns_default(self):
        """No provenance anywhere → default trust score."""
        assert resolve_trust_score({}) == DEFAULT_TRUST_SCORE
        assert resolve_trust_score({"metadata": {}}) == DEFAULT_TRUST_SCORE
        assert resolve_trust_score({"metadata": {"other": "stuff"}}) == DEFAULT_TRUST_SCORE

    def test_nan_trust_score_returns_default(self):
        """NaN trust_score must be treated as missing → default."""
        result = {"provenance": {"trust_score": float("nan"), "source": "api"}}
        assert resolve_trust_score(result) == DEFAULT_TRUST_SCORE

    def test_inf_trust_score_returns_default(self):
        """Inf trust_score must be treated as invalid → default."""
        result = {"provenance": {"trust_score": float("inf"), "source": "api"}}
        assert resolve_trust_score(result) == DEFAULT_TRUST_SCORE
        result_neg = {"provenance": {"trust_score": float("-inf"), "source": "api"}}
        assert resolve_trust_score(result_neg) == DEFAULT_TRUST_SCORE

    def test_out_of_range_trust_score_returns_default(self):
        """Trust scores outside [0.0, 1.0] must be treated as invalid."""
        assert resolve_trust_score({"provenance": {"trust_score": -0.1}}) == DEFAULT_TRUST_SCORE
        assert resolve_trust_score({"provenance": {"trust_score": 1.1}}) == DEFAULT_TRUST_SCORE


# =============================================================================
# get_trust_score
# =============================================================================


class TestGetTrustScore:
    def test_extracts_score_from_provenance_in_metadata(self):
        metadata = {"provenance": {"trust_score": 0.9, "source": "api"}}
        assert get_trust_score(metadata) == 0.9

    def test_returns_neutral_when_no_provenance(self):
        assert get_trust_score({}) == DEFAULT_TRUST_SCORE
        assert get_trust_score({"other_key": "value"}) == DEFAULT_TRUST_SCORE

    def test_returns_neutral_when_provenance_is_not_dict(self):
        assert get_trust_score({"provenance": None}) == DEFAULT_TRUST_SCORE
        assert get_trust_score({"provenance": "invalid"}) == DEFAULT_TRUST_SCORE

    def test_returns_neutral_when_trust_score_missing_from_provenance(self):
        assert get_trust_score({"provenance": {"source": "api"}}) == DEFAULT_TRUST_SCORE

    def test_custom_trust_score_preserved(self):
        metadata = {"provenance": {"trust_score": 0.3}}
        assert get_trust_score(metadata) == 0.3

    def test_returns_default_for_invalid_trust_score_values(self):
        """Non-numeric, NaN, inf, and out-of-range trust_score values must fall back to default."""
        invalid_values = [
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.1,
            1.1,
            "not_a_number",
            [],
            {"nested": "dict"},
        ]
        for val in invalid_values:
            metadata = {"provenance": {"trust_score": val}}
            assert (
                get_trust_score(metadata) == DEFAULT_TRUST_SCORE
            ), f"get_trust_score should return DEFAULT_TRUST_SCORE for trust_score={val!r}"

    def test_default_uses_constant(self):
        """The default value must be the named constant, not a magic number."""
        assert DEFAULT_TRUST_SCORE == 0.5
        assert get_trust_score({}) == DEFAULT_TRUST_SCORE
