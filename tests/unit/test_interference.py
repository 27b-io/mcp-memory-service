"""
Tests for proactive interference and contradiction detection.

Covers:
- Negation asymmetry detection
- Antonym pair detection
- Temporal supersession detection
- InterferenceResult aggregation
- Edge cases (empty content, no signals, low confidence filtering)
"""

from mcp_memory_service.utils.interference import (
    ContradictionSignal,
    InterferenceResult,
    detect_contradiction_signals,
)

# ── ContradictionSignal tests ──────────────────────────────────────────


class TestContradictionSignal:
    def test_to_dict_roundtrip(self):
        signal = ContradictionSignal(
            existing_hash="abc123",
            similarity=0.85,
            signal_type="negation",
            confidence=0.72,
            detail="test detail",
        )
        d = signal.to_dict()
        assert d["existing_hash"] == "abc123"
        assert d["similarity"] == 0.85
        assert d["signal_type"] == "negation"
        assert d["confidence"] == 0.72
        assert d["detail"] == "test detail"

    def test_to_dict_rounds_floats(self):
        signal = ContradictionSignal(
            existing_hash="x",
            similarity=0.85555,
            signal_type="antonym",
            confidence=0.72222,
            detail="detail",
        )
        d = signal.to_dict()
        assert d["similarity"] == 0.856
        assert d["confidence"] == 0.722


# ── InterferenceResult tests ───────────────────────────────────────────


class TestInterferenceResult:
    def test_empty_result(self):
        result = InterferenceResult()
        assert not result.has_contradictions
        d = result.to_dict()
        assert d["has_contradictions"] is False
        assert d["contradiction_count"] == 0
        assert d["contradictions"] == []

    def test_result_with_contradictions(self):
        signal = ContradictionSignal(
            existing_hash="abc",
            similarity=0.8,
            signal_type="negation",
            confidence=0.6,
            detail="test",
        )
        result = InterferenceResult(contradictions=[signal])
        assert result.has_contradictions
        d = result.to_dict()
        assert d["has_contradictions"] is True
        assert d["contradiction_count"] == 1
        assert len(d["contradictions"]) == 1


# ── Negation detection tests ──────────────────────────────────────────


class TestNegationDetection:
    def test_detects_negation_asymmetry(self):
        """New memory negates what existing memory asserts."""
        new = "The API does not support pagination"
        existing = "The API supports pagination with page and page_size parameters"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.1,
        )
        negation_signals = [s for s in signals if s.signal_type == "negation"]
        assert len(negation_signals) >= 1
        assert negation_signals[0].confidence > 0.0

    def test_no_signal_when_both_have_negation(self):
        """No asymmetry when both texts have negation."""
        new = "The system is not ready and cannot handle load"
        existing = "The system was not tested and has not been deployed"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.1,
        )
        # Both have negation, so asymmetry is small/zero
        negation_signals = [s for s in signals if s.signal_type == "negation"]
        # If both texts contain equal negation, no signal should fire
        # (or confidence should be very low)
        if negation_signals:
            assert negation_signals[0].confidence < 0.3

    def test_no_signal_with_low_similarity(self):
        """Low similarity means different topics — negation isn't meaningful."""
        new = "Python does not support goto statements"
        existing = "Redis supports cluster mode with automatic sharding"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.3,
            min_confidence=0.1,
        )
        # Low similarity should produce low confidence or no signals
        negation_signals = [s for s in signals if s.signal_type == "negation"]
        for s in negation_signals:
            assert s.confidence < 0.3


# ── Antonym detection tests ───────────────────────────────────────────


class TestAntonymDetection:
    def test_detects_enabled_vs_disabled(self):
        """Classic antonym pair: enabled/disabled."""
        new = "Feature flag oauth_v2 is now disabled in production"
        existing = "Feature flag oauth_v2 is enabled in production"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.9,
            min_confidence=0.1,
        )
        antonym_signals = [s for s in signals if s.signal_type == "antonym"]
        assert len(antonym_signals) >= 1
        assert "enabled" in antonym_signals[0].detail or "disabled" in antonym_signals[0].detail

    def test_detects_success_vs_failure(self):
        """Antonym pair: success/failure."""
        new = "Database migration failed with connection timeout"
        existing = "Database migration succeeded and all tables created"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.1,
        )
        antonym_signals = [s for s in signals if s.signal_type == "antonym"]
        assert len(antonym_signals) >= 1

    def test_no_antonym_when_same_side(self):
        """Both texts use the same side of an antonym pair."""
        new = "Feature X is enabled and working correctly"
        existing = "Feature X was enabled last week and is active"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.9,
            min_confidence=0.1,
        )
        antonym_signals = [s for s in signals if s.signal_type == "antonym"]
        assert len(antonym_signals) == 0

    def test_no_antonym_when_discussing_both_states(self):
        """Text discusses both states (e.g., migration instructions)."""
        new = "To enable the feature, first disable the legacy mode"
        existing = "Enable the new API by disabling the old endpoints"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.9,
            min_confidence=0.1,
        )
        # Both texts contain both sides, so no cross-match should fire
        antonym_signals = [s for s in signals if s.signal_type == "antonym"]
        assert len(antonym_signals) == 0

    def test_detects_add_vs_remove(self):
        """Antonym pair: add/remove."""
        new = "Removed the caching layer from the API gateway"
        existing = "Added a caching layer to the API gateway"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.88,
            min_confidence=0.1,
        )
        antonym_signals = [s for s in signals if s.signal_type == "antonym"]
        assert len(antonym_signals) >= 1


# ── Temporal supersession tests ───────────────────────────────────────


class TestTemporalSupersession:
    def test_detects_no_longer(self):
        """'No longer' indicates supersession."""
        new = "The system no longer uses Redis for caching"
        existing = "Redis is used for caching layer"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.82,
            min_confidence=0.1,
        )
        temporal_signals = [s for s in signals if s.signal_type == "temporal"]
        assert len(temporal_signals) >= 1
        assert "no longer" in temporal_signals[0].detail.lower()

    def test_detects_switched_from(self):
        """'Switched from' indicates supersession."""
        new = "Switched from PostgreSQL to MongoDB for the events collection"
        existing = "PostgreSQL stores all event data with JSONB columns"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.78,
            min_confidence=0.1,
        )
        temporal_signals = [s for s in signals if s.signal_type == "temporal"]
        assert len(temporal_signals) >= 1

    def test_detects_replaced_by(self):
        """'Replaced by' indicates supersession."""
        new = "The old auth system was replaced by OAuth 2.1"
        existing = "Authentication uses custom JWT tokens with HS256"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.75,
            min_confidence=0.1,
        )
        temporal_signals = [s for s in signals if s.signal_type == "temporal"]
        assert len(temporal_signals) >= 1

    def test_no_temporal_without_pattern(self):
        """Normal text without temporal markers shouldn't trigger."""
        new = "The API uses REST endpoints for all CRUD operations"
        existing = "GraphQL is the primary API interface"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.8,
            min_confidence=0.1,
        )
        temporal_signals = [s for s in signals if s.signal_type == "temporal"]
        assert len(temporal_signals) == 0


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_new_content(self):
        signals = detect_contradiction_signals(
            "",
            "some existing content",
            "hash1",
            similarity=0.9,
        )
        assert signals == []

    def test_empty_existing_content(self):
        signals = detect_contradiction_signals(
            "some new content",
            "",
            "hash1",
            similarity=0.9,
        )
        assert signals == []

    def test_both_empty(self):
        signals = detect_contradiction_signals("", "", "hash1", similarity=0.9)
        assert signals == []

    def test_identical_content_no_contradiction(self):
        """Identical content shouldn't produce contradiction signals."""
        content = "The server runs on port 8080 with TLS enabled"
        signals = detect_contradiction_signals(
            content,
            content,
            "hash1",
            similarity=1.0,
            min_confidence=0.1,
        )
        # No negation asymmetry, no antonym cross-match, no temporal pattern
        assert len(signals) == 0

    def test_min_confidence_filtering(self):
        """High min_confidence should filter out weak signals."""
        new = "The API does not support pagination"
        existing = "The API supports pagination"
        # With very high confidence threshold, weak signals get filtered
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.95,
        )
        # Should have fewer or no signals compared to lower threshold
        weak_signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.1,
        )
        assert len(signals) <= len(weak_signals)

    def test_very_low_similarity_produces_low_confidence(self):
        """Contradictions with low similarity should have low confidence."""
        new = "Feature X is disabled"
        existing = "Feature Y is enabled"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.3,
            min_confidence=0.0,
        )
        for s in signals:
            # Low similarity should produce low confidence
            assert s.confidence < 0.5


# ── Combined signal tests ─────────────────────────────────────────────


class TestCombinedSignals:
    def test_multiple_signal_types(self):
        """A single comparison can produce multiple signal types."""
        new = "The system no longer uses caching and the feature is disabled"
        existing = "Caching is enabled and actively used in production"
        signals = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.85,
            min_confidence=0.1,
        )
        signal_types = {s.signal_type for s in signals}
        # Should detect at least temporal + antonym
        assert len(signal_types) >= 2

    def test_high_similarity_boosts_confidence(self):
        """Higher similarity should produce higher confidence signals."""
        new = "Feature X is disabled"
        existing = "Feature X is enabled"
        low_sim = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.6,
            min_confidence=0.0,
        )
        high_sim = detect_contradiction_signals(
            new,
            existing,
            "hash1",
            similarity=0.95,
            min_confidence=0.0,
        )
        # Get max confidence for antonym signals at each similarity
        low_max = max((s.confidence for s in low_sim if s.signal_type == "antonym"), default=0)
        high_max = max((s.confidence for s in high_sim if s.signal_type == "antonym"), default=0)
        assert high_max > low_max
