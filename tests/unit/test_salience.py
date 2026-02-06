"""
Unit tests for salience scoring utility.

Tests score computation, boost application, weight configuration,
boundary conditions, and the logarithmic frequency normalization.
"""

from mcp_memory_service.utils.salience import (
    SalienceFactors,
    apply_salience_boost,
    compute_salience,
)

# =============================================================================
# Basic Computation Tests
# =============================================================================


class TestComputeSalience:
    """Tests for the compute_salience function."""

    def test_all_zeros_returns_zero(self):
        result = compute_salience(SalienceFactors())
        assert result == 0.0

    def test_full_importance_gives_high_score(self):
        result = compute_salience(SalienceFactors(explicit_importance=1.0))
        # With default weights (importance_weight=0.4), should be 0.4
        assert abs(result - 0.4) < 0.01

    def test_full_emotion_gives_expected_score(self):
        result = compute_salience(SalienceFactors(emotional_magnitude=1.0))
        # With default weights (emotional_weight=0.3), should be 0.3
        assert abs(result - 0.3) < 0.01

    def test_all_factors_maxed(self):
        """All factors at maximum should approach 1.0."""
        result = compute_salience(
            SalienceFactors(
                emotional_magnitude=1.0,
                access_count=100,
                explicit_importance=1.0,
            )
        )
        assert result > 0.9
        assert result <= 1.0

    def test_bounded_zero_to_one(self):
        """Result should never exceed [0, 1] range."""
        # Test with extreme values
        result = compute_salience(
            SalienceFactors(
                emotional_magnitude=1.0,
                access_count=10000,
                explicit_importance=1.0,
            )
        )
        assert 0.0 <= result <= 1.0

    def test_negative_factors_clamped(self):
        """Negative inputs should be clamped to valid range."""
        result = compute_salience(
            SalienceFactors(
                emotional_magnitude=-0.5,
                access_count=0,
                explicit_importance=-0.5,
            )
        )
        assert result >= 0.0


# =============================================================================
# Frequency Normalization Tests
# =============================================================================


class TestFrequencyNormalization:
    """Tests for logarithmic access frequency normalization."""

    def test_zero_accesses(self):
        """Zero accesses should contribute zero to frequency component."""
        result = compute_salience(SalienceFactors(access_count=0))
        assert result == 0.0

    def test_diminishing_returns(self):
        """Going from 0 to 10 accesses should matter more than 90 to 100."""
        low = compute_salience(SalienceFactors(access_count=10))
        high = compute_salience(SalienceFactors(access_count=100))
        # Both should be positive
        assert low > 0.0
        assert high > low
        # But the gap should be smaller at the high end
        delta_low = low - compute_salience(SalienceFactors(access_count=0))
        delta_high = high - compute_salience(SalienceFactors(access_count=90))
        assert delta_low > delta_high

    def test_100_accesses_near_full(self):
        """100 accesses should normalize to ~1.0 for the frequency component."""
        result = compute_salience(SalienceFactors(access_count=100))
        # frequency_weight * 1.0 = 0.3
        assert abs(result - 0.3) < 0.02


# =============================================================================
# Custom Weight Tests
# =============================================================================


class TestCustomWeights:
    """Tests for configurable weight parameters."""

    def test_emotion_only_weight(self):
        result = compute_salience(
            SalienceFactors(emotional_magnitude=1.0),
            emotional_weight=1.0,
            frequency_weight=0.0,
            importance_weight=0.0,
        )
        assert abs(result - 1.0) < 0.01

    def test_frequency_only_weight(self):
        result = compute_salience(
            SalienceFactors(access_count=100),
            emotional_weight=0.0,
            frequency_weight=1.0,
            importance_weight=0.0,
        )
        assert abs(result - 1.0) < 0.02

    def test_importance_only_weight(self):
        result = compute_salience(
            SalienceFactors(explicit_importance=1.0),
            emotional_weight=0.0,
            frequency_weight=0.0,
            importance_weight=1.0,
        )
        assert abs(result - 1.0) < 0.01

    def test_zero_weights(self):
        """All zero weights should produce zero regardless of inputs."""
        result = compute_salience(
            SalienceFactors(emotional_magnitude=1.0, access_count=100, explicit_importance=1.0),
            emotional_weight=0.0,
            frequency_weight=0.0,
            importance_weight=0.0,
        )
        assert result == 0.0


# =============================================================================
# Boost Application Tests
# =============================================================================


class TestApplySalienceBoost:
    """Tests for the apply_salience_boost function."""

    def test_zero_salience_no_change(self):
        """Salience of 0 should not change the score."""
        assert apply_salience_boost(0.5, 0.0) == 0.5

    def test_max_salience_boosts_correctly(self):
        """Max salience with default weight should boost by 15%."""
        boosted = apply_salience_boost(1.0, 1.0, boost_weight=0.15)
        assert abs(boosted - 1.15) < 0.001

    def test_partial_salience(self):
        """Partial salience should give partial boost."""
        boosted = apply_salience_boost(1.0, 0.5, boost_weight=0.15)
        expected = 1.0 * (1.0 + 0.15 * 0.5)  # 1.075
        assert abs(boosted - expected) < 0.001

    def test_multiplicative_on_base(self):
        """Boost should scale with base score."""
        small = apply_salience_boost(0.1, 1.0, boost_weight=0.15)
        large = apply_salience_boost(1.0, 1.0, boost_weight=0.15)
        assert large > small
        # Ratio should be preserved
        assert abs(large / small - 10.0) < 0.01

    def test_zero_base_stays_zero(self):
        """Zero base score stays zero regardless of salience."""
        assert apply_salience_boost(0.0, 1.0) == 0.0

    def test_custom_boost_weight(self):
        """Custom boost weight should be respected."""
        boosted = apply_salience_boost(1.0, 1.0, boost_weight=0.5)
        assert abs(boosted - 1.5) < 0.001


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestSalienceIntegration:
    """Tests combining computation and application."""

    def test_neutral_memory_no_boost(self):
        """A neutral, never-accessed, unimportant memory gets no boost."""
        salience = compute_salience(SalienceFactors())
        boosted = apply_salience_boost(0.8, salience)
        assert boosted == 0.8

    def test_emotional_important_memory_gets_boost(self):
        """A high-emotion, explicitly important memory should get notable boost."""
        salience = compute_salience(
            SalienceFactors(
                emotional_magnitude=0.9,
                access_count=20,
                explicit_importance=0.8,
            )
        )
        assert salience > 0.5  # Should be significantly salient

        boosted = apply_salience_boost(0.8, salience, boost_weight=0.15)
        assert boosted > 0.8  # Should be boosted
        assert boosted < 0.95  # But not excessively
