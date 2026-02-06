"""
Unit tests for spaced repetition and adaptive LTP utilities.

Tests spacing quality computation, retrieval boost application,
LTP rate modulation, and boundary conditions.
"""

import time

from mcp_memory_service.utils.spaced_repetition import (
    apply_spacing_boost,
    compute_ltp_rate,
    compute_spacing_quality,
)

# =============================================================================
# Spacing Quality Tests
# =============================================================================


class TestComputeSpacingQuality:
    """Tests for the compute_spacing_quality function."""

    def test_empty_timestamps_returns_zero(self):
        assert compute_spacing_quality([]) == 0.0

    def test_single_timestamp_returns_zero(self):
        assert compute_spacing_quality([time.time()]) == 0.0

    def test_clustered_access_low_quality(self):
        """Rapid bursts of access should produce low spacing quality."""
        now = time.time()
        # 5 accesses within 10 seconds — cramming
        timestamps = [now + i for i in range(5)]
        quality = compute_spacing_quality(timestamps)
        assert quality < 0.1

    def test_well_spaced_access_high_quality(self):
        """Access at increasing intervals should produce high spacing quality."""
        now = time.time()
        # Access at 0, 1 hour, 1 day, 1 week — classic spaced repetition
        timestamps = [
            now,
            now + 3600,  # +1 hour
            now + 86400,  # +1 day
            now + 604800,  # +1 week
        ]
        quality = compute_spacing_quality(timestamps)
        assert quality > 0.5

    def test_expanding_intervals_better_than_uniform(self):
        """Expanding intervals (spaced repetition) should score higher than uniform."""
        now = time.time()

        # Expanding: 1h, 6h, 24h
        expanding = [now, now + 3600, now + 3600 + 21600, now + 3600 + 21600 + 86400]

        # Uniform: 10h, 10h, 10h (same total time range)
        total_time = 3600 + 21600 + 86400
        uniform = [now, now + total_time / 3, now + 2 * total_time / 3, now + total_time]

        expanding_q = compute_spacing_quality(expanding)
        uniform_q = compute_spacing_quality(uniform)

        assert expanding_q > uniform_q

    def test_contracting_intervals_lower_quality(self):
        """Contracting intervals (opposite of spaced repetition) should score lower."""
        now = time.time()

        # Expanding: 1h, 6h, 24h
        expanding = [now, now + 3600, now + 3600 + 21600, now + 3600 + 21600 + 86400]

        # Contracting: 24h, 6h, 1h (reverse pattern)
        contracting = [now, now + 86400, now + 86400 + 21600, now + 86400 + 21600 + 3600]

        expanding_q = compute_spacing_quality(expanding)
        contracting_q = compute_spacing_quality(contracting)

        assert expanding_q > contracting_q

    def test_bounded_zero_to_one(self):
        """Result should always be in [0, 1]."""
        now = time.time()
        # Extreme spacing: years apart
        timestamps = [now, now + 365 * 86400, now + 2 * 365 * 86400]
        quality = compute_spacing_quality(timestamps)
        assert 0.0 <= quality <= 1.0

    def test_unordered_timestamps_handled(self):
        """Timestamps in any order should produce the same result."""
        now = time.time()
        ordered = [now, now + 3600, now + 86400]
        shuffled = [now + 86400, now, now + 3600]
        assert compute_spacing_quality(ordered) == compute_spacing_quality(shuffled)

    def test_two_timestamps_uses_neutral_expansion(self):
        """With exactly 2 timestamps (1 interval), expansion ratio is neutral (0.5)."""
        now = time.time()
        quality = compute_spacing_quality([now, now + 86400])  # 1 day apart
        assert quality > 0.0
        # With neutral expansion (0.5), score = interval_score * (0.5 + 0.25) = 0.75 * interval_score
        # This ensures a single well-spaced pair still gets credit


# =============================================================================
# Spacing Boost Tests
# =============================================================================


class TestApplySpacingBoost:
    """Tests for the apply_spacing_boost function."""

    def test_zero_spacing_no_change(self):
        assert apply_spacing_boost(0.5, 0.0) == 0.5

    def test_max_spacing_boosts_correctly(self):
        boosted = apply_spacing_boost(1.0, 1.0, boost_weight=0.1)
        assert abs(boosted - 1.1) < 0.001

    def test_partial_spacing(self):
        boosted = apply_spacing_boost(1.0, 0.5, boost_weight=0.1)
        expected = 1.0 * (1.0 + 0.1 * 0.5)  # 1.05
        assert abs(boosted - expected) < 0.001

    def test_multiplicative_on_base(self):
        """Boost should scale with base score."""
        small = apply_spacing_boost(0.1, 1.0, boost_weight=0.1)
        large = apply_spacing_boost(1.0, 1.0, boost_weight=0.1)
        assert large > small
        assert abs(large / small - 10.0) < 0.01

    def test_zero_base_stays_zero(self):
        assert apply_spacing_boost(0.0, 1.0) == 0.0


# =============================================================================
# Adaptive LTP Rate Tests
# =============================================================================


class TestComputeLTPRate:
    """Tests for the compute_ltp_rate function."""

    def test_zero_weight_full_rate(self):
        """At weight=0, should get maximum effective rate (modulo spacing)."""
        rate = compute_ltp_rate(base_rate=0.15, current_weight=0.0, max_weight=1.0, spacing_quality=1.0)
        assert abs(rate - 0.15) < 0.001  # Full rate with perfect spacing

    def test_max_weight_zero_rate(self):
        """At weight=max_weight, effective rate should be 0."""
        rate = compute_ltp_rate(base_rate=0.15, current_weight=1.0, max_weight=1.0, spacing_quality=1.0)
        assert rate == 0.0

    def test_half_weight_half_saturation(self):
        """At half max weight, saturation factor should be 0.5."""
        rate = compute_ltp_rate(base_rate=0.15, current_weight=0.5, max_weight=1.0, spacing_quality=1.0)
        # rate = 0.15 * 0.5 * 1.0 = 0.075
        assert abs(rate - 0.075) < 0.001

    def test_zero_spacing_halves_rate(self):
        """Zero spacing quality (clustered access) should halve the rate."""
        full_spacing = compute_ltp_rate(base_rate=0.15, current_weight=0.0, max_weight=1.0, spacing_quality=1.0)
        no_spacing = compute_ltp_rate(base_rate=0.15, current_weight=0.0, max_weight=1.0, spacing_quality=0.0)
        assert abs(no_spacing / full_spacing - 0.5) < 0.001

    def test_negative_weight_clamped(self):
        """Negative current weight should not produce negative rate."""
        rate = compute_ltp_rate(base_rate=0.15, current_weight=-0.1, max_weight=1.0)
        assert rate >= 0.0

    def test_zero_max_weight_returns_zero(self):
        """Zero max_weight should return 0 (prevents division by zero)."""
        rate = compute_ltp_rate(base_rate=0.15, current_weight=0.0, max_weight=0.0)
        assert rate == 0.0

    def test_rate_decreases_with_weight(self):
        """Rate should monotonically decrease as weight increases."""
        rates = [compute_ltp_rate(0.15, w, 1.0, 0.5) for w in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_rate_increases_with_spacing(self):
        """Rate should increase with better spacing quality."""
        rates = [compute_ltp_rate(0.15, 0.3, 1.0, sq) for sq in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]


# =============================================================================
# Integration Tests
# =============================================================================


class TestSpacedRepetitionIntegration:
    """Tests combining spacing quality with boost and LTP."""

    def test_never_accessed_memory_no_boost(self):
        """A never-accessed memory gets no spacing boost."""
        quality = compute_spacing_quality([])
        boosted = apply_spacing_boost(0.8, quality)
        assert boosted == 0.8

    def test_well_studied_memory_gets_boost(self):
        """A memory with good spaced repetition should get boost."""
        now = time.time()
        timestamps = [
            now - 604800 * 4,  # 4 weeks ago
            now - 604800 * 2,  # 2 weeks ago
            now - 604800,  # 1 week ago
            now,  # now
        ]
        quality = compute_spacing_quality(timestamps)
        assert quality > 0.3  # Decent spacing

        boosted = apply_spacing_boost(0.8, quality, boost_weight=0.1)
        assert boosted > 0.8  # Should be boosted

    def test_crammed_memory_weak_ltp(self):
        """Rapid cramming should produce weak LTP strengthening."""
        now = time.time()
        crammed = [now + i for i in range(10)]  # 10 accesses in 10 seconds
        spacing = compute_spacing_quality(crammed)
        rate = compute_ltp_rate(0.15, 0.3, 1.0, spacing)

        # Compare with well-spaced access
        spaced = [now + i * 86400 for i in range(10)]  # daily for 10 days
        spacing_good = compute_spacing_quality(spaced)
        rate_good = compute_ltp_rate(0.15, 0.3, 1.0, spacing_good)

        assert rate < rate_good  # Crammed should strengthen less
