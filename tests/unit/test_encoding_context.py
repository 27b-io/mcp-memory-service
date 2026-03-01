"""
Unit tests for encoding context capture and context-dependent retrieval.

Tests context capture, context similarity computation, retrieval boost
application, and boundary conditions.
"""

from datetime import UTC, datetime

from mcp_memory_service.utils.encoding_context import (
    EncodingContext,
    apply_context_boost,
    capture_encoding_context,
    compute_context_similarity,
)

# =============================================================================
# Context Capture Tests
# =============================================================================


class TestCaptureEncodingContext:
    """Tests for the capture_encoding_context function."""

    def test_captures_tags(self):
        ctx = capture_encoding_context(tags=["python", "api"])
        assert ctx.task_tags == ("python", "api")

    def test_captures_agent(self):
        ctx = capture_encoding_context(agent="my-laptop")
        assert ctx.agent == "my-laptop"

    def test_empty_tags_default(self):
        ctx = capture_encoding_context()
        assert ctx.task_tags == ()

    def test_empty_agent_default(self):
        ctx = capture_encoding_context()
        assert ctx.agent == ""

    def test_caps_tags_at_10(self):
        tags = [f"tag{i}" for i in range(20)]
        ctx = capture_encoding_context(tags=tags)
        assert len(ctx.task_tags) == 10

    def test_morning_bucket(self):
        """8 AM UTC should be 'morning'."""
        # 2026-02-07 08:00 UTC
        ts = datetime(2026, 2, 7, 8, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.time_of_day == "morning"

    def test_afternoon_bucket(self):
        """14:00 UTC should be 'afternoon'."""
        ts = datetime(2026, 2, 7, 14, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.time_of_day == "afternoon"

    def test_evening_bucket(self):
        """20:00 UTC should be 'evening'."""
        ts = datetime(2026, 2, 7, 20, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.time_of_day == "evening"

    def test_night_bucket(self):
        """23:00 UTC should be 'night'."""
        ts = datetime(2026, 2, 7, 23, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.time_of_day == "night"

    def test_early_morning_is_night(self):
        """3 AM UTC should be 'night'."""
        ts = datetime(2026, 2, 7, 3, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.time_of_day == "night"

    def test_weekday_detected(self):
        """2026-02-07 is a Saturday, but let's test a weekday."""
        # 2026-02-09 is Monday
        ts = datetime(2026, 2, 9, 10, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.day_type == "weekday"

    def test_weekend_detected(self):
        """2026-02-07 is a Saturday."""
        ts = datetime(2026, 2, 7, 10, 0, 0, tzinfo=UTC).timestamp()
        ctx = capture_encoding_context(timestamp=ts)
        assert ctx.day_type == "weekend"


# =============================================================================
# Serialization Tests
# =============================================================================


class TestEncodingContextSerialization:
    """Tests for to_dict / from_dict round-tripping."""

    def test_round_trip(self):
        original = EncodingContext(
            time_of_day="afternoon",
            day_type="weekday",
            agent="my-host",
            task_tags=("python", "api"),
        )
        data = original.to_dict()
        restored = EncodingContext.from_dict(data)
        assert restored.time_of_day == "afternoon"
        assert restored.day_type == "weekday"
        assert restored.agent == "my-host"
        assert restored.task_tags == ("python", "api")

    def test_from_empty_dict(self):
        ctx = EncodingContext.from_dict({})
        assert ctx.time_of_day == "morning"
        assert ctx.day_type == "weekday"
        assert ctx.agent == ""
        assert ctx.task_tags == ()

    def test_to_dict_tags_are_list(self):
        """Storage format uses list, not tuple."""
        ctx = EncodingContext(task_tags=("a", "b"))
        d = ctx.to_dict()
        assert isinstance(d["task_tags"], list)


# =============================================================================
# Context Similarity Tests
# =============================================================================


class TestComputeContextSimilarity:
    """Tests for the compute_context_similarity function."""

    def test_identical_contexts_score_one(self):
        ctx = EncodingContext(
            time_of_day="morning",
            day_type="weekday",
            agent="host-a",
            task_tags=("python", "api"),
        )
        assert compute_context_similarity(ctx, ctx) == 1.0

    def test_completely_different_contexts_score_low(self):
        stored = EncodingContext(
            time_of_day="morning",
            day_type="weekday",
            agent="host-a",
            task_tags=("python",),
        )
        current = EncodingContext(
            time_of_day="night",
            day_type="weekend",
            agent="host-b",
            task_tags=("rust",),
        )
        score = compute_context_similarity(stored, current)
        assert score < 0.15

    def test_same_time_different_agent(self):
        stored = EncodingContext(time_of_day="morning", agent="host-a")
        current = EncodingContext(time_of_day="morning", agent="host-b")
        score = compute_context_similarity(stored, current)
        # Time matches (1.0*0.25), agent differs (0.0*0.25),
        # both empty tags (0.5*0.40), both empty day neutral (1.0*0.10)
        assert 0.3 < score < 0.7

    def test_adjacent_time_buckets_partial_match(self):
        """Morning → afternoon should be 0.5 for time dimension."""
        stored = EncodingContext(time_of_day="morning", agent="x", task_tags=("a",))
        current = EncodingContext(time_of_day="afternoon", agent="x", task_tags=("a",))
        score = compute_context_similarity(stored, current)
        # Adjacent time (0.5*0.25) + same day (1.0*0.10) + same agent (1.0*0.25) + same tags (1.0*0.40)
        expected_approx = 0.5 * 0.25 + 1.0 * 0.10 + 1.0 * 0.25 + 1.0 * 0.40
        assert abs(score - expected_approx) < 0.01

    def test_opposite_time_buckets_zero(self):
        """Morning → evening (distance 2) should be 0.0 for time dimension."""
        stored = EncodingContext(time_of_day="morning")
        current = EncodingContext(time_of_day="evening")
        # With all empty tags/agents, we get neutral scores for those dims
        score = compute_context_similarity(stored, current)
        # time=0.0 (dist 2), day=1.0 (both weekday), agent=0.5, tags=0.5
        expected = 0.0 * 0.25 + 1.0 * 0.10 + 0.5 * 0.25 + 0.5 * 0.40
        assert abs(score - expected) < 0.01

    def test_tag_jaccard_partial_overlap(self):
        """2 of 4 unique tags = 0.5 Jaccard."""
        stored = EncodingContext(task_tags=("python", "api", "auth"))
        current = EncodingContext(task_tags=("python", "api", "deploy"))
        score = compute_context_similarity(stored, current)
        # Tags Jaccard: |{python,api}| / |{python,api,auth,deploy}| = 2/4 = 0.5
        # time=1.0, day=1.0, agent=0.5, tags=0.5
        assert 0.5 < score < 0.8

    def test_no_tags_both_sides_neutral(self):
        """Both empty tags should give 0.5 (neutral, not punishing)."""
        stored = EncodingContext()
        current = EncodingContext()
        score = compute_context_similarity(stored, current)
        # time=1.0, day=1.0, agent=0.5, tags=0.5
        expected = 1.0 * 0.25 + 1.0 * 0.10 + 0.5 * 0.25 + 0.5 * 0.40
        assert abs(score - expected) < 0.01

    def test_agent_case_insensitive(self):
        stored = EncodingContext(agent="MyHost")
        current = EncodingContext(agent="myhost")
        score = compute_context_similarity(stored, current)
        # Agent should match
        assert score > 0.5

    def test_result_bounded_zero_to_one(self):
        """Similarity must always be in [0.0, 1.0]."""
        for time_val in ("morning", "afternoon", "evening", "night"):
            for day_val in ("weekday", "weekend"):
                stored = EncodingContext(time_of_day=time_val, day_type=day_val)
                current = EncodingContext(time_of_day="morning", day_type="weekday")
                score = compute_context_similarity(stored, current)
                assert 0.0 <= score <= 1.0


# =============================================================================
# Retrieval Boost Tests
# =============================================================================


class TestApplyContextBoost:
    """Tests for the apply_context_boost function."""

    def test_zero_similarity_no_change(self):
        assert apply_context_boost(0.8, 0.0) == 0.8

    def test_full_similarity_max_boost(self):
        """similarity=1.0 with default weight=0.1 → +10%."""
        boosted = apply_context_boost(0.8, 1.0, boost_weight=0.1)
        assert abs(boosted - 0.88) < 0.001

    def test_half_similarity_half_boost(self):
        boosted = apply_context_boost(1.0, 0.5, boost_weight=0.1)
        assert abs(boosted - 1.05) < 0.001

    def test_zero_base_score_stays_zero(self):
        """Multiplicative boost on zero base = zero."""
        assert apply_context_boost(0.0, 1.0) == 0.0

    def test_custom_boost_weight(self):
        boosted = apply_context_boost(1.0, 1.0, boost_weight=0.2)
        assert abs(boosted - 1.2) < 0.001


# =============================================================================
# Circular Time Distance Tests
# =============================================================================


class TestBucketDistance:
    """Tests for circular time bucket distance calculation."""

    def test_same_bucket_zero(self):
        from mcp_memory_service.utils.encoding_context import _bucket_distance

        assert _bucket_distance("morning", "morning") == 0

    def test_adjacent_buckets_one(self):
        from mcp_memory_service.utils.encoding_context import _bucket_distance

        assert _bucket_distance("morning", "afternoon") == 1
        assert _bucket_distance("evening", "night") == 1

    def test_opposite_buckets_two(self):
        from mcp_memory_service.utils.encoding_context import _bucket_distance

        assert _bucket_distance("morning", "evening") == 2
        assert _bucket_distance("afternoon", "night") == 2

    def test_circular_wrap(self):
        """Night → morning wraps around (distance 1, not 3)."""
        from mcp_memory_service.utils.encoding_context import _bucket_distance

        assert _bucket_distance("night", "morning") == 1
