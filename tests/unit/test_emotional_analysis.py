"""
Unit tests for emotional analysis utility.

Tests keyword-based emotion detection: sentiment, magnitude, category,
negation handling, intensifier handling, and edge cases.
"""

from mcp_memory_service.utils.emotional_analysis import (
    EmotionalValence,
    analyze_emotion,
)

# =============================================================================
# Basic Detection Tests
# =============================================================================


class TestAnalyzeEmotion:
    """Tests for the analyze_emotion function."""

    def test_empty_input_returns_neutral(self):
        result = analyze_emotion("")
        assert result.category == "neutral"
        assert result.sentiment == 0.0
        assert result.magnitude == 0.0

    def test_whitespace_only_returns_neutral(self):
        result = analyze_emotion("   \n\t  ")
        assert result.category == "neutral"

    def test_none_safe(self):
        """Should handle None-ish empty strings."""
        result = analyze_emotion("")
        assert result == EmotionalValence.neutral()

    def test_technical_content_is_neutral(self):
        """Typical technical memory content should be neutral."""
        result = analyze_emotion(
            "Added retry logic to the HTTP client with exponential backoff. "
            "The connection pool now handles 5xx errors gracefully."
        )
        assert result.category == "neutral"
        assert result.magnitude == 0.0

    def test_joy_detection(self):
        result = analyze_emotion("This is amazing! The fix is perfect and everything works great!")
        assert result.category == "joy"
        assert result.sentiment > 0
        assert result.magnitude > 0.5

    def test_frustration_detection(self):
        result = analyze_emotion("This is so frustrating, the damn thing keeps failing and I'm stuck")
        assert result.category == "frustration"
        assert result.sentiment < 0
        assert result.magnitude > 0.5

    def test_urgency_detection(self):
        result = analyze_emotion("URGENT: production outage, P0 incident, need fix immediately")
        assert result.category == "urgency"
        assert result.magnitude > 0.5

    def test_curiosity_detection(self):
        result = analyze_emotion("I'm curious about this fascinating approach, want to explore and research it")
        assert result.category == "curiosity"
        assert result.sentiment > 0

    def test_concern_detection(self):
        result = analyze_emotion("I'm worried about this vulnerability, it's a serious security risk")
        assert result.category == "concern"
        assert result.sentiment < 0

    def test_excitement_detection(self):
        result = analyze_emotion("This is incredible, a total game-changer! Wow, mind-blowing results!")
        assert result.category == "excitement"
        assert result.sentiment > 0
        assert result.magnitude > 0.5

    def test_sadness_detection(self):
        result = analyze_emotion("I'm disappointed and discouraged, the project feels hopeless")
        assert result.category == "sadness"
        assert result.sentiment < 0

    def test_confidence_detection(self):
        result = analyze_emotion("I'm confident this solution is proven and validated, it's solid and reliable")
        assert result.category == "confidence"
        assert result.sentiment > 0


# =============================================================================
# Modifier Tests
# =============================================================================


class TestEmotionalModifiers:
    """Tests for negation and intensifier handling."""

    def test_intensifier_boosts_magnitude(self):
        base = analyze_emotion("I'm happy about this")
        intensified = analyze_emotion("I'm extremely happy about this")
        assert intensified.magnitude >= base.magnitude

    def test_negation_flips_sentiment(self):
        positive = analyze_emotion("This is amazing and perfect")
        negated = analyze_emotion("This is not amazing and not perfect")
        # Negation should reduce or flip the sentiment
        assert negated.sentiment < positive.sentiment

    def test_negation_reduces_magnitude(self):
        base = analyze_emotion("I'm frustrated")
        negated = analyze_emotion("I'm not frustrated")
        assert negated.magnitude < base.magnitude


# =============================================================================
# Magnitude Scaling Tests
# =============================================================================


class TestMagnitudeScaling:
    """Tests for hit density magnitude scaling."""

    def test_single_keyword_lower_magnitude(self):
        """Single keyword hit should have lower magnitude than multiple."""
        single = analyze_emotion("I'm happy")
        multi = analyze_emotion("I'm happy, delighted, and thrilled about this amazing success")
        assert multi.magnitude > single.magnitude

    def test_magnitude_bounded(self):
        """Magnitude should never exceed 1.0."""
        result = analyze_emotion(
            "extremely very incredibly amazingly happy great excellent awesome fantastic wonderful perfect brilliant"
        )
        assert result.magnitude <= 1.0

    def test_sentiment_bounded(self):
        """Sentiment should stay within [-1.0, 1.0]."""
        pos = analyze_emotion("happy great excellent awesome fantastic wonderful amazing")
        neg = analyze_emotion("frustrated angry broken failing horrible terrible")
        assert -1.0 <= pos.sentiment <= 1.0
        assert -1.0 <= neg.sentiment <= 1.0


# =============================================================================
# Serialization Tests
# =============================================================================


class TestEmotionalValenceSerialization:
    """Tests for to_dict/from_dict round-tripping."""

    def test_to_dict(self):
        valence = EmotionalValence(sentiment=0.8, magnitude=0.6, category="joy")
        d = valence.to_dict()
        assert d["sentiment"] == 0.8
        assert d["magnitude"] == 0.6
        assert d["category"] == "joy"

    def test_from_dict(self):
        d = {"sentiment": -0.5, "magnitude": 0.9, "category": "frustration"}
        valence = EmotionalValence.from_dict(d)
        assert valence.sentiment == -0.5
        assert valence.magnitude == 0.9
        assert valence.category == "frustration"

    def test_from_dict_defaults(self):
        """Missing fields should default to neutral values."""
        valence = EmotionalValence.from_dict({})
        assert valence.sentiment == 0.0
        assert valence.magnitude == 0.0
        assert valence.category == "neutral"

    def test_round_trip(self):
        original = analyze_emotion("This is frustrating and broken")
        d = original.to_dict()
        restored = EmotionalValence.from_dict(d)
        assert restored.category == original.category
        assert abs(restored.sentiment - original.sentiment) < 0.001
        assert abs(restored.magnitude - original.magnitude) < 0.001

    def test_neutral_factory(self):
        n = EmotionalValence.neutral()
        assert n.sentiment == 0.0
        assert n.magnitude == 0.0
        assert n.category == "neutral"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_mixed_emotions_picks_dominant(self):
        """When multiple emotion categories match, the one with most hits wins."""
        result = analyze_emotion("I'm happy but also very frustrated and angry and stuck")
        # frustration has more keyword hits (frustrated, angry, stuck)
        assert result.category == "frustration"

    def test_case_insensitive(self):
        upper = analyze_emotion("FRUSTRATED ANGRY BROKEN")
        lower = analyze_emotion("frustrated angry broken")
        assert upper.category == lower.category

    def test_hyphenated_tokens(self):
        """Should handle hyphenated compound words."""
        result = analyze_emotion("This is a game-changer and mind-blowing")
        assert result.category == "excitement"

    def test_numbers_in_content(self):
        """Numeric content shouldn't cause errors."""
        result = analyze_emotion("Error code 42 on line 123 at offset 0xff")
        assert result.category == "neutral"

    def test_unicode_content(self):
        """Should handle unicode without errors."""
        result = analyze_emotion("This is amazing! \u2764\ufe0f Best solution ever \ud83d\ude80")
        assert result.category == "joy"
