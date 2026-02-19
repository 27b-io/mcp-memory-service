"""
Hybrid emotional analysis for memory content.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment
scoring — a validated, rule-based model with a 7,520-word lexicon that handles
negation windowing, degree modifiers, capitalization emphasis, and contrastive
conjunctions. Domain-specific keyword lexicon provides emotion categorization.

Design rationale:
    VADER gives us robust sentiment (-1 to +1) and intensity via its compound
    score and neutrality ratio. Our domain-specific keyword lexicon maps content
    to 8 emotion categories tuned for developer/technical language. This hybrid
    avoids VADER's limitation (no emotion categories) while avoiding our old
    lexicon's limitations (global negation, no degree modifiers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Domain-specific emotion category keywords.
# Used ONLY for category classification — VADER handles sentiment and magnitude.
_CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "joy": frozenset(
        {
            "happy",
            "great",
            "excellent",
            "awesome",
            "fantastic",
            "wonderful",
            "perfect",
            "amazing",
            "love",
            "brilliant",
            "success",
            "succeeded",
            "celebrate",
            "excited",
            "thrilled",
            "delighted",
            "pleased",
            "hooray",
            "finally",
            "breakthrough",
            "nailed",
            "crushed",
        }
    ),
    "frustration": frozenset(
        {
            "frustrated",
            "frustrating",
            "annoying",
            "annoyed",
            "angry",
            "broken",
            "fails",
            "failing",
            "failed",
            "stuck",
            "impossible",
            "ridiculous",
            "terrible",
            "horrible",
            "awful",
            "hate",
            "fuck",
            "shit",
            "damn",
            "crap",
            "wtf",
            "ugh",
            "sucks",
            "nightmare",
            "disaster",
            "furious",
            "rage",
            "infuriating",
        }
    ),
    "urgency": frozenset(
        {
            "urgent",
            "asap",
            "immediately",
            "critical",
            "emergency",
            "deadline",
            "blocker",
            "blocking",
            "showstopper",
            "priority",
            "hurry",
            "rush",
            "time-sensitive",
            "overdue",
            "outage",
            "downtime",
            "incident",
            "p0",
            "p1",
        }
    ),
    "curiosity": frozenset(
        {
            "wondering",
            "curious",
            "interesting",
            "fascinating",
            "explore",
            "investigate",
            "research",
            "discover",
            "learn",
            "understand",
            "hypothesis",
            "experiment",
            "what-if",
            "suppose",
            "theory",
        }
    ),
    "concern": frozenset(
        {
            "worried",
            "concern",
            "concerned",
            "risk",
            "risky",
            "danger",
            "dangerous",
            "careful",
            "caution",
            "warning",
            "alert",
            "vulnerability",
            "insecure",
            "unstable",
            "fragile",
            "brittle",
        }
    ),
    "excitement": frozenset(
        {
            "exciting",
            "thrilling",
            "incredible",
            "remarkable",
            "extraordinary",
            "game-changer",
            "revolutionary",
            "mind-blowing",
            "wow",
            "whoa",
            "cool",
            "neat",
            "sweet",
            "epic",
            "legendary",
        }
    ),
    "sadness": frozenset(
        {
            "sad",
            "depressed",
            "disappointed",
            "unfortunate",
            "regret",
            "sorry",
            "loss",
            "lost",
            "miss",
            "missing",
            "grief",
            "hopeless",
            "discouraged",
            "defeated",
            "gave-up",
            "abandoned",
        }
    ),
    "confidence": frozenset(
        {
            "confident",
            "certain",
            "sure",
            "proven",
            "verified",
            "confirmed",
            "validated",
            "solid",
            "robust",
            "reliable",
            "stable",
            "tested",
            "works",
            "working",
            "solved",
        }
    ),
}

# Pre-compile tokenizer for category keyword matching
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Lazy-initialized VADER analyzer (loads 7,520-word lexicon once)
_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Get or create the singleton VADER analyzer."""
    global _analyzer  # noqa: PLW0603
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


@dataclass(frozen=True, slots=True)
class EmotionalValence:
    """Immutable emotional analysis result."""

    sentiment: float  # -1.0 (negative) to 1.0 (positive)
    magnitude: float  # 0.0 (neutral) to 1.0 (intense)
    category: str  # Primary emotion category

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment": round(self.sentiment, 3),
            "magnitude": round(self.magnitude, 3),
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalValence:
        return cls(
            sentiment=float(data.get("sentiment", 0.0)),
            magnitude=float(data.get("magnitude", 0.0)),
            category=str(data.get("category", "neutral")),
        )

    @classmethod
    def neutral(cls) -> EmotionalValence:
        return cls(sentiment=0.0, magnitude=0.0, category="neutral")


def analyze_emotion(text: str) -> EmotionalValence:
    """
    Analyze emotional content of text using VADER + domain keyword hybrid.

    Algorithm:
        1. VADER computes sentiment using the compound score; magnitude starts
           as the absolute value of this compound score
        2. A domain keyword lexicon determines the emotion category via set
           intersection and can boost magnitude based on keyword density
        3. When no domain keywords are present, the category falls back to
           "neutral" while preserving VADER sentiment/magnitude

    Args:
        text: Raw text content to analyze

    Returns:
        EmotionalValence with sentiment, magnitude, and category
    """
    if not text or not text.strip():
        return EmotionalValence.neutral()

    # VADER: sentiment via compound score
    scores = _get_analyzer().polarity_scores(text)
    sentiment = scores["compound"]  # [-1, 1] normalized

    # Category: domain-specific keyword matching
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        # No lexical tokens (emoji-only, punctuation-heavy), but VADER may
        # still have a sentiment signal — preserve it with neutral category
        magnitude = min(abs(sentiment), 1.0)
        sentiment = max(-1.0, min(1.0, sentiment))
        return EmotionalValence(
            sentiment=sentiment,
            magnitude=magnitude,
            category="neutral",
        )

    token_set = set(tokens)
    category_scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        hits = len(token_set & keywords)
        if hits > 0:
            category_scores[cat] = hits

    has_domain_keywords = bool(category_scores)

    if has_domain_keywords:
        category = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
    elif abs(sentiment) >= 0.3:
        # Strong VADER signal without domain keywords — keep neutral category
        # to avoid expanding the API surface beyond the 8 domain categories
        category = "neutral"
    else:
        category = "neutral"

    # Magnitude: abs(compound) as base, boosted by keyword density
    magnitude = abs(sentiment)
    if has_domain_keywords:
        # Keyword density boost: more hits = stronger signal
        max_hits = max(category_scores.values())
        density_boost = min(max_hits / 3.0, 1.0) * 0.3
        magnitude = min(magnitude + density_boost, 1.0)
        # Floor: domain keywords present means at least minimal magnitude
        magnitude = max(magnitude, 0.15)

    # Clamp to valid ranges
    sentiment = max(-1.0, min(1.0, sentiment))
    magnitude = max(0.0, min(1.0, magnitude))

    return EmotionalValence(
        sentiment=sentiment,
        magnitude=magnitude,
        category=category,
    )
