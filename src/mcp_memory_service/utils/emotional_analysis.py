"""
Lightweight emotional analysis for memory content.

Keyword-based emotion detection — no ML models, no external dependencies.
Assigns sentiment (positive/negative), magnitude (intensity), and categorical
emotion to text content at store time.

Design rationale:
    LLM-stored memories are overwhelmingly technical/factual. Emotional signals
    are sparse but high-value (frustration, breakthroughs, urgency). A simple
    lexicon approach captures these signals without the overhead of a sentiment
    model. False negatives are acceptable — false positives are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Emotion categories with associated keywords and sentiment polarity.
# Keywords are lowercase, matched against tokenized content.
# Polarity: positive (+1), negative (-1), neutral (0).

_EMOTION_LEXICON: dict[str, dict[str, Any]] = {
    "joy": {
        "keywords": frozenset({
            "happy", "great", "excellent", "awesome", "fantastic", "wonderful",
            "perfect", "amazing", "love", "brilliant", "success", "succeeded",
            "celebrate", "excited", "thrilled", "delighted", "pleased",
            "hooray", "finally", "breakthrough", "nailed", "crushed",
        }),
        "polarity": 1.0,
        "magnitude": 0.8,
    },
    "frustration": {
        "keywords": frozenset({
            "frustrated", "frustrating", "annoying", "annoyed", "angry",
            "broken", "fails", "failing", "failed", "stuck", "impossible",
            "ridiculous", "terrible", "horrible", "awful", "hate",
            "fuck", "shit", "damn", "crap", "wtf", "ugh", "sucks",
            "nightmare", "disaster", "furious", "rage", "infuriating",
        }),
        "polarity": -1.0,
        "magnitude": 0.9,
    },
    "urgency": {
        "keywords": frozenset({
            "urgent", "asap", "immediately", "critical", "emergency",
            "deadline", "blocker", "blocking", "showstopper", "priority",
            "hurry", "rush", "time-sensitive", "overdue",
            "outage", "downtime", "incident", "p0", "p1",
        }),
        "polarity": -0.3,
        "magnitude": 0.85,
    },
    "curiosity": {
        "keywords": frozenset({
            "wondering", "curious", "interesting", "fascinating", "explore",
            "investigate", "research", "discover", "learn", "understand",
            "hypothesis", "experiment", "what-if", "suppose", "theory",
        }),
        "polarity": 0.3,
        "magnitude": 0.5,
    },
    "concern": {
        "keywords": frozenset({
            "worried", "concern", "concerned", "risk", "risky", "danger",
            "dangerous", "careful", "caution", "warning", "alert",
            "vulnerability", "insecure", "unstable", "fragile", "brittle",
        }),
        "polarity": -0.4,
        "magnitude": 0.6,
    },
    "excitement": {
        "keywords": frozenset({
            "exciting", "thrilling", "incredible", "remarkable", "extraordinary",
            "game-changer", "revolutionary", "mind-blowing", "wow", "whoa",
            "cool", "neat", "sweet", "epic", "legendary",
        }),
        "polarity": 0.8,
        "magnitude": 0.75,
    },
    "sadness": {
        "keywords": frozenset({
            "sad", "depressed", "disappointed", "unfortunate", "regret",
            "sorry", "loss", "lost", "miss", "missing", "grief",
            "hopeless", "discouraged", "defeated", "gave-up", "abandoned",
        }),
        "polarity": -0.7,
        "magnitude": 0.7,
    },
    "confidence": {
        "keywords": frozenset({
            "confident", "certain", "sure", "proven", "verified",
            "confirmed", "validated", "solid", "robust", "reliable",
            "stable", "tested", "works", "working", "solved",
        }),
        "polarity": 0.5,
        "magnitude": 0.6,
    },
}

# Pre-compile tokenizer
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Intensifiers amplify magnitude by this factor
_INTENSIFIERS: frozenset[str] = frozenset({
    "very", "extremely", "incredibly", "absolutely", "completely",
    "totally", "utterly", "really", "seriously", "deeply",
})

# Negators flip sentiment polarity
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "hardly", "barely",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent",
    "cant", "couldnt", "shouldnt", "wouldnt", "wont",
})


@dataclass(frozen=True, slots=True)
class EmotionalValence:
    """Immutable emotional analysis result."""

    sentiment: float  # -1.0 (negative) to 1.0 (positive)
    magnitude: float  # 0.0 (neutral) to 1.0 (intense)
    category: str     # Primary emotion category

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
    Analyze emotional content of text using keyword matching.

    Algorithm:
        1. Tokenize to lowercase words
        2. Detect negators and intensifiers in context
        3. Match tokens against emotion lexicons
        4. Score category with most keyword hits as primary
        5. Compute weighted sentiment and magnitude

    Args:
        text: Raw text content to analyze

    Returns:
        EmotionalValence with sentiment, magnitude, and category
    """
    if not text or not text.strip():
        return EmotionalValence.neutral()

    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return EmotionalValence.neutral()

    token_set = set(tokens)

    # Check for negators and intensifiers
    has_negation = bool(token_set & _NEGATORS)
    has_intensifier = bool(token_set & _INTENSIFIERS)

    # Score each emotion category by keyword hit count
    category_scores: dict[str, int] = {}
    for category, lexicon in _EMOTION_LEXICON.items():
        hits = len(token_set & lexicon["keywords"])
        if hits > 0:
            category_scores[category] = hits

    # No emotional keywords found
    if not category_scores:
        return EmotionalValence.neutral()

    # Primary category = most keyword hits (ties broken by lexicon magnitude)
    primary = max(
        category_scores,
        key=lambda c: (category_scores[c], _EMOTION_LEXICON[c]["magnitude"]),
    )
    lexicon = _EMOTION_LEXICON[primary]

    # Compute sentiment and magnitude
    sentiment = lexicon["polarity"]
    magnitude = lexicon["magnitude"]

    # Scale magnitude by hit density (more hits = stronger signal)
    # Cap at 3 hits for full magnitude, normalize below that
    hit_count = category_scores[primary]
    density_factor = min(hit_count / 3.0, 1.0)
    magnitude *= density_factor

    # Apply intensifier boost (+25%)
    if has_intensifier:
        magnitude = min(magnitude * 1.25, 1.0)

    # Apply negation (flip sentiment, reduce magnitude slightly)
    if has_negation:
        sentiment *= -0.7  # Partial flip — negation is imperfect
        magnitude *= 0.8   # Negated emotions are weaker

    # Clamp to valid ranges
    sentiment = max(-1.0, min(1.0, sentiment))
    magnitude = max(0.0, min(1.0, magnitude))

    return EmotionalValence(
        sentiment=sentiment,
        magnitude=magnitude,
        category=primary,
    )
