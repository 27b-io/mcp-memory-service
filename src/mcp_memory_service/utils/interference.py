"""
Proactive interference and contradiction detection for memory storage.

Detects when a new memory contradicts existing ones using lexical signals.
Operates at store time to flag contradictions for resolution rather than
silently overwriting conflicting information.

Design rationale:
    We can't run an LLM to detect contradictions, so we use a layered approach:
    1. Vector similarity identifies semantically related memories (same topic)
    2. Negation pattern analysis detects opposing assertions
    3. Antonym pairs detect reversed claims
    4. Temporal supersession detects "used to X, now Y" patterns

    High similarity + contradiction signal = likely conflict.
    High similarity alone = related, not contradictory.
    Contradiction signal alone = probably different topics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ContradictionSignal:
    """A detected contradiction between two memories."""

    existing_hash: str
    similarity: float
    signal_type: str  # "negation", "antonym", "temporal", "sentiment_flip"
    confidence: float  # 0.0–1.0
    detail: str  # Human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "existing_hash": self.existing_hash,
            "similarity": round(self.similarity, 3),
            "signal_type": self.signal_type,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
        }


@dataclass
class InterferenceResult:
    """Result of proactive interference detection."""

    contradictions: list[ContradictionSignal] = field(default_factory=list)

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_contradictions": self.has_contradictions,
            "contradiction_count": len(self.contradictions),
            "contradictions": [c.to_dict() for c in self.contradictions],
        }


# ── Negation patterns ──────────────────────────────────────────────────

# Words that negate a statement's meaning
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "neither", "nor",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent",
    "cant", "couldnt", "shouldnt", "wouldnt", "wont", "hasnt", "havent",
    "without", "lack", "lacks", "lacking", "absent", "missing",
    "false", "incorrect", "wrong", "invalid",
})

# ── Antonym pairs (bidirectional) ──────────────────────────────────────

# Each tuple is a pair of opposing concepts. Order doesn't matter.
_ANTONYM_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    (frozenset({"enable", "enabled", "activate", "activated", "on"}),
     frozenset({"disable", "disabled", "deactivate", "deactivated", "off"})),
    (frozenset({"true", "yes", "correct", "right"}),
     frozenset({"false", "no", "incorrect", "wrong"})),
    (frozenset({"add", "added", "include", "included", "install", "installed"}),
     frozenset({"remove", "removed", "exclude", "excluded", "uninstall", "uninstalled"})),
    (frozenset({"success", "succeeded", "pass", "passed", "works", "working"}),
     frozenset({"fail", "failed", "failure", "broken", "crash", "crashed"})),
    (frozenset({"increase", "increased", "raise", "raised", "up", "higher", "more"}),
     frozenset({"decrease", "decreased", "lower", "lowered", "down", "less", "fewer"})),
    (frozenset({"start", "started", "begin", "began", "open", "opened"}),
     frozenset({"stop", "stopped", "end", "ended", "close", "closed"})),
    (frozenset({"allow", "allowed", "permit", "permitted", "accept", "accepted"}),
     frozenset({"deny", "denied", "reject", "rejected", "block", "blocked", "forbid"})),
    (frozenset({"safe", "secure", "protected"}),
     frozenset({"unsafe", "insecure", "vulnerable", "exposed"})),
    (frozenset({"required", "mandatory", "necessary", "must"}),
     frozenset({"optional", "unnecessary", "redundant"})),
    (frozenset({"deprecated", "obsolete", "legacy", "outdated"}),
     frozenset({"current", "modern", "recommended", "preferred"})),
    (frozenset({"sync", "synchronous", "blocking"}),
     frozenset({"async", "asynchronous", "non-blocking"})),
]

# ── Temporal supersession patterns ─────────────────────────────────────

# Patterns indicating that information has changed over time
_TEMPORAL_SUPERSESSION_RE = re.compile(
    r"\b(?:"
    r"(?:no longer|not anymore|stopped|switched from|migrated from|moved away from|"
    r"replaced by|superseded by|previously|used to|was .+ now|changed .+ to|"
    r"updated .+ to|upgraded .+ to)"
    r")\b",
    re.IGNORECASE,
)

# Pre-compile tokenizer (matches emotional_analysis.py)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> set[str]:
    """Tokenize text to lowercase word set."""
    return set(_TOKEN_RE.findall(text.lower()))


def detect_contradiction_signals(
    new_content: str,
    existing_content: str,
    existing_hash: str,
    similarity: float,
    min_confidence: float = 0.3,
) -> list[ContradictionSignal]:
    """
    Detect contradiction signals between new and existing memory content.

    Analyzes lexical patterns to determine if two semantically similar
    memories express contradictory information.

    Args:
        new_content: Content of the memory being stored
        existing_content: Content of an existing similar memory
        existing_hash: Content hash of the existing memory
        similarity: Cosine similarity between the two memories
        min_confidence: Minimum confidence threshold to report a signal

    Returns:
        List of ContradictionSignal objects (may be empty)
    """
    signals: list[ContradictionSignal] = []

    new_tokens = _tokenize(new_content)
    existing_tokens = _tokenize(existing_content)

    if not new_tokens or not existing_tokens:
        return signals

    # Signal 1: Negation asymmetry
    # One text negates concepts present in the other
    negation_signal = _check_negation_asymmetry(
        new_tokens, existing_tokens, new_content, existing_content,
        existing_hash, similarity,
    )
    if negation_signal and negation_signal.confidence >= min_confidence:
        signals.append(negation_signal)

    # Signal 2: Antonym pairs
    # Texts use opposing terms for same concept
    antonym_signal = _check_antonym_pairs(
        new_tokens, existing_tokens,
        existing_hash, similarity,
    )
    if antonym_signal and antonym_signal.confidence >= min_confidence:
        signals.append(antonym_signal)

    # Signal 3: Temporal supersession
    # New content explicitly states something changed
    temporal_signal = _check_temporal_supersession(
        new_content, existing_hash, similarity,
    )
    if temporal_signal and temporal_signal.confidence >= min_confidence:
        signals.append(temporal_signal)

    return signals


def _check_negation_asymmetry(
    new_tokens: set[str],
    existing_tokens: set[str],
    new_content: str,
    existing_content: str,
    existing_hash: str,
    similarity: float,
) -> ContradictionSignal | None:
    """
    Detect when one text negates assertions in the other.

    Logic: If tokens overlap significantly (same topic) but one has
    negation words where the other doesn't, that's a contradiction signal.
    """
    new_negations = new_tokens & _NEGATION_WORDS
    existing_negations = existing_tokens & _NEGATION_WORDS

    # Asymmetric negation: one has negation, the other doesn't
    negation_diff = len(new_negations) - len(existing_negations)
    if negation_diff == 0:
        return None

    # Calculate topic overlap (shared non-negation words)
    content_tokens = (new_tokens - _NEGATION_WORDS) & (existing_tokens - _NEGATION_WORDS)
    if not content_tokens:
        return None

    overlap_ratio = len(content_tokens) / max(
        len(new_tokens - _NEGATION_WORDS),
        len(existing_tokens - _NEGATION_WORDS),
        1,
    )

    # Confidence = similarity * overlap_ratio * negation strength
    # High similarity + high overlap + negation = strong contradiction
    confidence = similarity * overlap_ratio * min(abs(negation_diff) / 2.0, 1.0)

    if confidence < 0.1:
        return None

    negator = "new" if len(new_negations) > len(existing_negations) else "existing"
    return ContradictionSignal(
        existing_hash=existing_hash,
        similarity=similarity,
        signal_type="negation",
        confidence=min(confidence, 1.0),
        detail=f"Negation asymmetry: {negator} memory contains negation "
               f"({', '.join(sorted(new_negations | existing_negations))}), "
               f"topic overlap: {len(content_tokens)} words",
    )


def _check_antonym_pairs(
    new_tokens: set[str],
    existing_tokens: set[str],
    existing_hash: str,
    similarity: float,
) -> ContradictionSignal | None:
    """
    Detect when texts use antonym pairs on the same topic.

    Logic: If new text uses "enabled" and existing uses "disabled" (or vice
    versa), that's a direct contradiction on the same attribute.
    """
    found_pairs: list[tuple[str, str]] = []

    for side_a, side_b in _ANTONYM_PAIRS:
        new_a = new_tokens & side_a
        new_b = new_tokens & side_b
        existing_a = existing_tokens & side_a
        existing_b = existing_tokens & side_b

        # Cross-match: new uses side_a, existing uses side_b (or vice versa)
        if (new_a and existing_b) or (new_b and existing_a):
            # But not if both sides appear in the same text (discussing both states)
            if not (new_a and new_b) and not (existing_a and existing_b):
                pair_new = new_a | new_b
                pair_existing = existing_a | existing_b
                found_pairs.append((
                    next(iter(pair_new)),
                    next(iter(pair_existing)),
                ))

    if not found_pairs:
        return None

    # Confidence scales with similarity and number of antonym pairs found
    pair_factor = min(len(found_pairs) / 2.0, 1.0)
    confidence = similarity * (0.5 + 0.5 * pair_factor)

    pair_strs = [f"{a} vs {b}" for a, b in found_pairs[:3]]
    return ContradictionSignal(
        existing_hash=existing_hash,
        similarity=similarity,
        signal_type="antonym",
        confidence=min(confidence, 1.0),
        detail=f"Antonym pairs detected: {', '.join(pair_strs)}",
    )


def _check_temporal_supersession(
    new_content: str,
    existing_hash: str,
    similarity: float,
) -> ContradictionSignal | None:
    """
    Detect when new content explicitly states something has changed.

    Logic: Phrases like "no longer", "switched from", "was X now Y"
    indicate the new memory supersedes an older one.
    """
    matches = _TEMPORAL_SUPERSESSION_RE.findall(new_content)
    if not matches:
        return None

    # Temporal supersession is a strong signal when combined with similarity
    confidence = similarity * 0.8

    return ContradictionSignal(
        existing_hash=existing_hash,
        similarity=similarity,
        signal_type="temporal",
        confidence=min(confidence, 1.0),
        detail=f"Temporal supersession detected: '{matches[0].strip()}'",
    )
