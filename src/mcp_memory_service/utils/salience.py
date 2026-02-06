"""
Salience scoring for memory retrieval boosting.

Computes a 0.0–1.0 salience score for each memory based on:
    - Emotional magnitude (high-emotion memories are more memorable)
    - Access frequency (frequently retrieved = more important)
    - Explicit importance (user-assigned priority)

Salience acts as a multiplicative boost during retrieval:
    final_score = base_score * (1 + salience_boost_weight * salience)

Design rationale:
    Human memory is biased toward emotional and frequently-accessed content.
    Salience mimics this by promoting important memories without penalizing
    neutral ones (salience of 0.0 means no boost, not a penalty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SalienceFactors:
    """Input factors for salience computation."""

    emotional_magnitude: float = 0.0  # 0.0–1.0 from emotional analysis
    access_count: int = 0             # Times retrieved
    explicit_importance: float = 0.0  # 0.0–1.0 user-assigned


def compute_salience(
    factors: SalienceFactors,
    emotional_weight: float = 0.3,
    frequency_weight: float = 0.3,
    importance_weight: float = 0.4,
) -> float:
    """
    Compute salience score from contributing factors.

    Formula:
        salience = clamp(
            emotional_weight * emotional_magnitude
            + frequency_weight * log_frequency
            + importance_weight * explicit_importance,
            0.0, 1.0
        )

    Where log_frequency = log(1 + access_count) / log(1 + 100)
    (normalized so 100 accesses ≈ 1.0)

    Args:
        factors: Input factors for computation
        emotional_weight: Weight for emotional magnitude component
        frequency_weight: Weight for access frequency component
        importance_weight: Weight for explicit importance component

    Returns:
        Salience score between 0.0 and 1.0
    """
    # Normalize access frequency using log scale
    # log(1+100)/log(1+100) = 1.0 at 100 accesses
    log_normalizer = math.log(101)  # log(1 + 100)
    log_frequency = math.log(1 + factors.access_count) / log_normalizer

    # Weighted sum
    score = (
        emotional_weight * factors.emotional_magnitude
        + frequency_weight * min(log_frequency, 1.0)
        + importance_weight * factors.explicit_importance
    )

    return max(0.0, min(1.0, score))


def apply_salience_boost(
    base_score: float,
    salience_score: float,
    boost_weight: float = 0.15,
) -> float:
    """
    Apply salience boost to a retrieval score.

    Formula: boosted = base_score * (1 + boost_weight * salience_score)

    This is multiplicative so:
    - salience=0.0 → no change
    - salience=1.0, weight=0.15 → +15% boost

    Args:
        base_score: Original retrieval score
        salience_score: Memory's salience (0.0–1.0)
        boost_weight: Maximum boost factor

    Returns:
        Boosted score
    """
    return base_score * (1.0 + boost_weight * salience_score)
