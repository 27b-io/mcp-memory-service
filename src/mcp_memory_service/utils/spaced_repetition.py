"""
Spaced repetition and adaptive LTP (Long-Term Potentiation) utilities.

Implements two neuroscience-inspired memory strengthening mechanisms:

1. Spacing effect (Ebbinghaus, 1885):
   Memories accessed at increasing intervals consolidate better than those
   accessed in rapid bursts. We compute a "spacing quality" score from the
   access timestamp history and use it to boost retrieval scores.

2. Adaptive LTP (BCM theory — Bienenstock, Cooper, Munro 1982):
   Synaptic strengthening rate decreases as connection weight approaches
   its maximum. This prevents runaway potentiation and creates natural
   saturation for well-established memories.

Both mechanisms are pure functions with no side effects, making them
easy to test and reason about independently of storage backends.
"""

from __future__ import annotations

import math


def compute_spacing_quality(timestamps: list[float]) -> float:
    """
    Compute how well-spaced memory accesses are (0.0–1.0).

    Combines two signals:
    - Mean interval score: log-normalized average gap between accesses.
      A mean interval of ~1 week normalizes to ~1.0.
    - Expansion ratio: fraction of consecutive interval pairs where the
      later interval is larger than the earlier one. Expanding intervals
      (1h -> 1d -> 1w) are the hallmark of effective spaced repetition.

    Args:
        timestamps: List of access timestamps (epoch floats, any order).

    Returns:
        0.0 for < 2 accesses or fully clustered access,
        up to 1.0 for well-spaced, expanding intervals.
    """
    if len(timestamps) < 2:
        return 0.0

    sorted_ts = sorted(timestamps)
    intervals = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]

    # Mean interval in hours, log-normalized
    # log(1 + 168) normalizer means ~1 week average interval → score ≈ 1.0
    mean_seconds = sum(intervals) / len(intervals)
    mean_hours = mean_seconds / 3600.0
    normalizer = math.log(1 + 168)  # 168 hours = 1 week
    interval_score = min(1.0, math.log(1 + mean_hours) / normalizer)

    # Expansion ratio: how many intervals are increasing?
    if len(intervals) >= 2:
        expanding = sum(1 for i in range(1, len(intervals)) if intervals[i] > intervals[i - 1])
        expansion_ratio = expanding / (len(intervals) - 1)
    else:
        # Single interval — no expansion signal, use neutral 0.5
        expansion_ratio = 0.5

    # Combine: well-spaced AND expanding = best
    return interval_score * (0.5 + 0.5 * expansion_ratio)


def apply_spacing_boost(
    base_score: float,
    spacing_quality: float,
    boost_weight: float = 0.1,
) -> float:
    """
    Apply spacing quality boost to a retrieval score.

    Formula: boosted = base_score * (1 + boost_weight * spacing_quality)

    Multiplicative so:
    - spacing_quality=0.0 → no change (no access history or clustered)
    - spacing_quality=1.0, weight=0.1 → +10% boost

    Args:
        base_score: Original retrieval score
        spacing_quality: Memory's spacing quality (0.0–1.0)
        boost_weight: Maximum boost factor

    Returns:
        Boosted score
    """
    return base_score * (1.0 + boost_weight * spacing_quality)


def compute_ltp_rate(
    base_rate: float,
    current_weight: float,
    max_weight: float,
    spacing_quality: float = 0.0,
) -> float:
    """
    Compute adaptive LTP-modulated Hebbian strengthen rate.

    Two dampening factors:
    1. Weight saturation: rate decreases as weight approaches max_weight.
       At weight=0, full rate. At weight=max_weight, rate=0.
    2. Spacing modulation: poorly-spaced access (cramming) only gets
       half the strengthen rate; well-spaced access gets full rate.

    Formula:
        effective_rate = base_rate
                         * (1 - current_weight / max_weight)
                         * (0.5 + 0.5 * spacing_quality)

    Args:
        base_rate: Base Hebbian strengthen rate (e.g., 0.15)
        current_weight: Current edge weight
        max_weight: Maximum allowed edge weight
        spacing_quality: How well-spaced the access pattern is (0.0–1.0)

    Returns:
        Effective strengthen rate (always >= 0)
    """
    if max_weight <= 0:
        return 0.0

    # Weight saturation: approaches 0 as weight nears max
    saturation = max(0.0, 1.0 - current_weight / max_weight)

    # Spacing modulation: 0.5 base + up to 0.5 bonus for good spacing
    spacing_mod = 0.5 + 0.5 * max(0.0, min(1.0, spacing_quality))

    return base_rate * saturation * spacing_mod
