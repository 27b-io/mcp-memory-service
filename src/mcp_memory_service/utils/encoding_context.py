"""
Encoding context capture and context-dependent retrieval boosting.

Implements the encoding specificity principle (Tulving & Thomson, 1973):
memories encoded in a particular context are more effectively retrieved
when that same context is present at retrieval time.

Context dimensions captured at storage time:
    - time_of_day: Categorical temporal bucket (morning/afternoon/evening/night)
    - day_type: Weekday vs weekend
    - agent: Who/what created the memory (client hostname or agent name)
    - task_tags: Primary tags at encoding time (project/topic context)

Context similarity is computed as a weighted mean of per-dimension matches,
then applied as a multiplicative retrieval boost:
    final_score = base_score * (1 + boost_weight * context_similarity)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Time-of-day buckets (hour ranges, UTC)
# ---------------------------------------------------------------------------
_TIME_BUCKETS = {
    "morning": range(6, 12),  # 06:00–11:59
    "afternoon": range(12, 18),  # 12:00–17:59
    "evening": range(18, 22),  # 18:00–21:59
    "night": list(range(22, 24)) + list(range(0, 6)),  # 22:00–05:59
}

# Adjacency for partial-match scoring (1 step away = 0.5)
_BUCKET_ORDER = ["morning", "afternoon", "evening", "night"]


def _hour_to_bucket(hour: int) -> str:
    """Map an hour (0-23) to a time-of-day bucket."""
    for name, hours in _TIME_BUCKETS.items():
        if hour in hours:
            return name
    return "night"  # fallback


def _bucket_distance(a: str, b: str) -> int:
    """Circular distance between two time buckets (0, 1, or 2)."""
    if a == b:
        return 0
    ia = _BUCKET_ORDER.index(a)
    ib = _BUCKET_ORDER.index(b)
    dist = abs(ia - ib)
    return min(dist, len(_BUCKET_ORDER) - dist)


# ---------------------------------------------------------------------------
# EncodingContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodingContext:
    """Environmental context captured at memory encoding time."""

    time_of_day: str = "morning"  # morning | afternoon | evening | night
    day_type: str = "weekday"  # weekday | weekend
    agent: str = ""  # Source agent/hostname
    task_tags: tuple[str, ...] = ()  # Primary tags at encoding time

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for storage in Qdrant payload."""
        return {
            "time_of_day": self.time_of_day,
            "day_type": self.day_type,
            "agent": self.agent,
            "task_tags": list(self.task_tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncodingContext:
        """Deserialize from a stored dict."""
        return cls(
            time_of_day=data.get("time_of_day", "morning"),
            day_type=data.get("day_type", "weekday"),
            agent=data.get("agent", ""),
            task_tags=tuple(data.get("task_tags", [])),
        )


# ---------------------------------------------------------------------------
# Context capture
# ---------------------------------------------------------------------------


def capture_encoding_context(
    tags: list[str] | None = None,
    agent: str = "",
    timestamp: float | None = None,
) -> EncodingContext:
    """
    Capture the current environmental context for a memory being stored.

    Args:
        tags: Tags being applied to the memory (used as task context)
        agent: Agent/hostname identifier
        timestamp: Override timestamp (defaults to now, UTC)

    Returns:
        EncodingContext with all dimensions populated
    """
    ts = timestamp or time.time()
    dt = datetime.fromtimestamp(ts, tz=UTC)

    return EncodingContext(
        time_of_day=_hour_to_bucket(dt.hour),
        day_type="weekend" if dt.weekday() >= 5 else "weekday",
        agent=agent or "",
        task_tags=tuple(tags[:10]) if tags else (),  # Cap at 10 for storage
    )


# ---------------------------------------------------------------------------
# Context similarity
# ---------------------------------------------------------------------------


def compute_context_similarity(
    stored: EncodingContext,
    current: EncodingContext,
    time_weight: float = 0.25,
    day_weight: float = 0.10,
    agent_weight: float = 0.25,
    task_weight: float = 0.40,
) -> float:
    """
    Compute similarity between a stored encoding context and the current context.

    Each dimension produces a 0.0–1.0 score, combined as a weighted mean.

    Dimension scoring:
        - time_of_day: Same bucket=1.0, adjacent=0.5, opposite=0.0
        - day_type: Same=1.0, different=0.0
        - agent: Exact match=1.0, no match=0.0
        - task_tags: Jaccard similarity of tag sets

    Args:
        stored: The encoding context captured at storage time
        current: The context at retrieval time
        time_weight: Weight for time-of-day dimension
        day_weight: Weight for day-type dimension
        agent_weight: Weight for agent dimension
        task_weight: Weight for task/tag dimension

    Returns:
        Context similarity score between 0.0 and 1.0
    """
    # Time-of-day: distance-based
    dist = _bucket_distance(stored.time_of_day, current.time_of_day)
    time_score = {0: 1.0, 1: 0.5}.get(dist, 0.0)

    # Day type: exact match
    day_score = 1.0 if stored.day_type == current.day_type else 0.0

    # Agent: exact match (case-insensitive)
    if stored.agent and current.agent:
        agent_score = 1.0 if stored.agent.lower() == current.agent.lower() else 0.0
    elif not stored.agent and not current.agent:
        agent_score = 0.5  # Both unknown = neutral
    else:
        agent_score = 0.0

    # Task tags: Jaccard similarity
    s_tags = set(stored.task_tags)
    c_tags = set(current.task_tags)
    if s_tags and c_tags:
        task_score = len(s_tags & c_tags) / len(s_tags | c_tags)
    elif not s_tags and not c_tags:
        task_score = 0.5  # Both empty = neutral
    else:
        task_score = 0.0

    # Weighted mean
    total_weight = time_weight + day_weight + agent_weight + task_weight
    if total_weight <= 0:
        return 0.0

    similarity = (
        time_weight * time_score + day_weight * day_score + agent_weight * agent_score + task_weight * task_score
    ) / total_weight

    return max(0.0, min(1.0, similarity))


# ---------------------------------------------------------------------------
# Retrieval boost
# ---------------------------------------------------------------------------


def apply_context_boost(
    base_score: float,
    context_similarity: float,
    boost_weight: float = 0.1,
) -> float:
    """
    Apply encoding-context boost to a retrieval score.

    Formula: boosted = base_score * (1 + boost_weight * context_similarity)

    This is multiplicative so:
    - similarity=0.0 → no change
    - similarity=1.0, weight=0.1 → +10% boost

    Args:
        base_score: Original retrieval score
        context_similarity: Context match score (0.0–1.0)
        boost_weight: Maximum boost factor

    Returns:
        Boosted score
    """
    return base_score * (1.0 + boost_weight * context_similarity)
