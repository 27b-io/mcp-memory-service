"""Memory-related data models.

Pydantic v2 models replacing the original dataclass-based Memory
and MemoryQueryResult with full validation and timestamp synchronisation.
"""

import calendar
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .validators import ContentHash, NonNegativeInt, Tags

# Try to import dateutil, but fall back to standard datetime parsing if not available
try:
    from dateutil import parser as dateutil_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timestamp helpers (module-level, shared by model validator and touch())
# ---------------------------------------------------------------------------

# Fields that are *not* metadata — everything else is overflow metadata
_KNOWN_FIELDS = frozenset(
    {
        "content",
        "content_hash",
        "tags_str",
        "type",
        "timestamp",
        "timestamp_float",
        "timestamp_str",
        "created_at",
        "created_at_iso",
        "updated_at",
        "updated_at_iso",
        "emotional_valence",
        "salience_score",
        "access_count",
        "access_timestamps",
        "encoding_context",
        "summary",
        "provenance",
    }
)


def _iso_to_float(iso_str: str) -> float:
    """Convert ISO string to float timestamp, ensuring UTC interpretation."""
    if DATEUTIL_AVAILABLE:
        return dateutil_parser.isoparse(iso_str).timestamp()

    try:
        if iso_str.endswith("Z"):
            dt = datetime.fromisoformat(iso_str[:-1])
            return calendar.timegm(dt.timetuple()) + dt.microsecond / 1_000_000.0
        elif "+" in iso_str or iso_str.count("-") > 2:
            return datetime.fromisoformat(iso_str).timestamp()
        else:
            dt = datetime.fromisoformat(iso_str)
            return calendar.timegm(dt.timetuple()) + dt.microsecond / 1_000_000.0
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
            return float(calendar.timegm(dt.timetuple()))
        except (ValueError, TypeError):
            logger.warning("Failed to parse timestamp '%s', using current time", iso_str)
            return time.time()


def _float_to_iso(ts: float) -> str:
    """Convert float timestamp to ISO string (UTC, Z-suffix)."""
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _sync_pair(
    ts_float: float | None,
    ts_iso: str | None,
    now: float,
    label: str,
) -> tuple[float, str]:
    """Synchronise a (float, iso) timestamp pair.

    Returns a consistent (float, iso) tuple, filling in whichever is missing
    and resolving mismatches.
    """
    if ts_float is not None and ts_iso is not None:
        try:
            parsed = _iso_to_float(ts_iso)
            diff = abs(ts_float - parsed)
            if diff > 1.0 and diff < 86400:
                logger.info("Timezone mismatch in %s (diff: %.1fs), preferring float", label, diff)
                return ts_float, _float_to_iso(ts_float)
            elif diff >= 86400:
                logger.warning("Large %s diff (%.1fs), using current time", label, diff)
                return now, _float_to_iso(now)
            return ts_float, ts_iso
        except Exception as e:
            logger.warning("Error parsing %s timestamps: %s, using float", label, e)
            ts_float = ts_float if ts_float is not None else now
            return ts_float, _float_to_iso(ts_float)

    if ts_float is not None:
        return ts_float, _float_to_iso(ts_float)

    if ts_iso:
        try:
            return _iso_to_float(ts_iso), ts_iso
        except ValueError as e:
            logger.warning("Invalid %s_iso: %s", label, e)

    return now, _float_to_iso(now)


# ---------------------------------------------------------------------------
# Safe numeric parsing helpers (for legacy/malformed storage data)
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert *v* to float, returning *default* on failure or non-finite values."""
    try:
        result = float(v)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    """Convert *v* to int, returning *default* on failure."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    """Represents a single memory entry with validated fields."""

    model_config = ConfigDict(populate_by_name=True)

    content: str = Field(min_length=1)
    content_hash: ContentHash
    tags: Tags = []
    # str (not Literal) — storage may contain legacy values
    memory_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    # Timestamps: model_validator syncs float <-> ISO automatically
    created_at: float | None = None
    created_at_iso: str | None = None
    updated_at: float | None = None
    updated_at_iso: str | None = None

    # Emotional tagging and salience scoring
    emotional_valence: dict[str, Any] | None = None
    salience_score: float = Field(default=0.0, ge=0.0, le=1.0)
    access_count: NonNegativeInt = 0

    # Spaced repetition: recent access timestamps (ring buffer, newest last)
    access_timestamps: list[float] = Field(default_factory=list)

    # Encoding context: environmental context captured at storage time
    encoding_context: dict[str, Any] | None = None

    # Extractive summary: one-line summary for token-efficient scanning
    summary: str | None = None

    # Provenance: source, creation method, trust score, modification history
    provenance: dict[str, Any] | None = None

    # Legacy timestamp field — computed from created_at, excluded from dumps
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), exclude=True)

    @model_validator(mode="after")
    def sync_timestamps(self) -> Self:
        """Synchronise float and ISO timestamp pairs, filling in missing values."""
        now = time.time()

        self.created_at, self.created_at_iso = _sync_pair(
            self.created_at,
            self.created_at_iso,
            now,
            "created_at",
        )
        self.updated_at, self.updated_at_iso = _sync_pair(
            self.updated_at,
            self.updated_at_iso,
            now,
            "updated_at",
        )

        # Keep legacy field in sync
        self.timestamp = datetime.fromtimestamp(self.created_at, timezone.utc)
        return self

    def touch(self) -> None:
        """Update the updated_at timestamps to the current time."""
        now = time.time()
        self.updated_at = now
        self.updated_at_iso = _float_to_iso(now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to storage-compatible dictionary with legacy field names."""
        return {
            "content": self.content,
            "content_hash": self.content_hash,
            "tags_str": ",".join(self.tags) if self.tags else "",
            "type": self.memory_type,
            # Legacy timestamp fields
            "timestamp": float(self.created_at),
            "timestamp_float": self.created_at,
            "timestamp_str": self.created_at_iso,
            # New timestamp fields
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "updated_at": self.updated_at,
            "updated_at_iso": self.updated_at_iso,
            # Emotional tagging and salience
            "emotional_valence": self.emotional_valence,
            "salience_score": self.salience_score,
            "access_count": self.access_count,
            # Spaced repetition
            "access_timestamps": self.access_timestamps,
            # Encoding context
            "encoding_context": self.encoding_context,
            # Extractive summary
            "summary": self.summary,
            # Provenance
            "provenance": self.provenance,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embedding: list[float] | None = None) -> "Memory":
        """Create a Memory instance from storage dictionary data.

        Handles legacy field names (tags_str, type, timestamp_float, etc.)
        and extracts overflow keys into metadata.
        """
        tags = data.get("tags_str", "").split(",") if data.get("tags_str") else []

        # Extract timestamps: prefer new fields, fall back to legacy
        created_at = data.get("created_at")
        created_at_iso = data.get("created_at_iso")
        updated_at = data.get("updated_at")
        updated_at_iso = data.get("updated_at_iso")

        if created_at is None and created_at_iso is None:
            if "timestamp_float" in data:
                created_at = _safe_float(data["timestamp_float"])
            elif "timestamp" in data:
                created_at = _safe_float(data["timestamp"])
            if "timestamp_str" in data and created_at_iso is None:
                created_at_iso = data["timestamp_str"]

        # Overflow keys → metadata
        metadata = {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}

        return cls(
            content=data["content"],
            content_hash=data["content_hash"],
            tags=[tag for tag in tags if tag],
            memory_type=data.get("type"),
            metadata=metadata,
            embedding=embedding,
            created_at=created_at,
            created_at_iso=created_at_iso,
            updated_at=updated_at,
            updated_at_iso=updated_at_iso,
            emotional_valence=data.get("emotional_valence"),
            salience_score=_safe_float(data.get("salience_score", 0.0)),
            access_count=_safe_int(data.get("access_count", 0)),
            access_timestamps=data.get("access_timestamps", []),
            encoding_context=data.get("encoding_context"),
            summary=data.get("summary"),
            provenance=data.get("provenance"),
        )


class MemoryQueryResult(BaseModel):
    """Memory query result with relevance score and debug information."""

    model_config = ConfigDict(populate_by_name=True)

    memory: Memory
    relevance_score: float
    debug_info: dict[str, Any] = Field(default_factory=dict)

    @property
    def similarity_score(self) -> float:
        """Alias for relevance_score for backward compatibility."""
        return self.relevance_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
            "similarity_score": self.relevance_score,
            "debug_info": self.debug_info,
        }
