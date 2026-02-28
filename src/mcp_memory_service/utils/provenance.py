"""
Memory provenance tracking utilities.

Tracks the origin, creation method, and modification history of memories.
Supports trust scoring based on source reliability to enable filtering by
source quality.

Trust scores range from 0.0 (untrusted) to 1.0 (fully trusted). Sources not
in the DEFAULT_SOURCE_TRUST map receive DEFAULT_TRUST_SCORE (neutral).
"""

import math
import time
from typing import Any

# Neutral trust score for memories without provenance or with invalid scores.
DEFAULT_TRUST_SCORE: float = 0.5

# Default trust scores by source type.
# Add sources here to set their default reliability.
DEFAULT_SOURCE_TRUST: dict[str, float] = {
    "api": 0.8,
    "working_memory_consolidation": 0.9,
    "batch": 0.7,
    "auto_split": 0.8,
    "import": 0.6,
    "unknown": DEFAULT_TRUST_SCORE,
}

# Keys that build_provenance() sets — extra dict must not overwrite these.
_RESERVED_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "creation_method",
        "trust_score",
        "created_at",
        "modification_history",
    }
)


def compute_trust_score(source: str) -> float:
    """Return trust score for a given source string.

    Supports hostname-prefixed sources like "source:hostname" (strips prefix).
    Returns DEFAULT_TRUST_SCORE for unknown sources.
    """
    if source.startswith("source:"):
        source = source[len("source:") :]
    return DEFAULT_SOURCE_TRUST.get(source, DEFAULT_TRUST_SCORE)


def build_provenance(
    source: str,
    creation_method: str,
    *,
    actor: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict for a newly stored memory.

    Args:
        source: Originating source identifier (e.g. "api", hostname, "consolidation")
        creation_method: How the memory was created ("direct", "auto_split", "batch", "consolidation")
        actor: Optional agent/client identifier (e.g. client_hostname)
        extra: Optional additional fields to merge into the provenance record.
            Reserved keys (source, trust_score, etc.) are silently ignored.

    Returns:
        Provenance dict suitable for storage in metadata["provenance"]
    """
    trust_score = compute_trust_score(source)
    record: dict[str, Any] = {
        "source": source,
        "creation_method": creation_method,
        "trust_score": trust_score,
        "created_at": time.time(),
        "modification_history": [],
    }
    if actor:
        record["actor"] = actor
    if extra:
        safe_extra = {k: v for k, v in extra.items() if k not in _RESERVED_PROVENANCE_KEYS}
        record.update(safe_extra)
    return record


def _extract_trust_from_provenance(provenance: Any) -> float | None:
    """Extract trust_score from a provenance dict, returning None if invalid."""
    if not isinstance(provenance, dict):
        return None
    raw = provenance.get("trust_score")
    if raw is None:
        return None
    score = float(raw)
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def resolve_trust_score(result: dict[str, Any]) -> float:
    """Resolve the canonical trust score from a search result dict.

    Checks top-level provenance first (canonical post-roundtrip location),
    then falls back to metadata.provenance (pre-roundtrip location).
    Returns DEFAULT_TRUST_SCORE if neither location has valid provenance.

    Handles NaN/inf by treating them as missing.
    """
    # Top-level provenance is canonical (post-roundtrip)
    score = _extract_trust_from_provenance(result.get("provenance"))
    if score is not None:
        return score

    # Fallback: metadata.provenance (pre-roundtrip)
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        score = _extract_trust_from_provenance(metadata.get("provenance"))
        if score is not None:
            return score

    return DEFAULT_TRUST_SCORE


def get_trust_score(memory_metadata: dict[str, Any]) -> float:
    """Extract trust score from a memory's metadata dict.

    Returns DEFAULT_TRUST_SCORE if no provenance is present (neutral default).
    """
    provenance = memory_metadata.get("provenance")
    if not isinstance(provenance, dict):
        return DEFAULT_TRUST_SCORE
    return float(provenance.get("trust_score", DEFAULT_TRUST_SCORE))
