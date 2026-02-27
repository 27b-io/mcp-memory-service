"""
Memory provenance tracking utilities.

Tracks the origin, creation method, and modification history of memories.
Supports trust scoring based on source reliability to enable filtering by
source quality.

Trust scores range from 0.0 (untrusted) to 1.0 (fully trusted). Sources not
in the DEFAULT_SOURCE_TRUST map receive a score of 0.5 (neutral).
"""

import time
from typing import Any

# Default trust scores by source type.
# Add sources here to set their default reliability.
DEFAULT_SOURCE_TRUST: dict[str, float] = {
    "api": 0.8,
    "working_memory_consolidation": 0.9,
    "batch": 0.7,
    "auto_split": 0.8,
    "import": 0.6,
    "unknown": 0.5,
}

# Modification history is capped to prevent unbounded metadata growth
MAX_MODIFICATION_HISTORY = 50


def compute_trust_score(source: str) -> float:
    """Return trust score for a given source string.

    Supports hostname-prefixed sources like "source:hostname" (strips prefix).
    Returns 0.5 for unknown sources.
    """
    if source.startswith("source:"):
        source = source[len("source:") :]
    return DEFAULT_SOURCE_TRUST.get(source, 0.5)


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
        extra: Optional additional fields to merge into the provenance record

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
        record.update(extra)
    return record


def record_modification(
    provenance: dict[str, Any],
    operation: str,
    actor: str | None = None,
) -> dict[str, Any]:
    """Append a modification record to provenance modification_history.

    Returns a new provenance dict (does not mutate the original).
    History is capped at MAX_MODIFICATION_HISTORY entries (oldest dropped).
    """
    provenance = dict(provenance)
    history: list[dict[str, Any]] = list(provenance.get("modification_history", []))

    entry: dict[str, Any] = {
        "timestamp": time.time(),
        "operation": operation,
    }
    if actor:
        entry["actor"] = actor

    history.append(entry)
    if len(history) > MAX_MODIFICATION_HISTORY:
        history = history[-MAX_MODIFICATION_HISTORY:]

    provenance["modification_history"] = history
    return provenance


def get_trust_score(memory_metadata: dict[str, Any]) -> float:
    """Extract trust score from a memory's metadata dict.

    Returns 0.5 if no provenance is present (neutral default).
    """
    provenance = memory_metadata.get("provenance")
    if not isinstance(provenance, dict):
        return 0.5
    return float(provenance.get("trust_score", 0.5))
