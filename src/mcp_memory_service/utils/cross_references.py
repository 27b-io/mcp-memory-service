"""
Memory cross-referencing: candidate filtering utility.

Identifies which similar memories should receive RELATES_TO edges when a new
memory is stored. Operates on already-retrieved similarity results — no I/O,
pure filtering logic.

Design rationale:
    Cross-referencing complements contradiction detection (interference.py).
    Where interference creates CONTRADICTS edges for opposing memories,
    cross-referencing creates RELATES_TO edges for semantically related ones.
    Keeping the filtering logic here makes it independently testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def filter_cross_reference_candidates(
    new_hash: str,
    similar: list[Any],
    exclude_hashes: set[str] | None = None,
) -> list[str]:
    """
    Filter similarity results to produce cross-reference candidates.

    Given a list of MemoryQueryResult-like objects (anything with
    `.memory.content_hash`), returns the hashes that should receive a
    RELATES_TO edge from the new memory.

    Excludes:
    - The new memory itself (self-reference)
    - Hashes already tagged as contradictions (those get CONTRADICTS instead)
    - Duplicates (first occurrence wins, preserving order)

    Args:
        new_hash: Content hash of the memory being stored
        similar: Sequence of MemoryQueryResult-like objects with memory.content_hash
        exclude_hashes: Hashes to exclude (e.g., known contradictions)

    Returns:
        Ordered list of content hashes to create RELATES_TO edges to
    """
    exclude = exclude_hashes or set()
    seen: set[str] = set()
    result: list[str] = []

    for match in similar:
        h = match.memory.content_hash
        if h == new_hash or h in exclude or h in seen:
            continue
        seen.add(h)
        result.append(h)

    return result
