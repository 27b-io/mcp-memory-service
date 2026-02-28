"""Shared Pydantic types and validators for reuse across models.

Centralises tag normalisation, range-clamped floats, content-hash
constraints, and Literal enums so every model speaks the same language.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field

# ---------------------------------------------------------------------------
# Tag normalisation
# ---------------------------------------------------------------------------


def normalize_tags(v: Any) -> list[str]:
    """Accept ``str | list | None`` and return a clean ``list[str]``.

    * ``"a, b, c"`` → ``["a", "b", "c"]``
    * ``["a", None, " b "]`` → ``["a", "b"]``
    * ``None`` → ``[]``
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    if isinstance(v, list):
        return [s for item in v if item is not None and (s := str(item).strip())]
    return []


Tags = Annotated[list[str], BeforeValidator(normalize_tags)]
"""Flexible tag input: accepts str, list, or None — always outputs list[str]."""


# ---------------------------------------------------------------------------
# Numeric types
# ---------------------------------------------------------------------------

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
"""Float clamped to [0.0, 1.0] — for scores, thresholds, similarities."""

NonNegativeInt = Annotated[int, Field(ge=0)]
"""Integer ≥ 0 — for counts, page numbers."""


# ---------------------------------------------------------------------------
# String constraints
# ---------------------------------------------------------------------------

ContentHash = Annotated[str, Field(min_length=1)]
"""Non-empty content hash identifier."""


# ---------------------------------------------------------------------------
# Literal enums
# ---------------------------------------------------------------------------

SearchMode = Literal["hybrid", "scan", "similar", "tag", "recent"]
MemoryType = Literal["note", "decision", "task", "reference"]
RelationType = Literal["RELATES_TO", "PRECEDES", "CONTRADICTS"]
OutputFormat = Literal["full", "summary", "both"]
