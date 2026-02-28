"""MCP tool input models.

Pydantic models that replace the inline validation scattered across
``mcp_server.py``.  Each MCP tool function validates its inputs by
constructing the corresponding model — range clamping, mode checking,
and required-field logic all live here as declarative constraints.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from .validators import ContentHash, OutputFormat, RelationType, SearchMode, Tags


class StoreMemoryParams(BaseModel):
    """Validated input for the ``store_memory`` MCP tool."""

    content: str = Field(min_length=1)
    tags: Tags = []
    memory_type: str = "note"
    metadata: dict[str, Any] | None = None
    client_hostname: str | None = None
    summary: str | None = None


class SearchParams(BaseModel):
    """Validated input for the ``search`` MCP tool.

    All range clamping and mode/query cross-validation is handled here.
    """

    query: str = ""
    mode: SearchMode = "hybrid"
    tags: Tags = []
    match_all: bool = False
    k: int = Field(default=10, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    output: OutputFormat = "full"
    memory_type: str | None = None
    encoding_context: dict[str, Any] | None = None
    include_superseded: bool = False
    min_trust_score: float | None = None

    @field_validator("min_trust_score", mode="before")
    @classmethod
    def sanitize_trust_score(cls, v: Any) -> float | None:
        """Reject NaN/Inf and clamp to [0, 1]."""
        if v is None:
            return None
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return max(0.0, min(v, 1.0))

    @model_validator(mode="after")
    def query_required_for_modes(self) -> Self:
        """Ensure query is present for modes that require it."""
        if self.mode in {"hybrid", "scan", "similar"} and not self.query.strip():
            raise ValueError(f"query is required for '{self.mode}' mode")
        return self


class DeleteMemoryParams(BaseModel):
    """Validated input for the ``delete_memory`` MCP tool."""

    content_hash: ContentHash


class RelationParams(BaseModel):
    """Validated input for the ``relation`` MCP tool."""

    action: Literal["create", "get", "delete"]
    content_hash: ContentHash
    target_hash: ContentHash | None = None
    relation_type: RelationType | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        """Require target_hash and relation_type for create/delete."""
        if self.action in {"create", "delete"}:
            if not self.target_hash or not self.relation_type:
                raise ValueError(f"target_hash and relation_type required for '{self.action}'")
        return self


class SupersedeParams(BaseModel):
    """Validated input for the ``memory_supersede`` MCP tool."""

    old_id: ContentHash
    new_id: ContentHash
    reason: str = ""


class FindDuplicatesParams(BaseModel):
    """Validated input for the ``find_duplicates`` MCP tool."""

    similarity_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    limit: int = Field(default=500, ge=10, le=2000)
    strategy: Literal["keep_newest", "keep_oldest", "keep_most_accessed"] = "keep_newest"


class MergeDuplicatesParams(BaseModel):
    """Validated input for the ``merge_duplicates`` MCP tool."""

    canonical_hash: ContentHash
    duplicate_hashes: list[ContentHash] = Field(min_length=1)
    reason: str = "Merged by deduplication"
    dry_run: bool = False


class ContradictionsParams(BaseModel):
    """Validated input for the ``memory_contradictions`` MCP tool."""

    limit: int = Field(default=20, ge=1, le=100)
