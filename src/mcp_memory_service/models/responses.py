"""Service-layer response models.

Typed Pydantic models replacing the ``dict[str, Any]`` returns
throughout ``MemoryService``.  Every service method returns one of
these; callers get attribute access and IDE completion instead of
``result.get("key", default)`` roulette.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .validators import ContentHash

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ServiceResult(BaseModel):
    """Common base for operation results."""

    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Memory data (wire-format for API responses)
# ---------------------------------------------------------------------------


class MemoryData(BaseModel):
    """Serialised memory for response payloads.

    This is the *response* shape (what the caller sees), not the internal
    ``Memory`` model.  It intentionally excludes embeddings.
    """

    content: str
    content_hash: ContentHash
    tags: list[str] = Field(default_factory=list)
    memory_type: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: float | None = None
    updated_at: float | None = None
    created_at_iso: str | None = None
    updated_at_iso: str | None = None
    emotional_valence: dict[str, Any] | None = None
    salience_score: float = 0.0
    encoding_context: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    summary: str | None = None
    # Search-specific fields (populated when returned from search)
    similarity_score: float | None = None
    hybrid_debug: dict[str, Any] | None = None
    contradictions: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginationMeta(BaseModel):
    """Pagination metadata included in list/search responses.

    Used as a base class for paginated result models so that pagination
    fields stay centralised (DRY) and serialise flat in the wire format.
    """

    total: int = 0
    page: int = 1
    page_size: int = 10
    total_pages: int = 1
    has_more: bool = False


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class StoreResult(ServiceResult):
    """Result of a ``store_memory()`` call."""

    memory: MemoryData | None = None
    # Chunked storage
    memories: list[MemoryData] | None = None
    total_chunks: int | None = None
    original_hash: ContentHash | None = None
    # Interference detection
    interference: dict[str, Any] | None = None
    cross_references: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class DeleteResult(ServiceResult):
    """Result of a ``delete_memory()`` call."""

    content_hash: ContentHash | None = None


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class GetMemoryResult(BaseModel):
    """Result of a ``get_memory_by_hash()`` call."""

    found: bool = False
    content_hash: ContentHash | None = None
    memory: MemoryData | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResult(BaseModel):
    """Result of a ``check_database_health()`` call."""

    healthy: bool = True
    storage_type: str = "unknown"
    total_memories: int = 0
    last_updated: str | None = None
    graph: dict[str, Any] | None = None
    write_queue: dict[str, Any] | None = None
    error: str | None = None
    # Extra stats from storage.get_stats() are dynamic
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search / Retrieve
# ---------------------------------------------------------------------------


class RetrieveResult(PaginationMeta):
    """Result of a ``retrieve_memories()`` call."""

    memories: list[MemoryData] = Field(default_factory=list)
    query: str = ""
    hybrid_enabled: bool = True
    alpha_used: float | None = None
    keywords_extracted: list[str] = Field(default_factory=list)
    filtered_below_threshold: int | None = None
    error: str | None = None


class TagSearchResult(PaginationMeta):
    """Result of a ``search_by_tag()`` call."""

    memories: list[MemoryData] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    match_type: Literal["ALL", "ANY"] | None = None
    error: str | None = None


class ListResult(PaginationMeta):
    """Result of a ``list_memories()`` call."""

    memories: list[MemoryData] = Field(default_factory=list)
    error: str | None = None


class ScanResult(BaseModel):
    """Result of a ``scan_memories()`` call."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    total_scanned: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


class RelationResult(ServiceResult):
    """Result of create/delete relation operations."""

    source: ContentHash | None = None
    target: ContentHash | None = None
    relation_type: str | None = None


class GetRelationsResult(BaseModel):
    """Result of a ``get_relations()`` call."""

    relations: list[dict[str, Any]] = Field(default_factory=list)
    content_hash: ContentHash | None = None
    count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class DuplicateGroup(BaseModel):
    """A group of duplicate memories with a canonical choice."""

    hashes: list[ContentHash]
    canonical_hash: ContentHash
    max_similarity: float
    size: int


class FindDuplicatesResult(ServiceResult):
    """Result of a ``find_duplicates()`` call."""

    groups: list[DuplicateGroup] = Field(default_factory=list)
    total_memories_scanned: int = 0
    total_duplicates_found: int = 0


class MergeDuplicatesResult(ServiceResult):
    """Result of a ``merge_duplicate_group()`` call."""

    canonical_hash: ContentHash | None = None
    superseded: list[ContentHash] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Supersede
# ---------------------------------------------------------------------------


class SupersedeResult(ServiceResult):
    """Result of a ``supersede_memory()`` call."""

    superseded: ContentHash | None = None
    superseded_by: ContentHash | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------


class ContradictionPair(BaseModel):
    """A pair of contradictory memories."""

    memory_a_hash: ContentHash
    memory_b_hash: ContentHash
    confidence: float | None = None
    memory_a_content: str = ""
    memory_b_content: str = ""
    memory_a_superseded: bool = False
    memory_b_superseded: bool = False


class ContradictionsResult(ServiceResult):
    """Result of a ``get_contradictions_dashboard()`` call."""

    pairs: list[ContradictionPair] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


class ConsolidateResult(ServiceResult):
    """Result of a ``consolidate()`` call."""

    edges_decayed: int = 0
    stale_edges_decayed: int = 0
    edges_pruned: int = 0
    orphan_nodes: int = 0
    duplicates_found: int = 0
    duplicates_merged: int = 0
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


class BatchItemResult(BaseModel):
    """Result for a single item in a batch operation."""

    index: int
    success: bool
    content_hash: ContentHash | None = None
    error: str | None = None


class BatchResult(ServiceResult):
    """Result of batch store/delete/update operations.

    Operation-specific counters are optional — only the relevant one
    is populated per call (e.g. ``created`` for batch_store, ``deleted``
    for batch_delete).  ``failed`` is common to all operations.
    """

    results: list[BatchItemResult] = Field(default_factory=list)
    failed: int = 0
    # Operation-specific counters (mutually exclusive per call)
    created: int | None = None
    deleted: int | None = None
    updated: int | None = None
    rolled_back: bool | None = None
