"""
Conflict resolution API endpoints for memory contradictions.

Provides endpoints to:
- List all detected contradictions (CONTRADICTS edges in the graph)
- Get details of specific conflicts
- Resolve conflicts using various strategies (keep/delete/merge)
"""

import logging
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...graph.client import GraphClient
from ...services.memory_service import MemoryService
from ...shared_storage import get_graph_client
from ...storage.base import MemoryStorage
from ..dependencies import get_memory_service, get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conflicts", tags=["conflicts"])


class ResolutionStrategy(str, Enum):
    """Strategies for resolving memory conflicts."""

    KEEP_NEW = "keep_new"  # Keep the newer memory, delete the older one
    KEEP_OLD = "keep_old"  # Keep the older memory, delete the newer one
    DELETE_BOTH = "delete_both"  # Delete both conflicting memories
    KEEP_BOTH = "keep_both"  # Acknowledge but keep both (delete the edge only)


class ConflictInfo(BaseModel):
    """Information about a detected conflict."""

    source_hash: str = Field(..., description="Content hash of the source memory")
    target_hash: str = Field(..., description="Content hash of the target memory")
    created_at: float | None = Field(None, description="Timestamp when conflict was detected")


class ConflictDetail(BaseModel):
    """Detailed information about a conflict including both memories."""

    source_hash: str
    target_hash: str
    created_at: float | None
    source_memory: dict[str, Any] | None = Field(None, description="Source memory content and metadata")
    target_memory: dict[str, Any] | None = Field(None, description="Target memory content and metadata")


class ConflictListResponse(BaseModel):
    """Response model for listing conflicts."""

    conflicts: list[ConflictInfo]
    total: int
    message: str | None = None


class ResolutionRequest(BaseModel):
    """Request model for resolving a conflict."""

    strategy: ResolutionStrategy = Field(..., description="Resolution strategy to apply")


class ResolutionResponse(BaseModel):
    """Response model for conflict resolution."""

    success: bool
    message: str
    deleted_hashes: list[str] = Field(default_factory=list, description="Hashes of memories that were deleted")


@router.get("/", response_model=ConflictListResponse)
async def list_conflicts(
    limit: int = 100,
    graph: GraphClient | None = Depends(get_graph_client),
) -> ConflictListResponse:
    """
    List all detected memory conflicts (CONTRADICTS edges).

    Args:
        limit: Maximum number of conflicts to return

    Returns:
        List of conflicts with source/target hashes and detection timestamps
    """
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph client unavailable. Conflict detection requires FalkorDB.",
        )

    try:
        # Query all CONTRADICTS edges in the graph
        result = await graph.graph.query(
            "MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory) "
            "RETURN a.content_hash, b.content_hash, e.created_at "
            "ORDER BY e.created_at DESC "
            "LIMIT $lim",
            params={"lim": limit},
        )

        conflicts = [
            ConflictInfo(
                source_hash=row[0],
                target_hash=row[1],
                created_at=float(row[2]) if row[2] is not None else None,
            )
            for row in result.result_set
        ]

        return ConflictListResponse(
            conflicts=conflicts,
            total=len(conflicts),
            message=f"Found {len(conflicts)} conflict(s)" if conflicts else "No conflicts detected",
        )

    except Exception as e:
        logger.error(f"Failed to list conflicts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conflicts: {str(e)}") from e


@router.get("/{source_hash}/{target_hash}", response_model=ConflictDetail)
async def get_conflict_detail(
    source_hash: str,
    target_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    graph: GraphClient | None = Depends(get_graph_client),
) -> ConflictDetail:
    """
    Get detailed information about a specific conflict, including both memories.

    Args:
        source_hash: Content hash of the source memory
        target_hash: Content hash of the target memory

    Returns:
        Detailed conflict information with full memory content
    """
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph client unavailable. Conflict detection requires FalkorDB.",
        )

    try:
        # Verify the CONTRADICTS edge exists
        result = await graph.graph.query(
            "MATCH (a:Memory {content_hash: $src})-[e:CONTRADICTS]->(b:Memory {content_hash: $dst}) RETURN e.created_at",
            params={"src": source_hash, "dst": target_hash},
        )

        if not result.result_set:
            raise HTTPException(
                status_code=404,
                detail=f"No conflict found between {source_hash[:8]} and {target_hash[:8]}",
            )

        created_at = float(result.result_set[0][0]) if result.result_set[0][0] is not None else None

        # Retrieve both memories from storage
        source_results = await storage.retrieve(query="", content_hash=source_hash, n_results=1)
        target_results = await storage.retrieve(query="", content_hash=target_hash, n_results=1)

        source_memory = source_results[0].memory.to_dict() if source_results else None
        target_memory = target_results[0].memory.to_dict() if target_results else None

        return ConflictDetail(
            source_hash=source_hash,
            target_hash=target_hash,
            created_at=created_at,
            source_memory=source_memory,
            target_memory=target_memory,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conflict detail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conflict detail: {str(e)}") from e


@router.post("/{source_hash}/{target_hash}/resolve", response_model=ResolutionResponse)
async def resolve_conflict(
    source_hash: str,
    target_hash: str,
    request: ResolutionRequest,
    service: MemoryService = Depends(get_memory_service),
    storage: MemoryStorage = Depends(get_storage),
    graph: GraphClient | None = Depends(get_graph_client),
) -> ResolutionResponse:
    """
    Resolve a memory conflict using the specified strategy.

    Strategies:
    - keep_new: Keep newer memory (by timestamp), delete older
    - keep_old: Keep older memory, delete newer
    - delete_both: Delete both conflicting memories
    - keep_both: Keep both memories but remove the CONTRADICTS edge

    Args:
        source_hash: Content hash of the source memory
        target_hash: Content hash of the target memory
        request: Resolution strategy to apply

    Returns:
        Success status and list of deleted memory hashes
    """
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph client unavailable. Conflict resolution requires FalkorDB.",
        )

    try:
        # Verify the conflict exists
        conflict_result = await graph.graph.query(
            "MATCH (a:Memory {content_hash: $src})-[e:CONTRADICTS]->(b:Memory {content_hash: $dst}) RETURN e.created_at",
            params={"src": source_hash, "dst": target_hash},
        )

        if not conflict_result.result_set:
            raise HTTPException(
                status_code=404,
                detail=f"No conflict found between {source_hash[:8]} and {target_hash[:8]}",
            )

        deleted_hashes: list[str] = []

        if request.strategy == ResolutionStrategy.KEEP_BOTH:
            # Just delete the CONTRADICTS edge, keep both memories
            await graph.delete_typed_edge(source_hash, target_hash, "CONTRADICTS")
            return ResolutionResponse(
                success=True,
                message="Conflict edge removed, both memories preserved",
                deleted_hashes=[],
            )

        # For other strategies, we need to get the memories to check timestamps
        source_results = await storage.retrieve(query="", content_hash=source_hash, n_results=1)
        target_results = await storage.retrieve(query="", content_hash=target_hash, n_results=1)

        if not source_results or not target_results:
            raise HTTPException(
                status_code=404,
                detail="One or both memories not found in storage",
            )

        source_mem = source_results[0].memory
        target_mem = target_results[0].memory

        if request.strategy == ResolutionStrategy.DELETE_BOTH:
            # Delete both memories
            await storage.delete(source_hash)
            await storage.delete(target_hash)
            deleted_hashes = [source_hash, target_hash]
            message = "Both conflicting memories deleted"

        elif request.strategy == ResolutionStrategy.KEEP_NEW:
            # Keep newer, delete older
            if source_mem.created_at >= target_mem.created_at:
                await storage.delete(target_hash)
                deleted_hashes = [target_hash]
                message = f"Kept newer memory ({source_hash[:8]}), deleted older ({target_hash[:8]})"
            else:
                await storage.delete(source_hash)
                deleted_hashes = [source_hash]
                message = f"Kept newer memory ({target_hash[:8]}), deleted older ({source_hash[:8]})"

        elif request.strategy == ResolutionStrategy.KEEP_OLD:
            # Keep older, delete newer
            if source_mem.created_at <= target_mem.created_at:
                await storage.delete(target_hash)
                deleted_hashes = [target_hash]
                message = f"Kept older memory ({source_hash[:8]}), deleted newer ({target_hash[:8]})"
            else:
                await storage.delete(source_hash)
                deleted_hashes = [source_hash]
                message = f"Kept older memory ({target_hash[:8]}), deleted newer ({source_hash[:8]})"

        # The CONTRADICTS edge is automatically removed when either memory is deleted
        # because of DETACH DELETE in the storage layer or will be cleaned up during orphan pruning

        return ResolutionResponse(
            success=True,
            message=message,
            deleted_hashes=deleted_hashes,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve conflict: {str(e)}") from e
