# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Management endpoints for the HTTP interface.

Provides memory maintenance, bulk operations, and system management tools.
"""

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...config import OAUTH_ENABLED, settings
from ...storage.base import MemoryStorage
from ..dependencies import get_memory_service, get_storage

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import AuthenticationResult, require_write_access
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_write_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations."""

    tag: str | None = Field(None, description="Delete all memories with this tag")
    before_date: str | None = Field(None, description="Delete memories before this date (YYYY-MM-DD)")
    memory_type: str | None = Field(None, description="Delete memories of this type")
    confirm_count: int | None = Field(None, description="Confirmation of number of memories to delete")


class TagManagementRequest(BaseModel):
    """Request model for tag management operations."""

    operation: str = Field(..., description="Operation: 'rename', 'merge', or 'delete'")
    old_tag: str = Field(..., description="Original tag name")
    new_tag: str | None = Field(None, description="New tag name (for rename/merge)")
    confirm_count: int | None = Field(None, description="Confirmation count for destructive operations")


class BulkOperationResponse(BaseModel):
    """Response model for bulk operations."""

    success: bool
    message: str
    affected_count: int
    operation: str


class TagStatsResponse(BaseModel):
    """Response model for tag statistics."""

    tag: str
    count: int
    last_used: float | None
    memory_types: list[str]


class TagStatsListResponse(BaseModel):
    """Response model for tag statistics list."""

    tags: list[TagStatsResponse]
    total_tags: int


class SystemOperationRequest(BaseModel):
    """Request model for system operations."""

    operation: str = Field(..., description="Operation: 'cleanup_duplicates', 'optimize_db', 'rebuild_index'")


@router.post("/bulk-delete", response_model=BulkOperationResponse, tags=["management"])
async def bulk_delete_memories(
    request: BulkDeleteRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Perform bulk delete operations on memories.

    Supports deletion by tag, date range, or memory type.
    Requires confirmation count for safety.
    """
    try:
        affected_count = 0
        operation_desc = ""

        # Validate that at least one filter is provided
        if not any([request.tag, request.before_date, request.memory_type]):
            raise HTTPException(
                status_code=400, detail="At least one filter (tag, before_date, or memory_type) must be specified"
            )

        # Count memories that would be affected
        if request.tag:
            # Count memories with this tag
            if hasattr(storage, "count_memories_by_tag"):
                affected_count = await storage.count_memories_by_tag([request.tag])
            else:
                # Fallback: search and count
                tag_memories = await storage.search_by_tag([request.tag])
                affected_count = len(tag_memories)
            operation_desc = f"Delete memories with tag '{request.tag}'"

        elif request.before_date:
            # Count memories before date
            try:
                before_dt = datetime.fromisoformat(request.before_date)
                before_dt.timestamp()
                # This would need a method to count by date range
                # For now, we'll estimate or implement a simple approach
                affected_count = 0  # Placeholder
                operation_desc = f"Delete memories before {request.before_date}"
            except ValueError as e:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD") from e

        elif request.memory_type:
            # Count memories by type
            if hasattr(storage, "count_all_memories"):
                affected_count = await storage.count_all_memories(memory_type=request.memory_type)
            else:
                affected_count = 0  # Placeholder
            operation_desc = f"Delete memories of type '{request.memory_type}'"

        # Safety check: require confirmation count
        if request.confirm_count is not None and request.confirm_count != affected_count:
            raise HTTPException(
                status_code=400, detail=f"Confirmation count mismatch. Expected {affected_count}, got {request.confirm_count}"
            )

        # Perform the deletion
        success = False
        message = ""

        if request.tag:
            if hasattr(storage, "delete_by_tag"):
                success_count, message = await storage.delete_by_tag(request.tag)
                success = success_count > 0
                affected_count = success_count
            else:
                raise HTTPException(status_code=501, detail="Tag-based deletion not supported by storage backend")

        elif request.before_date:
            # Implement date-based deletion
            # This would need to be implemented in the storage layer
            raise HTTPException(status_code=501, detail="Date-based bulk deletion not yet implemented")

        elif request.memory_type:
            # Implement type-based deletion
            # This would need to be implemented in the storage layer
            raise HTTPException(status_code=501, detail="Type-based bulk deletion not yet implemented")

        return BulkOperationResponse(
            success=success,
            message=message or f"Successfully deleted {affected_count} memories",
            affected_count=affected_count,
            operation=operation_desc,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk delete operation failed: {str(e)}") from e


@router.post("/cleanup-duplicates", response_model=BulkOperationResponse, tags=["management"])
async def cleanup_duplicates(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Clean up duplicate memories in the database.

    Removes duplicate entries based on content hash and merges metadata.
    """
    try:
        if hasattr(storage, "cleanup_duplicates"):
            count, message = await storage.cleanup_duplicates()
            return BulkOperationResponse(
                success=count > 0, message=message, affected_count=count, operation="cleanup_duplicates"
            )
        else:
            raise HTTPException(status_code=501, detail="Duplicate cleanup not supported by storage backend")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Duplicate cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Duplicate cleanup failed: {str(e)}") from e


@router.get("/tags/stats", response_model=TagStatsListResponse, tags=["management"])
async def get_tag_statistics(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Get detailed statistics for all tags.

    Returns tag usage counts, last usage times, and associated memory types.
    """
    try:
        # Get all tags with counts
        if hasattr(storage, "get_all_tags_with_counts"):
            tag_data = await storage.get_all_tags_with_counts()

            # For now, provide basic tag stats without additional queries
            # TODO: Implement efficient batch queries in storage layer for last_used and memory_types
            enhanced_tags = []
            for tag_item in tag_data:
                enhanced_tags.append(
                    TagStatsResponse(
                        tag=tag_item["tag"],
                        count=tag_item["count"],
                        last_used=None,  # Would need efficient batch query
                        memory_types=[],  # Would need efficient batch query
                    )
                )

            return TagStatsListResponse(tags=enhanced_tags, total_tags=len(enhanced_tags))
        else:
            raise HTTPException(status_code=501, detail="Tag statistics not supported by storage backend")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tag statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag statistics: {str(e)}") from e


@router.put("/tags/{old_tag}", response_model=BulkOperationResponse, tags=["management"])
async def rename_tag(
    old_tag: str,
    new_tag: str,
    confirm_count: int | None = Query(None, description="Confirmation count"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Rename a tag across all memories.

    Updates all memories that have the old tag to use the new tag instead.
    """
    try:
        # Count memories with this tag
        if hasattr(storage, "count_memories_by_tag"):
            affected_count = await storage.count_memories_by_tag([old_tag])
        else:
            tag_memories = await storage.search_by_tag([old_tag])
            affected_count = len(tag_memories)

        # Safety check
        if confirm_count is not None and confirm_count != affected_count:
            raise HTTPException(
                status_code=400, detail=f"Confirmation count mismatch. Expected {affected_count}, got {confirm_count}"
            )

        # Implement tag renaming (this would need to be implemented in storage layer)
        # For now, return not implemented
        raise HTTPException(status_code=501, detail="Tag renaming not yet implemented")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tag rename failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tag rename failed: {str(e)}") from e


@router.post("/system/{operation}", response_model=BulkOperationResponse, tags=["management"])
async def perform_system_operation(
    operation: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Perform system maintenance operations.

    Supported operations: cleanup_duplicates, optimize_db, rebuild_index
    """
    try:
        if operation == "cleanup_duplicates":
            return await cleanup_duplicates(storage, user)

        elif operation == "optimize_db":
            # Database optimization (would need storage-specific implementation)
            raise HTTPException(status_code=501, detail="Database optimization not yet implemented")

        elif operation == "rebuild_index":
            # Rebuild search indexes (would need storage-specific implementation)
            raise HTTPException(status_code=501, detail="Index rebuilding not yet implemented")

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System operation {operation} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System operation failed: {str(e)}") from e


# =============================================================================
# Conflict Resolution Endpoints
# =============================================================================


class ConflictResponse(BaseModel):
    """Response model for a single conflict."""

    source_hash: str
    target_hash: str
    confidence: float | None
    signal_type: str | None
    created_at: float | None
    source_memory: dict[str, Any] | None = None
    target_memory: dict[str, Any] | None = None


class ConflictListResponse(BaseModel):
    """Response model for conflict list."""

    conflicts: list[ConflictResponse]
    total: int


class ConflictResolveRequest(BaseModel):
    """Request model for conflict resolution."""

    action: str = Field(
        ...,
        description="Resolution action: 'keep_source', 'keep_target', 'merge', 'dismiss'",
    )
    merge_tags: bool = Field(
        default=True,
        description="When action='merge', whether to merge tags from both memories",
    )
    custom_content: str | None = Field(
        None,
        description="Custom content for merged memory (overrides both sources)",
    )


class ConflictResolveResponse(BaseModel):
    """Response model for conflict resolution."""

    success: bool
    action: str
    message: str
    kept_hash: str | None = None
    deleted_hash: str | None = None


@router.get("/conflicts", response_model=ConflictListResponse, tags=["conflicts"])
async def list_conflicts(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of conflicts to return"),
    include_memories: bool = Query(False, description="Include full memory content for each conflict"),
    memory_service = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    List unresolved memory conflicts.

    Returns memories with CONTRADICTS edges that haven't been resolved yet.
    """
    try:
        if memory_service._graph is None:
            raise HTTPException(
                status_code=503,
                detail="Graph layer not enabled. Set MCP_FALKORDB_ENABLED=true to use conflict resolution.",
            )

        # Get unresolved conflicts from graph
        conflicts = await memory_service._graph.list_unresolved_conflicts(limit=limit)

        # Optionally fetch full memory content
        results = []
        for conflict in conflicts:
            conflict_data = ConflictResponse(
                source_hash=conflict["source"],
                target_hash=conflict["target"],
                confidence=conflict.get("confidence"),
                signal_type=conflict.get("signal_type"),
                created_at=conflict.get("created_at"),
            )

            if include_memories:
                # Fetch both memories
                source_mem = await memory_service.storage.get_memory_by_hash(conflict["source"])
                target_mem = await memory_service.storage.get_memory_by_hash(conflict["target"])

                if source_mem:
                    conflict_data.source_memory = memory_service._format_memory_response(source_mem)
                if target_mem:
                    conflict_data.target_memory = memory_service._format_memory_response(target_mem)

            results.append(conflict_data)

        return ConflictListResponse(conflicts=results, total=len(results))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list conflicts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list conflicts: {str(e)}") from e


@router.get("/conflicts/{source_hash}/{target_hash}", response_model=ConflictResponse, tags=["conflicts"])
async def get_conflict_details(
    source_hash: str,
    target_hash: str,
    memory_service = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Get detailed information about a specific conflict.

    Returns both memories involved in the conflict and the contradiction details.
    """
    try:
        if memory_service._graph is None:
            raise HTTPException(
                status_code=503,
                detail="Graph layer not enabled.",
            )

        # Check if CONTRADICTS edge exists
        edges = await memory_service._graph.get_typed_edges(
            content_hash=source_hash,
            relation_type="CONTRADICTS",
            direction="outgoing",
        )

        # Find the edge to target
        edge = next((e for e in edges if e["target"] == target_hash), None)
        if not edge:
            raise HTTPException(
                status_code=404,
                detail=f"No CONTRADICTS edge found from {source_hash} to {target_hash}",
            )

        # Fetch both memories
        source_mem = await memory_service.storage.get_memory_by_hash(source_hash)
        target_mem = await memory_service.storage.get_memory_by_hash(target_hash)

        if not source_mem or not target_mem:
            raise HTTPException(
                status_code=404,
                detail="One or both memories not found in storage",
            )

        return ConflictResponse(
            source_hash=source_hash,
            target_hash=target_hash,
            confidence=edge.get("confidence"),
            signal_type=edge.get("signal_type"),
            created_at=edge.get("created_at"),
            source_memory=memory_service._format_memory_response(source_mem),
            target_memory=memory_service._format_memory_response(target_mem),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conflict details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conflict details: {str(e)}") from e


@router.post("/conflicts/{source_hash}/{target_hash}/resolve", response_model=ConflictResolveResponse, tags=["conflicts"])
async def resolve_conflict(
    source_hash: str,
    target_hash: str,
    request: ConflictResolveRequest,
    memory_service = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Manually resolve a memory conflict.

    Actions:
    - keep_source: Keep source memory, delete target, mark edge as resolved
    - keep_target: Keep target memory, delete source, mark edge as resolved
    - merge: Merge both memories (keep source, merge tags, delete target)
    - dismiss: Mark conflict as resolved without any changes to memories
    """
    try:
        if memory_service._graph is None:
            raise HTTPException(status_code=503, detail="Graph layer not enabled.")

        # Validate action
        valid_actions = {"keep_source", "keep_target", "merge", "dismiss"}
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action '{request.action}'. Must be one of: {', '.join(valid_actions)}",
            )

        # Fetch both memories
        source_mem = await memory_service.storage.get_memory_by_hash(source_hash)
        target_mem = await memory_service.storage.get_memory_by_hash(target_hash)

        if not source_mem or not target_mem:
            raise HTTPException(status_code=404, detail="One or both memories not found")

        # Execute resolution action
        kept_hash = None
        deleted_hash = None
        message = ""

        if request.action == "keep_source":
            # Delete target, keep source
            await memory_service.delete_memory(target_hash)
            kept_hash = source_hash
            deleted_hash = target_hash
            message = f"Kept source memory {source_hash[:8]}, deleted target {target_hash[:8]}"

        elif request.action == "keep_target":
            # Delete source, keep target
            await memory_service.delete_memory(source_hash)
            kept_hash = target_hash
            deleted_hash = source_hash
            message = f"Kept target memory {target_hash[:8]}, deleted source {source_hash[:8]}"

        elif request.action == "merge":
            # Merge tags and metadata, keep source
            if request.merge_tags:
                merged_tags = list(set(source_mem.tags) | set(target_mem.tags))
                # Update source with merged tags
                await memory_service.storage.update_memory_metadata(
                    source_hash,
                    {"tags_str": ",".join(merged_tags)},
                    preserve_timestamps=True,
                )

            # Record merge in conflict history
            source_mem.conflict_history = source_mem.conflict_history or []
            source_mem.conflict_history.append(
                {
                    "resolved_at": time.time(),
                    "strategy": "manual_merge",
                    "merged_from": target_hash,
                    "merged_tags": request.merge_tags,
                }
            )
            await memory_service.storage.update_memory_metadata(
                source_hash,
                {"conflict_history": source_mem.conflict_history},
                preserve_timestamps=True,
            )

            # Delete target
            await memory_service.delete_memory(target_hash)
            kept_hash = source_hash
            deleted_hash = target_hash
            message = f"Merged memories, kept {source_hash[:8]}, deleted {target_hash[:8]}"

        elif request.action == "dismiss":
            # Just mark as resolved, don't modify memories
            message = f"Conflict dismissed without changes"

        # Mark CONTRADICTS edge as resolved
        await memory_service._graph.update_typed_edge_metadata(
            source_hash=source_hash,
            target_hash=target_hash,
            relation_type="CONTRADICTS",
            metadata={
                "resolved_at": time.time(),
                "resolved_by": "manual",
                "resolution_action": request.action,
            },
        )

        return ConflictResolveResponse(
            success=True,
            action=request.action,
            message=message,
            kept_hash=kept_hash,
            deleted_hash=deleted_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve conflict: {str(e)}") from e
