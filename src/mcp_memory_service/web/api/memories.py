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
Memory CRUD endpoints for the HTTP interface.
"""

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ...config import INCLUDE_HOSTNAME, OAUTH_ENABLED
from ...models.memory import Memory
from ...services.memory_service import MemoryService
from ...storage.base import MemoryStorage
from ..dependencies import get_memory_service, get_storage
from ..sse import create_memory_deleted_event, create_memory_stored_event, sse_manager
from ..write_queue import write_queue

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import AuthenticationResult, require_read_access, require_write_access
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None
    require_write_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory."""

    content: str = Field(..., description="The memory content to store")
    tags: list[str] = Field(default=[], description="Tags to categorize the memory")
    memory_type: str | None = Field(None, description="Type of memory (e.g., 'note', 'reminder', 'fact')")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata for the memory")
    client_hostname: str | None = Field(None, description="Client machine hostname for source tracking")


class MemoryUpdateRequest(BaseModel):
    """Request model for updating memory metadata (tags, type, metadata only)."""

    tags: list[str] | None = Field(None, description="Updated tags to categorize the memory")
    memory_type: str | None = Field(None, description="Updated memory type (e.g., 'note', 'reminder', 'fact')")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata for the memory")


class MemoryResponse(BaseModel):
    """Response model for memory data."""

    content: str
    content_hash: str
    tags: list[str]
    memory_type: str | None
    metadata: dict[str, Any]
    created_at: float | None
    created_at_iso: str | None
    updated_at: float | None
    updated_at_iso: str | None


class MemoryListResponse(BaseModel):
    """Response model for paginated memory list."""

    memories: list[MemoryResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class MemoryCreateResponse(BaseModel):
    """Response model for memory creation."""

    success: bool
    message: str
    content_hash: str | None = None
    memory: MemoryResponse | None = None


class MemoryDeleteResponse(BaseModel):
    """Response model for memory deletion."""

    success: bool
    message: str
    content_hash: str


class MemoryUpdateResponse(BaseModel):
    """Response model for memory update."""

    success: bool
    message: str
    content_hash: str
    memory: MemoryResponse | None = None


class TagResponse(BaseModel):
    """Response model for a single tag with its count."""

    tag: str
    count: int


class TagListResponse(BaseModel):
    """Response model for tags list."""

    tags: list[TagResponse]


# Batch operation models
class BatchMemoryCreateRequest(BaseModel):
    """Request model for batch memory creation."""

    memories: list[MemoryCreateRequest] = Field(..., max_length=100, description="List of memories to create (max 100)")


class BatchMemoryItemResult(BaseModel):
    """Result for a single batch operation item."""

    success: bool
    content_hash: str | None = None
    message: str | None = None
    memory: MemoryResponse | None = None


class BatchMemoryCreateResponse(BaseModel):
    """Response model for batch memory creation."""

    total: int
    successful: int
    failed: int
    results: list[BatchMemoryItemResult]


class BatchMemoryUpdateItem(BaseModel):
    """Single update item in batch request."""

    content_hash: str = Field(..., description="Content hash of memory to update")
    tags: list[str] | None = Field(None, description="Updated tags")
    memory_type: str | None = Field(None, description="Updated memory type")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")


class BatchMemoryUpdateRequest(BaseModel):
    """Request model for batch memory updates."""

    updates: list[BatchMemoryUpdateItem] = Field(..., max_length=100, description="List of memory updates (max 100)")


class BatchMemoryUpdateResponse(BaseModel):
    """Response model for batch memory updates."""

    total: int
    successful: int
    failed: int
    results: list[BatchMemoryItemResult]


class BatchMemoryDeleteRequest(BaseModel):
    """Request model for batch memory deletion."""

    content_hashes: list[str] = Field(..., max_length=100, description="List of content hashes to delete (max 100)")


class BatchMemoryDeleteResponse(BaseModel):
    """Response model for batch memory deletion."""

    total: int
    successful: int
    failed: int
    results: list[BatchMemoryItemResult]


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Convert Memory model to response format."""
    return MemoryResponse(
        content=memory.content,
        content_hash=memory.content_hash,
        tags=memory.tags,
        memory_type=memory.memory_type,
        metadata=memory.metadata,
        created_at=memory.created_at,
        created_at_iso=memory.created_at_iso,
        updated_at=memory.updated_at,
        updated_at_iso=memory.updated_at_iso,
    )


@router.post("/memories", response_model=MemoryCreateResponse, tags=["memories"])
async def store_memory(
    request: MemoryCreateRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Store a new memory.

    Uses the MemoryService for consistent business logic including content processing,
    hostname tagging, and metadata enrichment. Write operations are queued to prevent
    contention from concurrent requests.
    """
    try:
        # Resolve hostname for consistent tagging (logic stays in API layer, tagging in service)
        client_hostname = None
        if INCLUDE_HOSTNAME:
            # Prioritize client-provided hostname, then header, then fallback to server
            # 1. Check if client provided hostname in request body
            if request.client_hostname:
                client_hostname = request.client_hostname
            # 2. Check for X-Client-Hostname header
            elif http_request.headers.get("X-Client-Hostname"):
                client_hostname = http_request.headers.get("X-Client-Hostname")
            # 3. Fallback to server hostname (original behavior)
            else:
                client_hostname = socket.gethostname()

        # Enqueue write operation to prevent contention
        # Returns a Future that will contain the result when processed
        result_future = await write_queue.enqueue(
            memory_service.store_memory,
            content=request.content,
            tags=request.tags,
            memory_type=request.memory_type,
            metadata=request.metadata,
            client_hostname=client_hostname,
        )

        # Trigger queue processor immediately (not as post-response background task)
        asyncio.create_task(write_queue.process_queue())

        # Wait for the result (queue processes concurrently)
        result = await result_future

        if result["success"]:
            # Broadcast SSE event for successful memory storage
            try:
                # Handle both single memory and chunked responses
                if "memory" in result:
                    memory_data = {
                        "content_hash": result["memory"]["content_hash"],
                        "content": result["memory"]["content"],
                        "tags": result["memory"]["tags"],
                        "memory_type": result["memory"]["memory_type"],
                    }
                else:
                    # For chunked responses, use the first chunk's data
                    first_memory = result["memories"][0]
                    memory_data = {
                        "content_hash": first_memory["content_hash"],
                        "content": first_memory["content"],
                        "tags": first_memory["tags"],
                        "memory_type": first_memory["memory_type"],
                    }

                event = create_memory_stored_event(memory_data)
                await sse_manager.broadcast_event(event)
            except Exception as e:
                # Don't fail the request if SSE broadcasting fails
                logger.warning(f"Failed to broadcast memory_stored event: {e}")

            # Return appropriate response based on MemoryService result
            if "memory" in result:
                # Single memory response
                return MemoryCreateResponse(
                    success=True,
                    message="Memory stored successfully",
                    content_hash=result["memory"]["content_hash"],
                    memory=result["memory"],
                )
            else:
                # Chunked memory response
                first_memory = result["memories"][0]
                return MemoryCreateResponse(
                    success=True,
                    message=f"Memory stored as {result['total_chunks']} chunks",
                    content_hash=first_memory["content_hash"],
                    memory=first_memory,
                )
        else:
            return MemoryCreateResponse(success=False, message=result.get("error", "Failed to store memory"), content_hash=None)

    except Exception as e:
        logger.error(f"Failed to store memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store memory. Please try again.") from e


@router.get("/memories", response_model=MemoryListResponse, tags=["memories"])
async def list_memories(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of memories per page"),
    tag: str | None = Query(None, description="Filter by tag"),
    memory_type: str | None = Query(None, description="Filter by memory type"),
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    List memories with pagination and optional filtering.

    Uses the MemoryService for consistent business logic and optimal database-level filtering.
    """
    try:
        # Use the injected service for consistent, performant memory listing
        result = await memory_service.list_memories(page=page, page_size=page_size, tag=tag, memory_type=memory_type)

        return MemoryListResponse(
            memories=result["memories"],
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            has_more=result["has_more"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}") from e


@router.get("/memories/{content_hash}", response_model=MemoryResponse, tags=["memories"])
async def get_memory(
    content_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Get a specific memory by its content hash.

    Retrieves a single memory entry using its unique content hash identifier.
    """
    try:
        # Use the new get_by_hash method for direct hash lookup
        memory = await storage.get_by_hash(content_hash)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        return memory_to_response(memory)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}") from e


@router.delete("/memories/{content_hash}", response_model=MemoryDeleteResponse, tags=["memories"])
async def delete_memory(
    content_hash: str,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Delete a memory by its content hash.

    Permanently removes a memory entry from the storage.
    """
    try:
        success, message = await storage.delete(content_hash)

        # Broadcast SSE event for memory deletion
        try:
            event = create_memory_deleted_event(content_hash, success)
            await sse_manager.broadcast_event(event)
        except Exception as e:
            # Don't fail the request if SSE broadcasting fails
            logger.warning(f"Failed to broadcast memory_deleted event: {e}")

        return MemoryDeleteResponse(success=success, message=message, content_hash=content_hash)

    except Exception as e:
        logger.error(f"Failed to delete memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete memory. Please try again.") from e


@router.put("/memories/{content_hash}", response_model=MemoryUpdateResponse, tags=["memories"])
async def update_memory(
    content_hash: str,
    request: MemoryUpdateRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Update memory metadata (tags, type, metadata) without changing content or timestamps.

    This endpoint allows updating only the metadata aspects of a memory while preserving
    the original content and creation timestamp. Only provided fields will be updated.
    """
    try:
        # First, check if the memory exists
        existing_memory = await storage.get_by_hash(content_hash)
        if not existing_memory:
            raise HTTPException(status_code=404, detail=f"Memory with hash {content_hash} not found")

        # Build the updates dictionary with only provided fields
        updates = {}
        if request.tags is not None:
            updates["tags"] = request.tags
        if request.memory_type is not None:
            updates["memory_type"] = request.memory_type
        if request.metadata is not None:
            updates["metadata"] = request.metadata

        # If no updates provided, return current memory
        if not updates:
            return MemoryUpdateResponse(
                success=True,
                message="No updates provided - memory unchanged",
                content_hash=content_hash,
                memory=memory_to_response(existing_memory),
            )

        # Perform the update
        success, message = await storage.update_memory_metadata(
            content_hash=content_hash, updates=updates, preserve_timestamps=True
        )

        if success:
            # Get the updated memory
            updated_memory = await storage.get_by_hash(content_hash)

            return MemoryUpdateResponse(
                success=True,
                message=message,
                content_hash=content_hash,
                memory=memory_to_response(updated_memory) if updated_memory else None,
            )
        else:
            return MemoryUpdateResponse(success=False, message=message, content_hash=content_hash)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}") from e


@router.get("/tags", response_model=TagListResponse, tags=["tags"])
async def get_tags(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """
    Get all tags with their usage counts.

    Returns a list of all unique tags along with how many memories use each tag,
    sorted by count in descending order.
    """
    try:
        # Get tags with counts from storage
        tag_data = await storage.get_all_tags_with_counts()

        # Convert to response format
        tags = [TagResponse(tag=item["tag"], count=item["count"]) for item in tag_data]

        return TagListResponse(tags=tags)

    except AttributeError as e:
        # Handle case where storage backend doesn't implement get_all_tags_with_counts
        raise HTTPException(status_code=501, detail=f"Tags endpoint not supported by current storage backend: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}") from e


@router.post("/memories/batch", response_model=BatchMemoryCreateResponse, tags=["memories", "batch"])
async def batch_store_memories(
    request: BatchMemoryCreateRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Store multiple memories in a single request.

    Supports batching up to 100 memories for efficiency. Each memory is processed
    independently, and partial failures are allowed (some memories may succeed while
    others fail).

    Returns detailed results for each memory including success status, content hash,
    and any error messages.
    """
    if not request.memories:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one memory")

    if len(request.memories) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100 memories")

    # Resolve hostname once for all memories (same logic as single store)
    client_hostname = None
    if INCLUDE_HOSTNAME:
        if http_request.headers.get("X-Client-Hostname"):
            client_hostname = http_request.headers.get("X-Client-Hostname")
        else:
            client_hostname = socket.gethostname()

    results: list[BatchMemoryItemResult] = []
    successful = 0
    failed = 0

    # Process each memory independently
    for mem_req in request.memories:
        try:
            # Use client_hostname from header if not specified in individual request
            hostname = mem_req.client_hostname or client_hostname

            # Enqueue write operation
            result_future = await write_queue.enqueue(
                memory_service.store_memory,
                content=mem_req.content,
                tags=mem_req.tags,
                memory_type=mem_req.memory_type,
                metadata=mem_req.metadata,
                client_hostname=hostname,
            )

            # Trigger queue processor
            asyncio.create_task(write_queue.process_queue())

            # Wait for result
            result = await result_future

            if result["success"]:
                successful += 1
                # Handle both single and chunked responses
                if "memory" in result:
                    memory_data = result["memory"]
                else:
                    memory_data = result["memories"][0]

                results.append(
                    BatchMemoryItemResult(
                        success=True,
                        content_hash=memory_data["content_hash"],
                        message="Memory stored successfully",
                        memory=MemoryResponse(**memory_data),
                    )
                )

                # Broadcast SSE event
                try:
                    event_data = {
                        "content_hash": memory_data["content_hash"],
                        "content": memory_data["content"],
                        "tags": memory_data["tags"],
                        "memory_type": memory_data["memory_type"],
                    }
                    event = create_memory_stored_event(event_data)
                    await sse_manager.broadcast_event(event)
                except Exception as e:
                    logger.warning(f"Failed to broadcast memory_stored event: {e}")

            else:
                failed += 1
                results.append(BatchMemoryItemResult(success=False, message=result.get("error", "Failed to store memory")))

        except Exception as e:
            failed += 1
            logger.error(f"Error storing memory in batch: {str(e)}")
            results.append(BatchMemoryItemResult(success=False, message=f"Error: {str(e)}"))

    return BatchMemoryCreateResponse(total=len(request.memories), successful=successful, failed=failed, results=results)


@router.put("/memories/batch", response_model=BatchMemoryUpdateResponse, tags=["memories", "batch"])
async def batch_update_memories(
    request: BatchMemoryUpdateRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Update metadata for multiple memories in a single request.

    Supports batching up to 100 memory updates. Each update is processed independently,
    allowing partial success (some updates may succeed while others fail).

    Only metadata fields (tags, type, metadata) can be updated. Content and timestamps
    are preserved.
    """
    if not request.updates:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one update")

    if len(request.updates) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100 updates")

    results: list[BatchMemoryItemResult] = []
    successful = 0
    failed = 0

    for update_item in request.updates:
        try:
            # Check if memory exists
            existing_memory = await storage.get_by_hash(update_item.content_hash)
            if not existing_memory:
                failed += 1
                results.append(
                    BatchMemoryItemResult(
                        success=False,
                        content_hash=update_item.content_hash,
                        message=f"Memory with hash {update_item.content_hash} not found",
                    )
                )
                continue

            # Build updates dictionary
            updates = {}
            if update_item.tags is not None:
                updates["tags"] = update_item.tags
            if update_item.memory_type is not None:
                updates["memory_type"] = update_item.memory_type
            if update_item.metadata is not None:
                updates["metadata"] = update_item.metadata

            # If no updates provided, skip
            if not updates:
                successful += 1
                results.append(
                    BatchMemoryItemResult(
                        success=True,
                        content_hash=update_item.content_hash,
                        message="No updates provided - memory unchanged",
                        memory=memory_to_response(existing_memory),
                    )
                )
                continue

            # Perform update
            success, message = await storage.update_memory_metadata(
                content_hash=update_item.content_hash, updates=updates, preserve_timestamps=True
            )

            if success:
                successful += 1
                updated_memory = await storage.get_by_hash(update_item.content_hash)
                results.append(
                    BatchMemoryItemResult(
                        success=True,
                        content_hash=update_item.content_hash,
                        message=message,
                        memory=memory_to_response(updated_memory) if updated_memory else None,
                    )
                )
            else:
                failed += 1
                results.append(BatchMemoryItemResult(success=False, content_hash=update_item.content_hash, message=message))

        except Exception as e:
            failed += 1
            logger.error(f"Error updating memory {update_item.content_hash} in batch: {str(e)}")
            results.append(
                BatchMemoryItemResult(success=False, content_hash=update_item.content_hash, message=f"Error: {str(e)}")
            )

    return BatchMemoryUpdateResponse(total=len(request.updates), successful=successful, failed=failed, results=results)


@router.delete("/memories/batch", response_model=BatchMemoryDeleteResponse, tags=["memories", "batch"])
async def batch_delete_memories(
    request: BatchMemoryDeleteRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None,
):
    """
    Delete multiple memories in a single request.

    Supports batching up to 100 memory deletions. Each deletion is processed independently,
    allowing partial success (some deletions may succeed while others fail).
    """
    if not request.content_hashes:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one content hash")

    if len(request.content_hashes) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100 deletions")

    results: list[BatchMemoryItemResult] = []
    successful = 0
    failed = 0

    for content_hash in request.content_hashes:
        try:
            success, message = await storage.delete(content_hash)

            if success:
                successful += 1
                results.append(BatchMemoryItemResult(success=True, content_hash=content_hash, message=message))

                # Broadcast SSE event
                try:
                    event = create_memory_deleted_event(content_hash, success)
                    await sse_manager.broadcast_event(event)
                except Exception as e:
                    logger.warning(f"Failed to broadcast memory_deleted event: {e}")
            else:
                failed += 1
                results.append(BatchMemoryItemResult(success=False, content_hash=content_hash, message=message))

        except Exception as e:
            failed += 1
            logger.error(f"Error deleting memory {content_hash} in batch: {str(e)}")
            results.append(BatchMemoryItemResult(success=False, content_hash=content_hash, message=f"Error: {str(e)}"))

    return BatchMemoryDeleteResponse(total=len(request.content_hashes), successful=successful, failed=failed, results=results)
