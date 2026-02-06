#!/usr/bin/env python3
"""
FastAPI MCP Server for Memory Service

This module implements a native MCP server using the FastAPI MCP framework,
replacing the Node.js HTTP-to-MCP bridge to resolve SSL connectivity issues
and provide direct MCP protocol support.

Features:
- Native MCP protocol implementation using FastMCP
- Direct integration with existing memory storage backends
- Streamable HTTP transport for remote access
- All 22 core memory operations (excluding dashboard tools)
- SSL/HTTPS support with proper certificate handling
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10
import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from fastmcp import Context, FastMCP  # noqa: E402

# Import existing memory service components
from .formatters.toon import format_search_results_as_toon  # noqa: E402
from .resources.toon_documentation import TOON_FORMAT_DOCUMENTATION  # noqa: E402
from .services.memory_service import MemoryService  # noqa: E402
from .storage.base import MemoryStorage  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)  # Default to INFO level
logger = logging.getLogger(__name__)


@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""

    storage: MemoryStorage
    memory_service: MemoryService


@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """Manage MCP server lifecycle with proper resource initialization and cleanup."""
    logger.info("Initializing MCP Memory Service components...")

    # Check if shared storage is already initialized (by unified_server)
    from .shared_storage import get_shared_storage, is_storage_initialized

    if is_storage_initialized():
        logger.info("Using pre-initialized shared storage instance")
        storage = await get_shared_storage()
    else:
        # Fallback to creating storage if running standalone
        logger.info("No shared storage found, initializing new instance (standalone mode)")
        from .storage.factory import create_storage_instance

        storage = await create_storage_instance()

    # Initialize memory service with shared business logic (including graph layer if available)
    from .shared_storage import get_graph_client, get_write_queue

    memory_service = MemoryService(
        storage,
        graph_client=get_graph_client(),
        write_queue=get_write_queue(),
    )

    try:
        yield MCPServerContext(storage=storage, memory_service=memory_service)
    finally:
        # Only close storage if we created it (standalone mode)
        # Shared storage is managed by unified_server
        if not is_storage_initialized():
            logger.info("Shutting down MCP Memory Service components...")
            if hasattr(storage, "close"):
                await storage.close()


# Create FastMCP server instance
mcp = FastMCP("MCP Memory Service", lifespan=mcp_server_lifespan)


# =============================================================================
# RESOURCES
# =============================================================================


@mcp.resource("toon://format/documentation")
def toon_format_docs() -> str:
    """
    Return comprehensive TOON format specification for LLM consumption.

    This resource provides the complete TOON (Terser Object Notation) format
    specification, including structure, field types, parsing strategies, and examples.
    Used by LLMs to understand the compact pipe-delimited format returned by memory tools.
    """
    return TOON_FORMAT_DOCUMENTATION


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class StoreMemorySuccess(TypedDict):
    """Return type for successful single memory storage."""

    success: bool
    message: str
    content_hash: str
    interference: NotRequired[dict[str, Any]]


class StoreMemorySplitSuccess(TypedDict):
    """Return type for successful chunked memory storage."""

    success: bool
    message: str
    chunks_created: int
    chunk_hashes: list[str]
    interference: NotRequired[dict[str, Any]]


class StoreMemoryFailure(TypedDict):
    """Return type for failed memory storage."""

    success: bool
    message: str
    chunks_created: NotRequired[int]
    chunk_hashes: NotRequired[list[str]]


# =============================================================================
# CORE MEMORY OPERATIONS
# =============================================================================


@mcp.tool()
async def store_memory(
    content: str,
    ctx: Context,
    tags: str | list[str] | None = None,
    memory_type: str = "note",
    metadata: dict[str, Any] | None = None,
    client_hostname: str | None = None,
) -> StoreMemorySuccess | StoreMemorySplitSuccess | StoreMemoryFailure:
    """
    Store a new memory for future semantic retrieval.

    Persists content with optional categorization (tags, type, metadata) for later
    retrieval via semantic search or tag filtering. Content is automatically vectorized
    for similarity matching.

    Emotional tagging and salience scoring are computed automatically:
    - Emotional valence (sentiment, magnitude, category) is detected from content
    - Salience score combines emotional intensity, access frequency, and explicit importance
    - Higher-salience memories receive a retrieval boost

    Proactive interference detection:
    - New memories are checked against existing ones for contradictions
    - Contradictions are flagged (not blocked) with signal type and confidence
    - CONTRADICTS edges are created in the knowledge graph for flagged pairs
    - Signal types: negation, antonym, temporal (supersession)

    Args:
        content: The text content to store (will be embedded for semantic search)
        tags: Categorization labels (accepts ["tag1", "tag2"] or "tag1,tag2")
        memory_type: Classification - "note", "decision", "task", or "reference"
        metadata: Additional structured data to attach. Special keys:
            - importance: float (0.0-1.0) - explicit importance weight for salience scoring
        client_hostname: Source machine identifier (optional)

    Content Length Handling:
        - No limit on content length
        - Auto-splitting preserves context with 50-char overlap
        - Respects natural boundaries: paragraphs → sentences → words

    Tag Formats (both supported):
        - Array: tags=["python", "bug-fix", "urgent"]
        - String: tags="python,bug-fix,urgent"

    Returns:
        Single memory:
            - success: True/False
            - message: Status description
            - content_hash: Unique identifier for retrieval/deletion
            - interference: (optional) Contradiction detection results if conflicts found

        Split memory (when auto-split enabled):
            - success: True/False
            - message: Status description
            - chunks_created: Number of linked chunks
            - chunk_hashes: List of content hashes
            - interference: (optional) Contradiction detection results if conflicts found

    Use this for: Capturing information for later retrieval, building knowledge base,
    recording decisions, storing context across conversations.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.store_memory(
        content=content,
        tags=tags,
        memory_type=memory_type,
        metadata=metadata,
        client_hostname=client_hostname,
    )

    # Transform MemoryService response to MCP schema
    if result["success"]:
        interference = result.get("interference")

        if "memory" in result:
            # Single memory case
            response = StoreMemorySuccess(
                success=True,
                message="Memory stored successfully",
                content_hash=result["memory"]["content_hash"],
            )
            if interference:
                response["interference"] = interference
            return response
        elif "memories" in result:
            # Chunked memory case
            chunk_hashes = [m["content_hash"] for m in result["memories"]]
            response = StoreMemorySplitSuccess(
                success=True,
                message=f"Memory stored as {result['total_chunks']} chunks",
                chunks_created=result["total_chunks"],
                chunk_hashes=chunk_hashes,
            )
            if interference:
                response["interference"] = interference
            return response

    # Failure case
    return StoreMemoryFailure(success=False, message=result.get("error", "Unknown error occurred"))


@mcp.tool()
async def retrieve_memory(
    query: str,
    ctx: Context,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float = 0.6,
    encoding_context: dict[str, Any] | None = None,
) -> str:
    """
    Retrieve memories using hybrid search (semantic + tag matching).

    Combines vector similarity with automatic tag extraction for improved retrieval.
    When query terms match existing tags, those memories receive a score boost.
    This solves the "rathole problem" where project-specific queries return
    semantically similar but categorically unrelated results.

    Hybrid search is enabled by default. To opt-out to pure vector search:
    - Set environment variable MCP_MEMORY_HYBRID_ALPHA=1.0

    Args:
        query: Natural language search query (tags extracted automatically)
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        min_similarity: Quality threshold (0.0-1.0, default: 0.6)
            - 0.6-0.7: Good matches (recommended)
            - 0.7-0.9: Very similar matches
            - 0.9+: Nearly identical
            - Lower for exploratory search, higher for precision
        encoding_context: Optional current context for context-dependent retrieval.
            Memories stored in a similar context receive a score boost.
            Keys: time_of_day (morning|afternoon|evening|night),
            day_type (weekday|weekend), agent (hostname/agent name),
            task_tags (list of current task/project tags)

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash|similarity_score

        For complete TOON specification, see resource: toon://format/documentation

    Use this for: Finding relevant context, answering questions, discovering related information.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.retrieve_memories(
        query=query,
        page=page,
        page_size=page_size,
        min_similarity=min_similarity,
        encoding_context=encoding_context,
    )

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


@mcp.tool()
async def search_by_tag(tags: str | list[str], ctx: Context, match_all: bool = False, page: int = 1, page_size: int = 10) -> str:
    """
    Search memories by exact tag matches with flexible filtering.

    Finds memories tagged with specific labels. Use for categorical retrieval
    when you know the exact tags (e.g., "python", "bug-fix", "customer-support").

    Args:
        tags: Single tag string or list of tags to search for
        match_all: Matching mode (default: False = ANY)
            - False (ANY): Returns memories with at least one matching tag
              Use for: Broad category search, exploration
            - True (ALL): Returns only memories with every specified tag
              Use for: Precise filtering, intersection of categories
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash

        For complete TOON specification, see resource: toon://format/documentation

    Examples:
        - ["python", "api"] with match_all=False → memories tagged python OR api
        - ["python", "api"] with match_all=True → memories tagged python AND api
        - "bug-fix" → all memories tagged bug-fix

    Use this for: Tag-based filtering, categorical search, known classification retrieval.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.search_by_tag(tags=tags, match_all=match_all, page=page, page_size=page_size)

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


@mcp.tool()
async def delete_memory(content_hash: str, ctx: Context) -> dict[str, bool | str]:
    """
    Permanently delete a specific memory by its unique identifier.

    Removes a memory from the database. This operation is irreversible.
    The content_hash is returned when storing memories or can be found in
    search/retrieve results.

    Args:
        content_hash: Unique identifier returned from store_memory or found in search results

    Returns:
        Dictionary with:
        - success: True if deleted, False if not found or error
        - message: Confirmation or error description

    Use this for: Removing outdated information, cleaning up test data, deleting
    sensitive content, managing storage space.

    Warning: Deletion is permanent. Verify the content_hash before deleting.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.delete_memory(content_hash)


@mcp.tool()
async def check_database_health(ctx: Context) -> dict[str, Any]:
    """
    Check memory database health and get storage statistics.

    Verifies database connectivity and returns operational metrics including
    total memory count, storage backend type, and system status.

    Returns:
        Dictionary with:
        - status: "healthy" or error state
        - backend: Storage backend in use (qdrant)
        - total_memories: Total count of stored memories
        - storage_info: Backend-specific statistics
        - version: Service version

    Use this for: Debugging connection issues, monitoring storage usage,
    verifying service status, troubleshooting performance problems.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.check_database_health()


@mcp.tool()
async def list_memories(
    ctx: Context,
    page: int = 1,
    page_size: int = 10,
    tag: str | None = None,
    memory_type: str | None = None,
) -> str:
    """
    List memories in chronological order with optional filtering.

    Returns memories ordered by creation time (newest first), without semantic
    ranking. Use this for browsing recent memories or getting a chronological view.

    Args:
        page: Page number (1-indexed, default: 1)
        page_size: Results per page (default: 10, max: 100)
        tag: Optional - return only memories with this specific tag
        memory_type: Optional - filter by type (note, decision, task, reference)

    Response Format:
        Returns memories in TOON (Terser Object Notation) format - a compact, pipe-delimited
        format optimized for LLM token efficiency.

        First line contains pagination metadata:
        # page=2 total=250 page_size=10 has_more=true total_pages=25

        Followed by memory records, each on a single line with fields:
        content|tags|metadata|created_at|updated_at|content_hash

        For complete TOON specification, see resource: toon://format/documentation

    Differences from other search tools:
        - retrieve_memory: Semantic similarity ranking (finds meaning)
        - search_by_tag: Exact tag matching with AND/OR logic
        - list_memories: Chronological order (this tool)

    Use this for: Browsing recent activity, reviewing what was stored,
    chronological exploration, getting latest entries.
    """
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.list_memories(page=page, page_size=page_size, tag=tag, memory_type=memory_type)

    # Extract pagination metadata
    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }

    # Convert results to TOON format with pagination
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return toon_output


# =============================================================================
# KNOWLEDGE GRAPH RELATIONSHIP OPERATIONS
# =============================================================================


@mcp.tool()
async def create_relation(
    source_hash: str,
    target_hash: str,
    relation_type: str,
    ctx: Context,
) -> dict[str, Any]:
    """
    Create a typed relationship between two memories in the knowledge graph.

    Typed edges represent explicit semantic relationships between memories,
    unlike Hebbian edges which form implicitly through co-retrieval patterns.
    These are lower-frequency writes typically created during memory storage
    or when a user identifies a connection between memories.

    Args:
        source_hash: Content hash of the source memory
        target_hash: Content hash of the target memory
        relation_type: Type of relationship. Must be one of:
            - "RELATES_TO": Generic semantic relationship
            - "PRECEDES": Temporal/causal ordering (source happened before target)
            - "CONTRADICTS": Conflicting information between memories

    Returns:
        Dictionary with:
        - success: True if relation was created
        - source: Source memory hash
        - target: Target memory hash
        - relation_type: Normalized relation type
        - error: Error message if failed

    Use this for: Building knowledge graphs, linking related memories,
    tracking temporal sequences, flagging contradictions.
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.create_relation(
        source_hash=source_hash,
        target_hash=target_hash,
        relation_type=relation_type,
    )


@mcp.tool()
async def get_relations(
    content_hash: str,
    ctx: Context,
    relation_type: str | None = None,
) -> dict[str, Any]:
    """
    Get typed relationships for a specific memory.

    Returns all typed edges (RELATES_TO, PRECEDES, CONTRADICTS) connected
    to the given memory, in both directions.

    Args:
        content_hash: Content hash of the memory to query
        relation_type: Optional filter - only return edges of this type

    Returns:
        Dictionary with:
        - relations: List of relationship edges, each containing:
            - source: Source memory hash
            - target: Target memory hash
            - relation_type: Edge type (RELATES_TO, PRECEDES, CONTRADICTS)
            - direction: "outgoing" or "incoming" relative to queried memory
            - created_at: Timestamp when relation was created
        - content_hash: The queried memory hash
        - count: Number of relations found

    Use this for: Exploring knowledge graph connections, finding related memories,
    understanding temporal sequences, identifying contradictions.
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.get_relations(
        content_hash=content_hash,
        relation_type=relation_type,
    )


@mcp.tool()
async def delete_relation(
    source_hash: str,
    target_hash: str,
    relation_type: str,
    ctx: Context,
) -> dict[str, Any]:
    """
    Delete a typed relationship between two memories.

    Removes a specific directed edge from the knowledge graph. This only
    removes the relationship — the memories themselves are not affected.

    Args:
        source_hash: Content hash of the source memory
        target_hash: Content hash of the target memory
        relation_type: Type of relationship to delete (RELATES_TO, PRECEDES, CONTRADICTS)

    Returns:
        Dictionary with:
        - success: True if relation was deleted, False if not found
        - source: Source memory hash
        - target: Target memory hash
        - relation_type: The relation type that was deleted

    Use this for: Removing incorrect relationships, cleaning up knowledge graph.
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    return await memory_service.delete_relation(
        source_hash=source_hash,
        target_hash=target_hash,
        relation_type=relation_type,
    )


# =============================================================================
# THREE-TIER MEMORY OPERATIONS
# =============================================================================


@mcp.tool()
async def push_to_sensory_buffer(
    content: str,
    ctx: Context,
    tags: str | list[str] | None = None,
    memory_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Push raw input to the sensory buffer for short-term capture.

    The sensory buffer is a ring buffer (~7 items, 1s TTL) that captures raw
    input before the system decides what's important. Items expire quickly and
    are not persisted. Use `attend_from_sensory` to promote items to working
    memory, or `flush_sensory_to_working` to promote all valid items.

    Part of Cowan's three-tier memory model:
    Sensory Buffer → Working Memory → Long-Term Memory

    Args:
        content: The raw content to buffer
        tags: Optional tags (accepts ["tag1", "tag2"] or "tag1,tag2")
        memory_type: Optional type classification
        metadata: Optional additional data

    Returns:
        Dictionary with buffer status and item info.
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    three_tier = memory_service.three_tier
    if three_tier is None:
        return {"success": False, "error": "Three-tier memory is disabled"}

    tag_list = _normalize_tags(tags)
    three_tier.push_sensory(content, metadata=metadata, tags=tag_list, memory_type=memory_type)
    return {
        "success": True,
        "message": "Item pushed to sensory buffer",
        "buffer": three_tier.sensory.stats(),
    }


@mcp.tool()
async def activate_working_memory(
    key: str,
    content: str,
    ctx: Context,
    tags: str | list[str] | None = None,
    memory_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Bring a memory into active working memory (attention gate).

    Working memory holds ~4 chunks for the current task context (Cowan's limit).
    Items accessed multiple times are automatically consolidated to long-term memory.
    Least-recently-used items are evicted when capacity is reached.

    This is the attention transition: items you reference are promoted to active context.

    Args:
        key: Unique identifier for this chunk (e.g., content hash or topic name)
        content: The content to hold in working memory
        tags: Optional tags
        memory_type: Optional type classification
        metadata: Optional additional data

    Returns:
        Dictionary with chunk status (access count, whether consolidated).
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    three_tier = memory_service.three_tier
    if three_tier is None:
        return {"success": False, "error": "Three-tier memory is disabled"}

    tag_list = _normalize_tags(tags)
    chunk = three_tier.attend(key=key, content=content, metadata=metadata, tags=tag_list, memory_type=memory_type)
    return {
        "success": True,
        "key": key,
        "access_count": chunk.access_count,
        "working_memory": three_tier.working.stats(),
    }


@mcp.tool()
async def flush_sensory_to_working(ctx: Context) -> dict[str, Any]:
    """
    Promote all valid sensory buffer items to working memory.

    Flushes non-expired items from the sensory buffer and activates them
    in working memory. Items that were already in working memory get their
    access count incremented (strengthening their consolidation candidacy).

    Returns:
        Dictionary with number of items promoted and tier stats.
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    three_tier = memory_service.three_tier
    if three_tier is None:
        return {"success": False, "error": "Three-tier memory is disabled"}

    activated = three_tier.flush_sensory_to_working()
    return {
        "success": True,
        "items_promoted": len(activated),
        "promoted_keys": [key[:12] + "..." for key, _ in activated],
        "tiers": three_tier.stats(),
    }


@mcp.tool()
async def consolidate_working_memory(ctx: Context) -> dict[str, Any]:
    """
    Consolidate eligible working memory items to long-term storage.

    Items accessed 2+ times in working memory are promoted to permanent
    Qdrant storage via the standard store_memory pipeline. This mimics
    the cognitive process of rehearsal leading to long-term encoding.

    Already-consolidated items are skipped (idempotent within a session).

    Returns:
        Dictionary with consolidation results (items stored, any errors).
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    three_tier = memory_service.three_tier
    if three_tier is None:
        return {"success": False, "error": "Three-tier memory is disabled"}

    results = await three_tier.consolidate()
    return {
        "success": True,
        "items_consolidated": len(results),
        "results": results,
        "working_memory": three_tier.working.stats(),
    }


@mcp.tool()
async def get_working_memory_status(ctx: Context) -> dict[str, Any]:
    """
    Get current status of sensory buffer and working memory.

    Returns capacity, occupancy, chunk access counts, and consolidation status
    for both ephemeral memory tiers. Useful for debugging and monitoring.

    Returns:
        Dictionary with tier statistics (sensory buffer + working memory).
    """
    memory_service = ctx.request_context.lifespan_context.memory_service
    three_tier = memory_service.three_tier
    if three_tier is None:
        return {"enabled": False, "message": "Three-tier memory is disabled"}

    return {"enabled": True, "tiers": three_tier.stats()}


def _normalize_tags(tags: str | list[str] | None) -> list[str]:
    """Normalize tags from string or list format to list."""
    if tags is None:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return tags


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the FastAPI MCP server."""
    # Configure for Claude Code integration
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting MCP Memory Service FastAPI server on {host}:{port}")
    logger.info("Storage backend: Qdrant")

    # Check transport mode from environment
    transport_mode = os.getenv("MCP_TRANSPORT_MODE", "http")

    if transport_mode == "stdio":
        # Run server with stdio transport
        mcp.run(transport="stdio")
    else:
        # Run server with HTTP transport (FastMCP v2.0 uses 'http' instead of 'streamable-http')
        mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
