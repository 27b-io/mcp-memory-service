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
- 5 core tools: store_memory, search, delete_memory, check_database_health, relation
- Three-tier memory tools available behind MCP_THREE_TIER_EXPOSE_TOOLS=true
- SSL/HTTPS support with proper certificate handling
"""

import logging
import time
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


def _latency_enabled() -> bool:
    """Check if latency metrics are enabled (lazy, reads config once per call)."""
    from .config import settings

    return settings.debug.latency_metrics


def _inject_latency(response: dict[str, Any] | str, start: float) -> dict[str, Any] | str:
    """Inject latency_ms into a response if metrics are enabled.

    For dict responses: adds 'latency_ms' key.
    For TOON strings: prepends '# latency_ms=N.N' comment line.
    """
    if not _latency_enabled():
        return response
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    if isinstance(response, dict):
        response["latency_ms"] = elapsed
        return response
    # TOON string — prepend latency as comment
    return f"# latency_ms={elapsed}\n{response}"


@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""

    storage: MemoryStorage
    memory_service: MemoryService


@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """Manage MCP server lifecycle with proper resource initialization and cleanup."""
    logger.info("Initializing MCP Memory Service components...")

    # Register optional three-tier tools before accepting requests
    _maybe_register_three_tier_tools()

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


def _normalize_tags(tags: str | list[str] | None) -> list[str]:
    """Normalize tags from string or list format to a clean list of trimmed, non-empty strings."""
    if tags is None:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return [s for item in tags if item is not None and (s := str(item).strip())]


@mcp.tool()
async def store_memory(
    content: str,
    ctx: Context,
    tags: str | list[str] | None = None,
    memory_type: str = "note",
    metadata: dict[str, Any] | None = None,
    client_hostname: str | None = None,
    summary: str | None = None,
) -> StoreMemorySuccess | StoreMemorySplitSuccess | StoreMemoryFailure:
    """Store a new memory for future semantic retrieval.

    Content is vectorized for similarity search. Emotional valence, salience scoring,
    and contradiction detection are computed automatically.

    Args:
        content: Text to store (embedded for semantic search)
        tags: Labels — accepts ["tag1", "tag2"] or "tag1,tag2"
        memory_type: Classification — "note", "decision", "task", or "reference"
        metadata: Structured data to attach. Special key: importance (float 0.0-1.0)
        client_hostname: Source machine identifier
        summary: One-line summary (~50 tokens). Auto-generated if omitted.

    Returns:
        {success, content_hash, message} or {success, chunks_created, chunk_hashes} if auto-split.
        May include interference dict if contradictions detected.
    """
    _t0 = time.perf_counter()
    # C1 fix: normalize tags before passing to service layer
    normalized_tags = _normalize_tags(tags)
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.store_memory(
        content=content,
        tags=normalized_tags or None,
        memory_type=memory_type,
        metadata=metadata,
        client_hostname=client_hostname,
        summary=summary,
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
            return _inject_latency(response, _t0)
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
            return _inject_latency(response, _t0)

    # Failure case
    return _inject_latency(StoreMemoryFailure(success=False, message=result.get("error", "Unknown error occurred")), _t0)


@mcp.tool()
async def search(
    ctx: Context,
    query: str = "",
    mode: str = "hybrid",
    tags: str | list[str] | None = None,
    match_all: bool = False,
    k: int = 10,
    page: int = 1,
    page_size: int = 10,
    min_similarity: float = 0.6,
    output: str = "full",
    memory_type: str | None = None,
    encoding_context: dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    """Search and retrieve memories. Consolidates all retrieval modes into one tool.

    Args:
        query: Natural language search query (required for hybrid/scan/similar modes)
        mode: Search strategy:
            - "hybrid" (default): Semantic + tag-boosted search. Best for most queries.
            - "scan": Like hybrid but returns ~50-token summaries (cheap triage).
            - "similar": Pure k-NN vector search, no tag boosting. For duplicate detection.
            - "tag": Exact tag matching. Requires `tags` param.
            - "recent": Chronological (newest first). Optional tag/memory_type filter.
        tags: Tags to filter by (for "tag" mode, or optional boost hints for "hybrid")
        match_all: For "tag" mode — True=AND, False=OR (default: OR)
        k: Max results for "scan" and "similar" modes (default: 10)
        page: Page number, 1-indexed (default: 1)
        page_size: Results per page (default: 10, max: 100)
        min_similarity: Similarity threshold 0.0-1.0 (default: 0.6). Higher=stricter.
        output: "full" (default), "summary" (token-efficient ~50-token summaries), or "both". Applies to scan mode.
        memory_type: Filter by type for "recent" mode (note/decision/task/reference)
        encoding_context: Context-dependent retrieval boost (time_of_day, day_type, agent, task_tags)

    Returns:
        hybrid/tag/recent: TOON-formatted string (pipe-delimited, with pagination header).
        scan/similar: dict with results list and metadata.
    """
    _t0 = time.perf_counter()

    # C2: validate mode
    _VALID_MODES = {"hybrid", "scan", "similar", "tag", "recent"}
    if mode not in _VALID_MODES:
        return _inject_latency({"error": f"Unknown mode: '{mode}'. Valid: {', '.join(sorted(_VALID_MODES))}"}, _t0)

    # C3: require query for query-dependent modes
    if mode in {"hybrid", "scan", "similar"} and not (query or "").strip():
        return _inject_latency({"error": f"query is required for '{mode}' mode"}, _t0)

    # M3: clamp numeric params to safe ranges
    k = max(1, min(k, 100))
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    memory_service = ctx.request_context.lifespan_context.memory_service

    if mode == "scan":
        # M4: output param applies to scan mode
        output_format = output if output in {"full", "summary", "both"} else "full"
        result = await memory_service.scan_memories(
            query=query,
            n_results=k,
            min_relevance=min_similarity,
            output_format=output_format,
        )
        return _inject_latency(result, _t0)

    if mode == "similar":
        result = await memory_service.find_similar_memories(query=query, k=k)
        return _inject_latency(result, _t0)

    if mode == "tag":
        normalized = _normalize_tags(tags)
        if not normalized:
            return _inject_latency({"error": "tags parameter required for tag mode"}, _t0)
        result = await memory_service.search_by_tag(tags=normalized, match_all=match_all, page=page, page_size=page_size)
    elif mode == "recent":
        normalized = _normalize_tags(tags)
        if len(normalized) > 1:
            logger.warning("recent mode only supports a single tag filter; using first tag '%s'", normalized[0])
        tag_filter = normalized[0] if normalized else None
        result = await memory_service.list_memories(page=page, page_size=page_size, tag=tag_filter, memory_type=memory_type)
    else:
        # hybrid search — M1: pass tags through for boosting
        normalized_tags = _normalize_tags(tags) or None
        result = await memory_service.retrieve_memories(
            query=query,
            page=page,
            page_size=page_size,
            min_similarity=min_similarity,
            encoding_context=encoding_context,
            tags=normalized_tags,
        )

    pagination = {
        "page": result.get("page", page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }
    toon_output, _ = format_search_results_as_toon(result["memories"], pagination=pagination)
    return _inject_latency(toon_output, _t0)


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
    _t0 = time.perf_counter()
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.delete_memory(content_hash)
    return _inject_latency(result, _t0)


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
    _t0 = time.perf_counter()
    # Delegate to shared MemoryService business logic
    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.check_database_health()
    return _inject_latency(result, _t0)


# =============================================================================
# KNOWLEDGE GRAPH RELATIONSHIP OPERATIONS
# =============================================================================


@mcp.tool()
async def relation(
    action: str,
    content_hash: str,
    ctx: Context,
    target_hash: str | None = None,
    relation_type: str | None = None,
) -> dict[str, Any]:
    """Manage typed relationships between memories in the knowledge graph.

    Args:
        action: Operation to perform:
            - "create": Create an edge (requires target_hash and relation_type)
            - "get": List edges for a memory (optional relation_type filter)
            - "delete": Remove an edge (requires target_hash and relation_type)
        content_hash: Content hash of the primary/source memory
        target_hash: Content hash of the related memory (required for create/delete)
        relation_type: Edge type — "RELATES_TO", "PRECEDES", or "CONTRADICTS"
            Required for create/delete. Optional filter for get.

    Returns:
        create: {success, source, target, relation_type}
        get: {relations: [...], content_hash, count}
        delete: {success, source, target, relation_type}
    """
    _t0 = time.perf_counter()

    _VALID_ACTIONS = {"create", "get", "delete"}
    if action not in _VALID_ACTIONS:
        return _inject_latency(
            {"success": False, "error": f"Unknown action: '{action}'. Valid: {', '.join(sorted(_VALID_ACTIONS))}"}, _t0
        )

    memory_service = ctx.request_context.lifespan_context.memory_service

    if action == "create":
        if not target_hash or not relation_type:
            return _inject_latency({"success": False, "error": "target_hash and relation_type required for create"}, _t0)
        result = await memory_service.create_relation(
            source_hash=content_hash, target_hash=target_hash, relation_type=relation_type
        )
    elif action == "get":
        result = await memory_service.get_relations(content_hash=content_hash, relation_type=relation_type)
    elif action == "delete":
        if not target_hash or not relation_type:
            return _inject_latency({"success": False, "error": "target_hash and relation_type required for delete"}, _t0)
        result = await memory_service.delete_relation(
            source_hash=content_hash, target_hash=target_hash, relation_type=relation_type
        )

    return _inject_latency(result, _t0)


# =============================================================================
# THREE-TIER MEMORY (server-side automation, not exposed as tools by default)
# =============================================================================
# The three-tier memory model (sensory → working → long-term) runs server-side.
# To expose these as MCP tools for autonomous agent rigs, set:
#   MCP_THREE_TIER_EXPOSE_TOOLS=true


def _register_three_tier_tools(mcp_instance: FastMCP) -> None:
    """Conditionally register three-tier memory tools behind feature flag."""

    @mcp_instance.tool()
    async def push_to_sensory_buffer(
        content: str,
        ctx: Context,
        tags: str | list[str] | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Push raw input to the sensory buffer (~7 items, 1s TTL ring buffer)."""
        memory_service = ctx.request_context.lifespan_context.memory_service
        three_tier = memory_service.three_tier
        if three_tier is None:
            return {"success": False, "error": "Three-tier memory is disabled"}
        tag_list = _normalize_tags(tags)
        three_tier.push_sensory(content, metadata=metadata, tags=tag_list, memory_type=memory_type)
        return {"success": True, "message": "Item pushed to sensory buffer", "buffer": three_tier.sensory.stats()}

    @mcp_instance.tool()
    async def activate_working_memory(
        key: str,
        content: str,
        ctx: Context,
        tags: str | list[str] | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bring a memory into active working memory (~4 chunks, Cowan's limit)."""
        memory_service = ctx.request_context.lifespan_context.memory_service
        three_tier = memory_service.three_tier
        if three_tier is None:
            return {"success": False, "error": "Three-tier memory is disabled"}
        tag_list = _normalize_tags(tags)
        chunk = three_tier.attend(key=key, content=content, metadata=metadata, tags=tag_list, memory_type=memory_type)
        return {"success": True, "key": key, "access_count": chunk.access_count, "working_memory": three_tier.working.stats()}

    @mcp_instance.tool()
    async def flush_sensory_to_working(ctx: Context) -> dict[str, Any]:
        """Promote valid sensory buffer items to working memory."""
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

    @mcp_instance.tool()
    async def consolidate_working_memory(ctx: Context) -> dict[str, Any]:
        """Consolidate working memory items accessed 2+ times to long-term storage."""
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

    @mcp_instance.tool()
    async def get_working_memory_status(ctx: Context) -> dict[str, Any]:
        """Get sensory buffer and working memory tier statistics."""
        memory_service = ctx.request_context.lifespan_context.memory_service
        three_tier = memory_service.three_tier
        if three_tier is None:
            return {"enabled": False, "message": "Three-tier memory is disabled"}
        return {"enabled": True, "tiers": three_tier.stats()}


def _maybe_register_three_tier_tools() -> None:
    """Register three-tier tools if expose_tools is enabled.

    Deferred to startup (not module-level) to preserve lazy _SettingsProxy behavior.
    """
    from .config import settings as _settings

    if _settings.three_tier.expose_tools:
        _register_three_tier_tools(mcp)


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
