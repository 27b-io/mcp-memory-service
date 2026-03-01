#!/usr/bin/env python3
"""FastAPI MCP Server for Memory Service.

Native MCP protocol implementation using FastMCP with Pydantic-validated
tool inputs.  Each tool handler constructs an input model for validation,
removing all inline range clamping, mode checking, and required-field logic.
"""

import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from fastmcp import Context, FastMCP  # noqa: E402

# Import existing memory service components
from .formatters.toon import format_search_results_as_toon  # noqa: E402
from .models.mcp_inputs import (  # noqa: E402
    ContradictionsParams,
    DeleteMemoryParams,
    FindDuplicatesParams,
    MergeDuplicatesParams,
    RelationParams,
    SearchParams,
    StoreMemoryParams,
    SupersedeParams,
)
from .models.validators import normalize_tags  # noqa: E402
from .resources.toon_documentation import TOON_FORMAT_DOCUMENTATION  # noqa: E402
from .services.memory_service import MemoryService  # noqa: E402
from .storage.base import MemoryStorage  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    return f"# latency_ms={elapsed}\n{response}"


@dataclass
class MCPServerContext:
    """Application context for the MCP server with all required components."""

    storage: MemoryStorage
    memory_service: MemoryService


@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterator[MCPServerContext]:
    """Manage MCP server lifecycle with proper resource initialization and cleanup."""

    # Register optional three-tier tools before accepting requests
    _maybe_register_three_tier_tools()

    # Check if shared storage is already initialized (by unified_server)
    from .shared_storage import get_shared_storage, is_storage_initialized

    if is_storage_initialized():
        logger.debug("Using pre-initialized shared storage instance")
        storage = await get_shared_storage()
    else:
        logger.info("No shared storage found, initializing new instance (standalone mode)")
        from .storage.factory import create_storage_instance

        storage = await create_storage_instance()

    # Initialize memory service with shared business logic
    from .shared_storage import get_graph_client, get_write_queue

    memory_service = MemoryService(
        storage,
        graph_client=get_graph_client(),
        write_queue=get_write_queue(),
    )

    try:
        yield MCPServerContext(storage=storage, memory_service=memory_service)
    finally:
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
    """Return comprehensive TOON format specification for LLM consumption."""
    return TOON_FORMAT_DOCUMENTATION


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
    summary: str | None = None,
) -> dict[str, Any]:
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

    try:
        params = StoreMemoryParams(
            content=content,
            tags=tags,
            memory_type=memory_type,
            metadata=metadata,
            client_hostname=client_hostname,
            summary=summary,
        )
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.store_memory(
        content=params.content,
        tags=params.tags or None,
        memory_type=params.memory_type,
        metadata=params.metadata,
        client_hostname=params.client_hostname,
        summary=params.summary,
    )

    # Transform service response to MCP wire format
    if result["success"]:
        interference = result.get("interference")

        if "memory" in result:
            response: dict[str, Any] = {
                "success": True,
                "message": "Memory stored successfully",
                "content_hash": result["memory"]["content_hash"],
            }
            if interference:
                response["interference"] = interference
            return _inject_latency(response, _t0)
        elif "memories" in result:
            chunk_hashes = [m["content_hash"] for m in result["memories"]]
            response = {
                "success": True,
                "message": f"Memory stored as {result['total_chunks']} chunks",
                "chunks_created": result["total_chunks"],
                "chunk_hashes": chunk_hashes,
            }
            if interference:
                response["interference"] = interference
            return _inject_latency(response, _t0)

    return _inject_latency(
        {"success": False, "error": result.get("error", "Unknown error occurred")},
        _t0,
    )


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
    min_similarity: float = 0.3,
    output: str = "full",
    memory_type: str | None = None,
    encoding_context: dict[str, Any] | None = None,
    include_superseded: bool = False,
    min_trust_score: float | None = None,
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
        min_similarity: Similarity threshold 0.0-1.0 (default: 0.3). Higher=stricter.
        output: "full" (default), "summary" (token-efficient ~50-token summaries), or "both". Applies to scan mode.
        memory_type: Filter by type for "recent" mode (note/decision/task/reference)
        encoding_context: Context-dependent retrieval boost (time_of_day, day_type, agent, task_tags)
        include_superseded: If True, include superseded memories in results (default: False)
        min_trust_score: Filter by minimum provenance trust score (0.0-1.0). Memories without
            provenance are treated as 0.5. Useful for filtering low-reliability sources.

    Returns:
        hybrid/tag/recent: TOON-formatted string (pipe-delimited, with pagination header).
        scan/similar: dict with results list and metadata.
    """
    _t0 = time.perf_counter()

    # Validate all inputs via Pydantic model
    try:
        params = SearchParams(
            query=query,
            mode=mode,
            tags=tags,
            match_all=match_all,
            k=k,
            page=page,
            page_size=page_size,
            min_similarity=min_similarity,
            output=output,
            memory_type=memory_type,
            encoding_context=encoding_context,
            include_superseded=include_superseded,
            min_trust_score=min_trust_score,
        )
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service

    if params.mode == "scan":
        result = await memory_service.scan_memories(
            query=params.query,
            n_results=params.k,
            min_relevance=params.min_similarity,
            output_format=params.output,
        )
        return _inject_latency(result, _t0)

    if params.mode == "similar":
        result = await memory_service.find_similar_memories(query=params.query, k=params.k)
        return _inject_latency(result, _t0)

    if params.mode == "tag":
        if not params.tags:
            return _inject_latency({"success": False, "error": "tags parameter required for tag mode"}, _t0)
        result = await memory_service.search_by_tag(
            tags=params.tags,
            match_all=params.match_all,
            page=params.page,
            page_size=params.page_size,
        )
    elif params.mode == "recent":
        if len(params.tags) > 1:
            logger.warning("recent mode only supports a single tag filter; using first tag '%s'", params.tags[0])
        tag_filter = params.tags[0] if params.tags else None
        result = await memory_service.list_memories(
            page=params.page,
            page_size=params.page_size,
            tag=tag_filter,
            memory_type=params.memory_type,
        )
    else:
        # hybrid search
        result = await memory_service.retrieve_memories(
            query=params.query,
            page=params.page,
            page_size=params.page_size,
            min_similarity=params.min_similarity,
            encoding_context=params.encoding_context,
            tags=params.tags or None,
            include_superseded=params.include_superseded,
            min_trust_score=params.min_trust_score,
        )

    memories = result.get("memories")
    if memories is None:
        return _inject_latency(result, _t0)

    pagination = {
        "page": result.get("page", params.page),
        "total": result.get("total", 0),
        "page_size": result.get("page_size", params.page_size),
        "has_more": result.get("has_more", False),
        "total_pages": result.get("total_pages", 0),
    }
    toon_output, _ = format_search_results_as_toon(memories, pagination=pagination)
    return _inject_latency(toon_output, _t0)


@mcp.tool()
async def delete_memory(content_hash: str, ctx: Context) -> dict[str, bool | str]:
    """Permanently delete a specific memory by its unique identifier.

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

    try:
        params = DeleteMemoryParams(content_hash=content_hash)
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.delete_memory(params.content_hash)
    return _inject_latency(result, _t0)


@mcp.tool()
async def check_database_health(ctx: Context) -> dict[str, Any]:
    """Check memory database health and get storage statistics.

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

    try:
        params = RelationParams(
            action=action,
            content_hash=content_hash,
            target_hash=target_hash,
            relation_type=relation_type,
        )
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service

    if params.action == "create":
        result = await memory_service.create_relation(
            source_hash=params.content_hash,
            target_hash=params.target_hash,
            relation_type=params.relation_type,
        )
    elif params.action == "get":
        result = await memory_service.get_relations(
            content_hash=params.content_hash,
            relation_type=params.relation_type,
        )
    else:  # delete
        result = await memory_service.delete_relation(
            source_hash=params.content_hash,
            target_hash=params.target_hash,
            relation_type=params.relation_type,
        )

    return _inject_latency(result, _t0)


# =============================================================================
# CONTRADICTION RESOLUTION TOOLS
# =============================================================================


@mcp.tool()
async def memory_supersede(
    old_id: str,
    new_id: str,
    ctx: Context,
    reason: str = "",
) -> dict[str, Any]:
    """Mark one memory as superseded by another, resolving a contradiction.

    The old memory is NOT deleted — it is marked as superseded in its metadata
    and excluded from default search results. A SUPERSEDES graph edge is created
    from the new memory to the old for audit trail purposes.

    Use this when you have confirmed that new_id contains more accurate or current
    information than old_id, and you want to declare new_id the authoritative version.

    Args:
        old_id: Content hash of the memory being superseded (the outdated one)
        new_id: Content hash of the newer memory that replaces it
        reason: Human-readable explanation for why old_id is superseded

    Returns:
        {success, superseded, superseded_by, reason} on success, or {success, error} on failure.
    """
    _t0 = time.perf_counter()

    try:
        params = SupersedeParams(old_id=old_id, new_id=new_id, reason=reason)
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.supersede_memory(
        old_hash=params.old_id,
        new_hash=params.new_id,
        reason=params.reason,
    )
    return _inject_latency(result, _t0)


@mcp.tool()
async def memory_contradictions(
    ctx: Context,
    limit: int = 20,
) -> dict[str, Any]:
    """List unresolved contradiction pairs for review and resolution.

    Returns pairs of memories connected by CONTRADICTS edges, with content
    previews to help decide which (if either) to supersede. Call memory_supersede
    to resolve a contradiction once you've determined the authoritative version.

    Args:
        limit: Maximum number of contradiction pairs to return (default: 20)

    Returns:
        {success, pairs: [{memory_a_hash, memory_b_hash, confidence, memory_a_content,
         memory_b_content, memory_a_superseded, memory_b_superseded}], total}
    """
    _t0 = time.perf_counter()

    try:
        params = ContradictionsParams(limit=limit)
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.get_contradictions_dashboard(limit=params.limit)
    return _inject_latency(result, _t0)


@mcp.tool()
async def find_duplicates(
    ctx: Context,
    similarity_threshold: float = 0.95,
    limit: int = 500,
    strategy: str = "keep_newest",
) -> dict[str, Any]:
    """Scan memories for near-duplicates using embedding cosine similarity.

    Loads up to *limit* memories, embeds them in a single batch pass, then
    clusters semantically similar pairs into duplicate groups. Each group
    includes a recommended canonical memory based on the chosen strategy.

    Call merge_duplicates to supersede the non-canonical memories once you
    have reviewed the groups.

    Args:
        similarity_threshold: Cosine similarity threshold (default 0.95).
            Memories above this are considered duplicates. Lower values
            (e.g. 0.90) find more aggressive duplicates; higher values
            (e.g. 0.99) find only near-exact restatements.
        limit: Maximum number of memories to scan (default 500).
            Larger values are more thorough but slower.
        strategy: Canonical selection — which memory to keep:
            "keep_newest" (default), "keep_oldest", "keep_most_accessed"

    Returns:
        {success, groups: [{hashes, canonical_hash, max_similarity, size}],
         total_memories_scanned, total_duplicates_found}
    """
    _t0 = time.perf_counter()

    try:
        params = FindDuplicatesParams(
            similarity_threshold=similarity_threshold,
            limit=limit,
            strategy=strategy,
        )
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.find_duplicates(
        similarity_threshold=params.similarity_threshold,
        limit=params.limit,
        strategy=params.strategy,
    )
    return _inject_latency(result, _t0)


@mcp.tool()
async def merge_duplicates(
    canonical_hash: str,
    duplicate_hashes: list[str],
    ctx: Context,
    reason: str = "Merged by deduplication",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Supersede duplicate memories in favour of a canonical one.

    Each memory in *duplicate_hashes* is marked as superseded by
    *canonical_hash* and excluded from future search results. The duplicates
    are NOT deleted — they remain for historical audit via the graph layer.

    Typical workflow:
        1. Call find_duplicates to identify groups
        2. Review the groups and confirm the canonical_hash
        3. Call merge_duplicates with the group's hashes
        4. Optionally set dry_run=True first to preview without modifying storage

    Args:
        canonical_hash: Content hash of the memory to keep
        duplicate_hashes: Content hashes of the memories to supersede
        reason: Human-readable reason stored in each supersession record
        dry_run: If True, validate inputs and preview result without modifying storage

    Returns:
        {success, canonical_hash, superseded: [hashes], errors: [], dry_run}
    """
    _t0 = time.perf_counter()

    try:
        params = MergeDuplicatesParams(
            canonical_hash=canonical_hash,
            duplicate_hashes=duplicate_hashes,
            reason=reason,
            dry_run=dry_run,
        )
    except ValidationError as e:
        return _inject_latency({"success": False, "error": str(e)}, _t0)

    memory_service = ctx.request_context.lifespan_context.memory_service
    result = await memory_service.merge_duplicate_group(
        canonical_hash=params.canonical_hash,
        duplicate_hashes=params.duplicate_hashes,
        reason=params.reason,
        dry_run=params.dry_run,
    )
    return _inject_latency(result, _t0)


# =============================================================================
# THREE-TIER MEMORY (server-side automation, not exposed as tools by default)
# =============================================================================


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
        tag_list = normalize_tags(tags)
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
        tag_list = normalize_tags(tags)
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


_three_tier_registered = False
_three_tier_lock = threading.Lock()


def _maybe_register_three_tier_tools() -> None:
    """Register three-tier tools if expose_tools is enabled (once only)."""
    global _three_tier_registered
    if _three_tier_registered:
        return
    with _three_tier_lock:
        if _three_tier_registered:
            return
        from .config import settings as _settings

        if _settings.three_tier.expose_tools:
            _register_three_tier_tools(mcp)
        _three_tier_registered = True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the FastAPI MCP server."""
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting MCP Memory Service FastAPI server on {host}:{port}")
    logger.info("Storage backend: Qdrant")

    transport_mode = os.getenv("MCP_TRANSPORT_MODE", "http")

    if transport_mode == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=host, port=port, stateless_http=True)


if __name__ == "__main__":
    main()
