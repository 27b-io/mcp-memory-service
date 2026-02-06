"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation and ensuring consistent behavior.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, TypedDict

from ..config import (
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT,
    settings,
)
from ..graph.client import GraphClient
from ..graph.queue import HebbianWriteQueue
from ..models.memory import Memory
from ..storage.base import MemoryStorage
from ..utils.content_splitter import split_content
from ..utils.emotional_analysis import analyze_emotion
from ..utils.hashing import generate_content_hash
from ..utils.hybrid_search import (
    apply_recency_decay,
    combine_results_rrf,
    extract_query_keywords,
    get_adaptive_alpha,
)
from ..utils.interference import ContradictionSignal, InterferenceResult, detect_contradiction_signals
from ..utils.salience import SalienceFactors, apply_salience_boost, compute_salience
from ..utils.spaced_repetition import apply_spacing_boost, compute_spacing_quality

logger = logging.getLogger(__name__)


class MemoryResult(TypedDict):
    """Type definition for memory operation results."""

    content: str
    content_hash: str
    tags: list[str]
    memory_type: str | None
    metadata: dict[str, Any] | None
    created_at: str
    updated_at: str
    created_at_iso: str
    updated_at_iso: str
    emotional_valence: dict[str, Any] | None
    salience_score: float


class MemoryService:
    """
    Shared service for memory operations with consistent business logic.

    This service centralizes all memory-related business logic to ensure
    consistent behavior across API endpoints and MCP tools, eliminating
    code duplication and potential inconsistencies.
    """

    # Tag cache TTL in seconds
    _TAG_CACHE_TTL = 60

    def __init__(
        self,
        storage: MemoryStorage,
        graph_client: GraphClient | None = None,
        write_queue: HebbianWriteQueue | None = None,
    ):
        self.storage = storage
        self._graph = graph_client
        self._write_queue = write_queue
        self._tag_cache: tuple[float, set[str]] | None = None

    async def _compute_graph_boosts(self, seed_hashes: list[str]) -> dict[str, float]:
        """
        Run spreading activation from seed memories to find graph-boosted neighbors.

        Non-blocking, non-fatal — graph failures don't affect the read path.

        Returns:
            Dict of {content_hash: activation_score} for activated neighbors.
        """
        if self._graph is None or not seed_hashes:
            return {}

        config = settings.falkordb
        if config.spreading_activation_boost <= 0:
            return {}

        try:
            return await self._graph.spreading_activation(
                seed_hashes=seed_hashes,
                max_hops=config.spreading_activation_max_hops,
                decay_factor=config.spreading_activation_decay,
                min_activation=config.spreading_activation_min_activation,
            )
        except Exception as e:
            logger.warning(f"Spreading activation failed (non-fatal): {e}")
            return {}

    async def _fire_hebbian_co_access(
        self, content_hashes: list[str], spacing_qualities: list[float] | None = None
    ) -> None:
        """
        Enqueue Hebbian strengthening for all pairs of co-retrieved memories.

        Called after a retrieval returns multiple results. Non-blocking,
        non-fatal — write failures don't affect the read path.

        Args:
            content_hashes: Hashes of co-retrieved memories
            spacing_qualities: Per-memory spacing quality scores for LTP modulation.
                If provided, the mean spacing of each pair is used.
        """
        if self._write_queue is None or len(content_hashes) < 2:
            return

        try:
            sq = spacing_qualities or [0.0] * len(content_hashes)
            for i, src in enumerate(content_hashes):
                for j, dst in enumerate(content_hashes[i + 1 :], start=i + 1):
                    # Use mean spacing quality of the pair
                    pair_spacing = (sq[i] + sq[j]) / 2.0
                    await self._write_queue.enqueue_strengthen(src, dst, pair_spacing)
                    await self._write_queue.enqueue_strengthen(dst, src, pair_spacing)
        except Exception as e:
            logger.warning(f"Hebbian co-access enqueue failed (non-fatal): {e}")

    def _compute_emotional_valence(self, content: str) -> dict[str, Any] | None:
        """
        Analyze emotional content and return valence dict if salience is enabled.

        Returns None if salience feature is disabled, letting the memory stay neutral.
        """
        if not settings.salience.enabled:
            return None
        valence = analyze_emotion(content)
        return valence.to_dict()

    def _compute_salience_score(
        self,
        emotional_magnitude: float = 0.0,
        access_count: int = 0,
        explicit_importance: float = 0.0,
    ) -> float:
        """Compute salience score using configured weights."""
        if not settings.salience.enabled:
            return 0.0
        config = settings.salience
        return compute_salience(
            SalienceFactors(
                emotional_magnitude=emotional_magnitude,
                access_count=access_count,
                explicit_importance=explicit_importance,
            ),
            emotional_weight=config.emotional_weight,
            frequency_weight=config.frequency_weight,
            importance_weight=config.importance_weight,
        )

    async def _detect_contradictions(self, content: str) -> InterferenceResult:
        """
        Check for contradictions between new content and existing memories.

        Searches for semantically similar memories, then applies lexical
        contradiction signals to determine if they conflict.

        Non-blocking to store: contradictions are flagged, not prevented.

        Args:
            content: The new memory content to check

        Returns:
            InterferenceResult with any detected contradictions
        """
        config = settings.interference
        result = InterferenceResult()

        if not config.enabled:
            return result

        try:
            # Find semantically similar existing memories
            similar = await self.storage.retrieve(
                query=content,
                n_results=config.max_candidates,
                min_similarity=config.similarity_threshold,
            )

            for match in similar:
                # Skip exact duplicates (same content hash)
                if match.memory.content_hash == generate_content_hash(content):
                    continue

                signals = detect_contradiction_signals(
                    new_content=content,
                    existing_content=match.memory.content,
                    existing_hash=match.memory.content_hash,
                    similarity=match.similarity_score,
                    min_confidence=config.min_confidence,
                )
                result.contradictions.extend(signals)

        except Exception as e:
            logger.warning(f"Contradiction detection failed (non-fatal): {e}")

        return result

    async def _create_contradiction_edges(
        self,
        new_hash: str,
        contradictions: list[ContradictionSignal],
    ) -> None:
        """
        Create CONTRADICTS edges in the graph for detected contradictions.

        Non-blocking, non-fatal — graph failures don't affect storage.
        """
        if self._graph is None or not contradictions:
            return

        # Deduplicate by existing_hash (keep highest confidence per pair)
        seen: dict[str, ContradictionSignal] = {}
        for c in contradictions:
            if c.existing_hash not in seen or c.confidence > seen[c.existing_hash].confidence:
                seen[c.existing_hash] = c

        for existing_hash, signal in seen.items():
            try:
                await self._graph.create_typed_edge(
                    source_hash=new_hash,
                    target_hash=existing_hash,
                    relation_type="CONTRADICTS",
                )
                logger.info(
                    f"Contradiction edge: {new_hash[:8]} -> {existing_hash[:8]} "
                    f"({signal.signal_type}, confidence={signal.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(f"Failed to create contradiction edge (non-fatal): {e}")

    def _fire_access_count_updates(self, content_hashes: list[str]) -> None:
        """
        Schedule access count increments as background tasks (fire-and-forget).

        Non-blocking, non-fatal — access tracking failures don't affect reads.
        """
        if not settings.salience.enabled or not content_hashes:
            return

        async def _do_increments():
            for h in content_hashes:
                try:
                    await self.storage.increment_access_count(h)
                except Exception as e:
                    logger.debug(f"Access count increment failed for {h[:8]}: {e}")

        try:
            asyncio.ensure_future(_do_increments())
        except RuntimeError:
            # No running event loop — skip silently
            pass

    async def _get_cached_tags(self) -> set[str]:
        """Get all tags with 60-second TTL caching for performance."""
        now = time.time()
        if self._tag_cache is not None:
            cache_time, cached_tags = self._tag_cache
            if now - cache_time < self._TAG_CACHE_TTL:
                return cached_tags

        # Cache miss - fetch from storage
        all_tags = await self.storage.get_all_tags()
        self._tag_cache = (now, set(all_tags))
        return self._tag_cache[1]

    async def _retrieve_vector_only(
        self,
        query: str,
        page: int,
        page_size: int,
        tags: list[str] | None,
        memory_type: str | None,
        min_similarity: float | None,
    ) -> dict[str, Any]:
        """Fallback to pure vector search (original behavior)."""
        offset = (page - 1) * page_size

        # Count matching memories for pagination
        try:
            total = await self.storage.count_semantic_search(
                query=query, tags=tags, memory_type=memory_type, min_similarity=min_similarity
            )
        except Exception:
            # Fallback: estimate based on page_size if count fails
            total = page_size * 10  # Reasonable estimate

        memories = await self.storage.retrieve(
            query=query,
            n_results=page_size,
            tags=tags,
            memory_type=memory_type,
            min_similarity=min_similarity,
            offset=offset,
        )

        # Extract hashes and scores for graph boosting
        results = []
        result_hashes = []
        result_memories = []  # Parallel list of Memory objects for spacing boost
        for item in memories:
            if hasattr(item, "memory"):
                memory_dict = self._format_memory_response(item.memory)
                memory_dict["similarity_score"] = item.similarity_score
                results.append(memory_dict)
                result_hashes.append(item.memory.content_hash)
                result_memories.append(item.memory)
            else:
                results.append(self._format_memory_response(item))
                result_hashes.append(item.content_hash)
                result_memories.append(item)

        # Apply spreading activation boost from graph layer
        graph_boosts = await self._compute_graph_boosts(result_hashes)
        if graph_boosts:
            boost_weight = settings.falkordb.spreading_activation_boost
            for result in results:
                activation = graph_boosts.get(result["content_hash"], 0.0)
                if activation > 0:
                    result["similarity_score"] = result.get("similarity_score", 0.0) + boost_weight * activation
                    result["graph_boost"] = activation
            results.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

        # Apply salience boost to final scores
        if settings.salience.enabled:
            for result in results:
                salience = result.get("salience_score", 0.0)
                if salience > 0:
                    result["similarity_score"] = apply_salience_boost(
                        result.get("similarity_score", 0.0),
                        salience,
                        settings.salience.boost_weight,
                    )
            results.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

        # Apply spaced repetition boost and collect spacing qualities for LTP
        spacing_qualities = []
        if settings.spaced_repetition.enabled:
            for result, mem in zip(results, result_memories):
                spacing = compute_spacing_quality(mem.access_timestamps)
                spacing_qualities.append(spacing)
                if spacing > 0:
                    result["similarity_score"] = apply_spacing_boost(
                        result.get("similarity_score", 0.0),
                        spacing,
                        settings.spaced_repetition.boost_weight,
                    )
                    result["spacing_quality"] = spacing
            results.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

        # Fire Hebbian co-access events for co-retrieved memories
        await self._fire_hebbian_co_access(result_hashes, spacing_qualities or None)

        # Track access counts (fire-and-forget)
        self._fire_access_count_updates(result_hashes)

        return {
            "memories": results,
            "query": query,
            "hybrid_enabled": False,
            **self._build_pagination_metadata(total, page, page_size),
        }

    def _build_pagination_metadata(self, total: int, page: int, page_size: int) -> dict[str, Any]:
        """
        Build consistent pagination metadata for all endpoints.

        DRY principle: Single source of truth for pagination structure.

        Args:
            total: Total number of matching records across all pages
            page: Current page number (1-indexed)
            page_size: Number of results per page

        Returns:
            Dictionary with pagination metadata
        """
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (page * page_size) < total,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 1,
        }

    async def list_memories(
        self, page: int = 1, page_size: int = 10, tag: str | None = None, memory_type: str | None = None
    ) -> dict[str, Any]:
        """
        List memories with pagination and optional filtering.

        This method provides database-level filtering for optimal performance,
        avoiding the common anti-pattern of loading all records into memory.

        Args:
            page: Page number (1-based)
            page_size: Number of memories per page
            tag: Filter by specific tag
            memory_type: Filter by memory type

        Returns:
            Dictionary with memories and pagination info
        """
        try:
            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Use database-level filtering for optimal performance
            tags_list = [tag] if tag else None
            memories = await self.storage.get_all_memories(
                limit=page_size, offset=offset, memory_type=memory_type, tags=tags_list
            )

            # Get accurate total count for pagination
            total = await self.storage.count_all_memories(memory_type=memory_type, tags=tags_list)

            # Format results for API response
            results = []
            for memory in memories:
                results.append(self._format_memory_response(memory))

            return {"memories": results, **self._build_pagination_metadata(total, page, page_size)}

        except Exception as e:
            logger.exception(f"Unexpected error listing memories: {e}")
            return {
                "success": False,
                "error": f"Failed to list memories: {str(e)}",
                "memories": [],
                "page": page,
                "page_size": page_size,
            }

    async def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        client_hostname: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a new memory with validation and content processing.

        Args:
            content: The memory content
            tags: Optional tags for the memory
            memory_type: Optional memory type classification
            metadata: Optional additional metadata
            client_hostname: Optional client hostname for source tagging

        Returns:
            Dictionary with operation result
        """
        try:
            # Prepare tags and metadata with optional hostname tagging
            final_tags = tags or []
            final_metadata = metadata or {}

            # Apply hostname tagging if provided (for consistent source tracking)
            if client_hostname:
                source_tag = f"source:{client_hostname}"
                if source_tag not in final_tags:
                    final_tags.append(source_tag)
                final_metadata["hostname"] = client_hostname

            # Generate content hash for deduplication
            content_hash = generate_content_hash(content)

            # Compute emotional valence and initial salience score
            emotional_valence = self._compute_emotional_valence(content)
            emotional_mag = emotional_valence.get("magnitude", 0.0) if emotional_valence else 0.0
            explicit_importance = float(final_metadata.pop("importance", 0.0))
            salience_score = self._compute_salience_score(
                emotional_magnitude=emotional_mag,
                access_count=0,
                explicit_importance=explicit_importance,
            )

            # Process content if auto-splitting is enabled and content exceeds max length
            max_length = self.storage.max_content_length
            if ENABLE_AUTO_SPLIT and max_length and len(content) > max_length:
                # Split content into chunks
                chunks = split_content(
                    content,
                    max_length=max_length,
                    preserve_boundaries=CONTENT_PRESERVE_BOUNDARIES,
                    overlap=CONTENT_SPLIT_OVERLAP,
                )
                stored_memories = []

                for i, chunk in enumerate(chunks):
                    chunk_hash = generate_content_hash(chunk)
                    chunk_metadata = final_metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["original_hash"] = content_hash

                    # Each chunk gets its own emotional analysis
                    chunk_valence = self._compute_emotional_valence(chunk)
                    chunk_mag = chunk_valence.get("magnitude", 0.0) if chunk_valence else 0.0
                    chunk_salience = self._compute_salience_score(
                        emotional_magnitude=chunk_mag,
                        explicit_importance=explicit_importance,
                    )

                    memory = Memory(
                        content=chunk,
                        content_hash=chunk_hash,
                        tags=final_tags,
                        memory_type=memory_type,
                        metadata=chunk_metadata,
                        emotional_valence=chunk_valence,
                        salience_score=chunk_salience,
                    )

                    success, message = await self.storage.store(memory)
                    if success:
                        stored_memories.append(self._format_memory_response(memory))

                result = {
                    "success": True,
                    "memories": stored_memories,
                    "total_chunks": len(chunks),
                    "original_hash": content_hash,
                }

                # Detect contradictions against the full (unsplit) content
                interference = await self._detect_contradictions(content)
                if interference.has_contradictions:
                    result["interference"] = interference.to_dict()
                    # Create graph edges for each stored chunk
                    for mem_dict in stored_memories:
                        await self._create_contradiction_edges(
                            mem_dict["content_hash"], interference.contradictions,
                        )

                return result
            else:
                # Store as single memory
                memory = Memory(
                    content=content,
                    content_hash=content_hash,
                    tags=final_tags,
                    memory_type=memory_type,
                    metadata=final_metadata,
                    emotional_valence=emotional_valence,
                    salience_score=salience_score,
                )

                success, message = await self.storage.store(memory)

                if success:
                    # Create corresponding graph node (non-blocking, non-fatal)
                    if self._graph is not None:
                        try:
                            await self._graph.ensure_memory_node(content_hash, memory.created_at)
                        except Exception as e:
                            logger.warning(f"Graph node creation failed (non-fatal): {e}")

                    result = {"success": True, "memory": self._format_memory_response(memory)}

                    # Detect contradictions with existing memories
                    interference = await self._detect_contradictions(content)
                    if interference.has_contradictions:
                        result["interference"] = interference.to_dict()
                        await self._create_contradiction_edges(
                            content_hash, interference.contradictions,
                        )

                    return result
                else:
                    return {"success": False, "error": message}

        except ValueError as e:
            # Handle validation errors specifically
            logger.warning(f"Validation error storing memory: {e}")
            return {"success": False, "error": f"Invalid memory data: {str(e)}"}
        except ConnectionError as e:
            # Handle storage connectivity issues
            logger.error(f"Storage connection error: {e}")
            return {"success": False, "error": f"Storage connection failed: {str(e)}"}
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error storing memory: {e}")
            return {"success": False, "error": f"Failed to store memory: {str(e)}"}

    async def retrieve_memories(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve memories using hybrid search (semantic + tag matching).

        Combines vector similarity with automatic tag extraction for improved retrieval.
        When query terms match existing tags, those memories receive a score boost.
        This solves the "rathole problem" where project-specific queries return
        semantically similar but categorically unrelated results.

        Hybrid search is enabled by default. To opt-out to pure vector search:
        - Set environment variable MCP_MEMORY_HYBRID_ALPHA=1.0

        Args:
            query: Search query string (tags extracted automatically)
            page: Page number (1-indexed)
            page_size: Number of results per page
            tags: Optional explicit tag filtering (bypasses hybrid, uses vector only)
            memory_type: Optional memory type filtering
            min_similarity: Optional minimum similarity threshold (0.0 to 1.0)

        Returns:
            Dictionary with search results and pagination metadata
        """
        try:
            config = settings.hybrid_search

            # If tags explicitly provided (even empty list), skip hybrid and use pure vector search
            # Distinguishes "no tags" (None) from "explicit empty tags" ([])
            if tags is not None:
                return await self._retrieve_vector_only(query, page, page_size, tags, memory_type, min_similarity)

            # Get cached tags for keyword extraction
            existing_tags = await self._get_cached_tags()

            # Extract potential tag keywords from query
            keywords = extract_query_keywords(query, existing_tags)

            # If no keywords match existing tags, fall back to vector-only
            if not keywords:
                return await self._retrieve_vector_only(query, page, page_size, None, memory_type, min_similarity)

            # Determine alpha (explicit > env > adaptive)
            corpus_size = await self.storage.count()
            alpha = get_adaptive_alpha(corpus_size, len(keywords), config)

            # If alpha is 1.0, pure vector search (opt-out)
            if alpha >= 1.0:
                return await self._retrieve_vector_only(query, page, page_size, None, memory_type, min_similarity)

            # Fetch larger result set for RRF combination
            # Must cover offset + page_size to support pagination beyond page 1
            offset = (page - 1) * page_size
            fetch_size = min(max(page_size * 3, offset + page_size), 100)

            # Parallel fetch: vector results + tag-matching memories
            vector_task = self.storage.retrieve(
                query=query,
                n_results=fetch_size,
                tags=None,
                memory_type=memory_type,
                min_similarity=min_similarity,
                offset=0,
            )
            tag_task = self.storage.search_by_tags(
                tags=keywords,
                match_all=False,  # ANY tag matches
                limit=fetch_size,
            )

            vector_results, tag_matches = await asyncio.gather(vector_task, tag_task)

            # Combine using RRF
            combined = combine_results_rrf(vector_results, tag_matches, alpha)

            # Apply recency decay
            if config.recency_decay > 0:
                combined = apply_recency_decay(combined, config.recency_decay)

            # Apply spreading activation boost from graph layer
            all_hashes = [m.content_hash for m, _, _ in combined]
            graph_boosts = await self._compute_graph_boosts(all_hashes)
            if graph_boosts:
                boost_weight = settings.falkordb.spreading_activation_boost
                boosted = []
                for memory, score, debug_info in combined:
                    activation = graph_boosts.get(memory.content_hash, 0.0)
                    if activation > 0:
                        score += boost_weight * activation
                        debug_info = {**debug_info, "graph_boost": activation}
                    boosted.append((memory, score, debug_info))
                combined = sorted(boosted, key=lambda x: x[1], reverse=True)

            # Apply salience boost
            if settings.salience.enabled:
                salience_boosted = []
                for memory, score, debug_info in combined:
                    salience = memory.salience_score
                    if salience > 0:
                        score = apply_salience_boost(score, salience, settings.salience.boost_weight)
                        debug_info = {**debug_info, "salience_boost": salience}
                    salience_boosted.append((memory, score, debug_info))
                combined = sorted(salience_boosted, key=lambda x: x[1], reverse=True)

            # Apply spaced repetition boost
            if settings.spaced_repetition.enabled:
                spacing_boosted = []
                for memory, score, debug_info in combined:
                    spacing = compute_spacing_quality(memory.access_timestamps)
                    if spacing > 0:
                        score = apply_spacing_boost(score, spacing, settings.spaced_repetition.boost_weight)
                        debug_info = {**debug_info, "spacing_quality": spacing}
                    spacing_boosted.append((memory, score, debug_info))
                combined = sorted(spacing_boosted, key=lambda x: x[1], reverse=True)

            # Apply pagination to combined results (offset calculated above for fetch_size)
            total = len(combined)
            paginated = combined[offset : offset + page_size]

            # Format results and collect hashes for Hebbian co-access
            results = []
            result_hashes = []
            result_spacing = []
            for memory, score, debug_info in paginated:
                memory_dict = self._format_memory_response(memory)
                memory_dict["similarity_score"] = score
                memory_dict["hybrid_debug"] = debug_info
                results.append(memory_dict)
                result_hashes.append(memory.content_hash)
                result_spacing.append(
                    compute_spacing_quality(memory.access_timestamps)
                    if settings.spaced_repetition.enabled
                    else 0.0
                )

            # Fire Hebbian co-access events with spacing qualities for LTP
            await self._fire_hebbian_co_access(result_hashes, result_spacing or None)

            # Track access counts (fire-and-forget)
            self._fire_access_count_updates(result_hashes)

            return {
                "memories": results,
                "query": query,
                "hybrid_enabled": True,
                "alpha_used": alpha,
                "keywords_extracted": keywords,
                **self._build_pagination_metadata(total, page, page_size),
            }

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"memories": [], "query": query, "error": f"Failed to retrieve memories: {str(e)}"}

    async def search_by_tag(
        self,
        tags: str | list[str],
        match_all: bool = False,
        page: int = 1,
        page_size: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, list[MemoryResult] | str | bool | int]:
        """
        Search memories by tags with flexible matching options, pagination, and optional date filtering.

        Args:
            tags: Tag or list of tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag
            page: Page number (1-indexed)
            page_size: Number of results per page
            start_date: Filter memories from this date (YYYY-MM-DD format)
            end_date: Filter memories until this date (YYYY-MM-DD format)

        Returns:
            Dictionary with matching memories and pagination metadata
        """
        try:
            # Normalize tags to list
            if isinstance(tags, str):
                tags = [tags]

            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Convert date strings to timestamps if provided
            from datetime import datetime

            start_timestamp = None
            end_timestamp = None

            if start_date:
                dt = datetime.fromisoformat(start_date)
                start_timestamp = datetime(dt.year, dt.month, dt.day).timestamp()

            if end_date:
                dt = datetime.fromisoformat(end_date)
                end_timestamp = datetime(dt.year, dt.month, dt.day, 23, 59, 59).timestamp()

            # Get total count for pagination
            total = await self.storage.count_tag_search(
                tags=tags, match_all=match_all, start_timestamp=start_timestamp, end_timestamp=end_timestamp
            )

            # Search using database-level filtering
            memories = await self.storage.search_by_tag(
                tags=tags,
                limit=page_size,
                offset=offset,
                match_all=match_all,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

            # Format results
            results = []
            for item in memories:
                # Handle both Memory and MemoryQueryResult objects
                if hasattr(item, "memory"):
                    results.append(self._format_memory_response(item.memory))
                else:
                    results.append(self._format_memory_response(item))

            # Determine match type description
            match_type = "ALL" if match_all else "ANY"

            return {
                "memories": results,
                "tags": tags,
                "match_type": match_type,
                **self._build_pagination_metadata(total, page, page_size),
            }

        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return {
                "memories": [],
                "tags": tags if isinstance(tags, list) else [tags],
                "error": f"Failed to search by tags: {str(e)}",
            }

    async def get_memory_by_hash(self, content_hash: str) -> dict[str, Any]:
        """
        Retrieve a specific memory by its content hash.

        Args:
            content_hash: The content hash of the memory

        Returns:
            Dictionary with memory data or error
        """
        try:
            # Use efficient database-level hash lookup
            memory = await self.storage.get_memory_by_hash(content_hash)

            if memory:
                return {"memory": self._format_memory_response(memory), "found": True}
            else:
                return {"found": False, "content_hash": content_hash}

        except Exception as e:
            logger.error(f"Error getting memory by hash: {e}")
            return {"found": False, "content_hash": content_hash, "error": f"Failed to get memory: {str(e)}"}

    async def delete_memory(self, content_hash: str) -> dict[str, Any]:
        """
        Delete a memory by its content hash.

        Args:
            content_hash: The content hash of the memory to delete

        Returns:
            Dictionary with operation result
        """
        try:
            success, message = await self.storage.delete(content_hash)
            if success:
                # Clean up graph node and edges (non-blocking, non-fatal)
                if self._graph is not None:
                    try:
                        await self._graph.delete_memory_node(content_hash)
                    except Exception as e:
                        logger.warning(f"Graph node deletion failed (non-fatal): {e}")
                return {"success": True, "content_hash": content_hash}
            else:
                return {"success": False, "content_hash": content_hash, "error": message}

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "content_hash": content_hash, "error": f"Failed to delete memory: {str(e)}"}

    async def check_database_health(self) -> dict[str, Any]:
        """
        Perform a health check on the memory storage system.

        Returns:
            Dictionary with health status and statistics
        """
        try:
            stats = await self.storage.get_stats()
            result = {
                "healthy": True,
                "storage_type": stats.get("backend", "unknown"),
                "total_memories": stats.get("total_memories", 0),
                "last_updated": datetime.now().isoformat(),
                **stats,
            }

            # Include graph stats if graph layer is available
            if self._graph is not None:
                try:
                    result["graph"] = await self._graph.get_graph_stats()
                except Exception as e:
                    result["graph"] = {"status": "error", "error": str(e)}

            if self._write_queue is not None:
                result["write_queue"] = self._write_queue.get_stats()

            return result

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": f"Health check failed: {str(e)}"}

    async def consolidate(self) -> dict[str, Any]:
        """
        Run a memory consolidation cycle: decay, prune, and detect duplicates.

        Implements biological sleep consolidation:
        1. Global decay: downscale ALL edge weights (synaptic homeostasis)
        2. Stale decay: extra penalty for edges not accessed recently
        3. Prune: delete edges that fell below threshold
        4. Duplicate detection: find near-identical memories via vector similarity

        Idempotent: safe to run multiple times. Each run applies one decay cycle.
        Crash-safe: each phase uses atomic graph operations. Partial completion
        leaves the graph in a valid state — the next run finishes the work.

        Returns:
            Dict with consolidation statistics
        """
        config = settings.consolidation
        stats: dict[str, Any] = {
            "edges_decayed": 0,
            "stale_edges_decayed": 0,
            "edges_pruned": 0,
            "orphan_nodes": 0,
            "duplicates_found": 0,
            "duplicates_merged": 0,
            "errors": [],
        }

        # Phase 1: Global edge decay (requires graph layer)
        if self._graph is not None:
            try:
                stats["edges_decayed"] = await self._graph.decay_all_edges(
                    decay_factor=config.decay_factor,
                    limit=config.max_edges_per_run,
                )
            except Exception as e:
                logger.error(f"Consolidation global decay failed: {e}")
                stats["errors"].append(f"global_decay: {e}")

            # Phase 2: Extra decay for stale edges
            # Note: stale edges get BOTH global decay (phase 1) and stale decay (phase 2).
            # E.g., with defaults: weight * 0.9 * 0.5 = weight * 0.45 per run.
            # This aggressive compounding is intentional — unused synapses are pruned fast.
            try:
                stale_before = time.time() - (config.stale_edge_days * 86400)
                stats["stale_edges_decayed"] = await self._graph.decay_stale_edges(
                    stale_before=stale_before,
                    decay_factor=config.stale_decay_factor,
                    limit=config.max_edges_per_run,
                )
            except Exception as e:
                logger.error(f"Consolidation stale decay failed: {e}")
                stats["errors"].append(f"stale_decay: {e}")

            # Phase 3: Prune weak edges
            try:
                stats["edges_pruned"] = await self._graph.prune_weak_edges(
                    threshold=config.prune_threshold,
                    limit=config.max_edges_per_run,
                )
            except Exception as e:
                logger.error(f"Consolidation pruning failed: {e}")
                stats["errors"].append(f"prune: {e}")

            # Phase 4: Report orphan nodes (informational only)
            try:
                orphans = await self._graph.get_orphan_nodes(limit=100)
                stats["orphan_nodes"] = len(orphans)
            except Exception as e:
                logger.error(f"Consolidation orphan detection failed: {e}")
                stats["errors"].append(f"orphan_detection: {e}")
        else:
            logger.info("Consolidation: graph layer not enabled, skipping edge operations")

        # Phase 5: Duplicate detection via vector similarity
        try:
            dup_stats = await self._find_and_merge_duplicates(
                similarity_threshold=config.duplicate_similarity_threshold,
                max_pairs=config.max_duplicates_per_run,
            )
            stats["duplicates_found"] = dup_stats["found"]
            stats["duplicates_merged"] = dup_stats["merged"]
        except Exception as e:
            logger.error(f"Consolidation duplicate detection failed: {e}")
            stats["errors"].append(f"duplicate_detection: {e}")

        stats["success"] = len(stats["errors"]) == 0
        logger.info(f"Consolidation complete: {stats}")
        return stats

    async def _find_and_merge_duplicates(
        self,
        similarity_threshold: float = 0.95,
        max_pairs: int = 100,
    ) -> dict[str, int]:
        """
        Find and merge near-duplicate memories using vector similarity.

        Strategy: iterate recent memories in batches, search for near-duplicates,
        merge by keeping the older memory and updating its tags/metadata.

        Args:
            similarity_threshold: Cosine similarity above which = duplicate
            max_pairs: Max duplicate pairs to process per run

        Returns:
            Dict with 'found' and 'merged' counts
        """
        found = 0
        merged = 0
        seen_hashes: set[str] = set()

        # Get recent memories in batches to check for duplicates
        batch_size = 50
        offset = 0
        memories = await self.storage.get_all_memories(limit=batch_size, offset=offset)

        while memories and found < max_pairs:
            for memory in memories:
                if memory.content_hash in seen_hashes:
                    continue
                seen_hashes.add(memory.content_hash)

                # Search for near-duplicates of this memory's content
                try:
                    similar = await self.storage.retrieve(
                        query=memory.content,
                        n_results=5,
                        min_similarity=similarity_threshold,
                    )
                except Exception:
                    continue

                for match in similar:
                    match_hash = match.memory.content_hash
                    if match_hash == memory.content_hash or match_hash in seen_hashes:
                        continue

                    found += 1
                    seen_hashes.add(match_hash)

                    # Merge: keep the older memory, absorb tags from the newer one
                    keeper, victim = (
                        (memory, match.memory)
                        if (memory.created_at or 0) <= (match.memory.created_at or 0)
                        else (match.memory, memory)
                    )

                    try:
                        # Absorb victim's tags into keeper
                        merged_tags = list(set(keeper.tags) | set(victim.tags))
                        if merged_tags != keeper.tags:
                            await self.storage.update_memory_metadata(
                                keeper.content_hash,
                                {"tags": merged_tags},
                                preserve_timestamps=True,
                            )

                        # Delete victim from storage
                        await self.storage.delete(victim.content_hash)

                        # Clean up graph node for victim
                        if self._graph is not None:
                            await self._graph.delete_memory_node(victim.content_hash)

                        merged += 1
                        logger.info(
                            f"Merged duplicate: kept {keeper.content_hash[:8]}, "
                            f"removed {victim.content_hash[:8]}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to merge duplicate pair: {e}")

                    # If current memory was the victim (deleted), stop searching against it
                    if victim.content_hash == memory.content_hash:
                        break

                    if found >= max_pairs:
                        break

                if found >= max_pairs:
                    break

            offset += batch_size
            memories = await self.storage.get_all_memories(limit=batch_size, offset=offset)

        return {"found": found, "merged": merged}

    async def create_relation(
        self,
        source_hash: str,
        target_hash: str,
        relation_type: str,
    ) -> dict[str, Any]:
        """
        Create a typed relationship between two memories.

        Typed edges (RELATES_TO, PRECEDES, CONTRADICTS) are explicit knowledge
        graph relationships, unlike Hebbian edges which form implicitly through
        co-retrieval. They are lower-frequency writes created during storage or
        by explicit user action.

        Args:
            source_hash: Content hash of the source memory
            target_hash: Content hash of the target memory
            relation_type: One of RELATES_TO, PRECEDES, CONTRADICTS

        Returns:
            Dict with success status and relation details
        """
        if self._graph is None:
            return {
                "success": False,
                "error": "Graph layer not enabled (set MCP_FALKORDB_ENABLED=true)",
            }

        try:
            created = await self._graph.create_typed_edge(
                source_hash=source_hash,
                target_hash=target_hash,
                relation_type=relation_type,
            )
            if created:
                return {
                    "success": True,
                    "source": source_hash,
                    "target": target_hash,
                    "relation_type": relation_type.upper(),
                }
            else:
                return {
                    "success": False,
                    "error": "Source or target memory not found in graph",
                    "source": source_hash,
                    "target": target_hash,
                }
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to create relation: {e}")
            return {"success": False, "error": f"Failed to create relation: {e}"}

    async def get_relations(
        self,
        content_hash: str,
        relation_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get typed relationships for a memory.

        Args:
            content_hash: Memory to get relations for
            relation_type: Filter by type (None = all typed edges)

        Returns:
            Dict with relations list
        """
        if self._graph is None:
            return {"relations": [], "content_hash": content_hash}

        try:
            edges = await self._graph.get_typed_edges(
                content_hash=content_hash,
                relation_type=relation_type,
            )
            return {
                "relations": edges,
                "content_hash": content_hash,
                "count": len(edges),
            }
        except ValueError as e:
            return {"relations": [], "content_hash": content_hash, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to get relations: {e}")
            return {"relations": [], "content_hash": content_hash, "error": str(e)}

    async def delete_relation(
        self,
        source_hash: str,
        target_hash: str,
        relation_type: str,
    ) -> dict[str, Any]:
        """
        Delete a typed relationship between two memories.

        Args:
            source_hash: Content hash of the source memory
            target_hash: Content hash of the target memory
            relation_type: One of RELATES_TO, PRECEDES, CONTRADICTS

        Returns:
            Dict with success status
        """
        if self._graph is None:
            return {
                "success": False,
                "error": "Graph layer not enabled (set MCP_FALKORDB_ENABLED=true)",
            }

        try:
            deleted = await self._graph.delete_typed_edge(
                source_hash=source_hash,
                target_hash=target_hash,
                relation_type=relation_type,
            )
            return {
                "success": deleted,
                "source": source_hash,
                "target": target_hash,
                "relation_type": relation_type.upper(),
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to delete relation: {e}")
            return {"success": False, "error": f"Failed to delete relation: {e}"}

    def _format_memory_response(self, memory: Memory) -> MemoryResult:
        """
        Format a memory object for API response.

        Args:
            memory: The memory object to format

        Returns:
            Formatted memory dictionary
        """
        return {
            "content": memory.content,
            "content_hash": memory.content_hash,
            "tags": memory.tags,
            "memory_type": memory.memory_type,
            "metadata": memory.metadata,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "created_at_iso": memory.created_at_iso,
            "updated_at_iso": memory.updated_at_iso,
            "emotional_valence": memory.emotional_valence,
            "salience_score": memory.salience_score,
        }
