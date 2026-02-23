"""
Memory Service - Shared business logic for memory operations.

This service contains the shared business logic that was previously duplicated
between mcp_server.py and server.py. It provides a single source of truth for
all memory operations, eliminating the DRY violation and ensuring consistent behavior.
"""

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, TypedDict

from ..config import (
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
    ENABLE_AUTO_SPLIT,
    settings,
)
from ..graph.client import GraphClient
from ..graph.queue import HebbianWriteQueue
from ..memory_tiers import ThreeTierMemory
from ..models.audit_log import AuditLog
from ..models.memory import Memory
from ..models.search_log import SearchLog
from ..storage.base import MemoryStorage
from ..utils.content_splitter import split_content
from ..utils.emotional_analysis import analyze_emotion
from ..utils.encoding_context import (
    EncodingContext,
    apply_context_boost,
    capture_encoding_context,
    compute_context_similarity,
)
from ..utils.hashing import generate_content_hash
from ..utils.hybrid_search import (
    TAG_ONLY_BASE_SCORE,
    apply_recency_decay,
    combine_results_rrf,
    combine_results_rrf_multi,
    extract_query_keywords,
    get_adaptive_alpha,
    temporal_decay_factor,
)
from ..utils.interference import ContradictionSignal, InterferenceResult, detect_contradiction_signals
from ..utils.query_intent import get_analyzer
from ..utils.salience import SalienceFactors, apply_salience_boost, compute_salience
from ..utils.spaced_repetition import apply_spacing_boost, compute_spacing_quality
from ..utils.summariser import summarise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CacheKit-backed caches (L1 in-process + L2 Redis when REDIS_URL is set)
# ---------------------------------------------------------------------------
_storage_ref: Any = None

try:
    from cachekit import cache as _cachekit_cache

    from ..config import SemanticTagSettings as _SemanticTagSettings

    _tag_embedding_ttl = _SemanticTagSettings().cache_ttl

    # Use Redis if REDIS_URL is set, otherwise L1-only
    _ck_kwargs: dict[str, Any] = {"serializer": "auto"}
    if not os.environ.get("REDIS_URL") and not os.environ.get("CACHEKIT_REDIS_URL"):
        _ck_kwargs["backend"] = None

    @_cachekit_cache(ttl=60, namespace="mcp_memory_tags", **_ck_kwargs)
    async def _cached_fetch_all_tags() -> set[str]:
        """Fetch all tags from storage, cached 60s."""
        return set(await _storage_ref.get_all_tags())

    @_cachekit_cache(ttl=_tag_embedding_ttl, namespace="mcp_memory_tag_embeddings", **_ck_kwargs)
    async def _cached_get_tag_embeddings() -> dict:
        """Cache raw tag+embedding data; numpy index rebuilt on retrieval."""
        tags = list(await _cached_fetch_all_tags())
        if not tags:
            return {"tags": [], "embeddings": []}
        embeddings = await _storage_ref.generate_embeddings_batch(tags)
        return {"tags": tags, "embeddings": [list(e) for e in embeddings]}

    _CACHEKIT_AVAILABLE = True
except ImportError:
    _CACHEKIT_AVAILABLE = False


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

    # Maximum search logs to keep in memory (circular buffer)
    _MAX_SEARCH_LOGS = 10000
    # Maximum audit logs to keep in memory (circular buffer)
    _MAX_AUDIT_LOGS = 10000

    def __init__(
        self,
        storage: MemoryStorage,
        graph_client: GraphClient | None = None,
        write_queue: HebbianWriteQueue | None = None,
    ):
        self.storage = storage
        self._graph = graph_client
        self._write_queue = write_queue
        self._three_tier: ThreeTierMemory | None = None
        self._search_logs: deque[SearchLog] = deque(maxlen=self._MAX_SEARCH_LOGS)
        self._audit_logs: deque[AuditLog] = deque(maxlen=self._MAX_AUDIT_LOGS)
        self._init_three_tier()

        # Set module-level storage ref for CacheKit-cached functions
        global _storage_ref  # noqa: PLW0603
        if _storage_ref is not None and _storage_ref is not storage:
            logger.warning(
                "MemoryService re-instantiated with a different storage backend; CacheKit caches will use the new storage."
            )
        _storage_ref = storage

    def _init_three_tier(self) -> None:
        """Initialize three-tier memory if enabled in config.

        Gracefully degrades: if config is unavailable or invalid, the feature
        stays disabled rather than crashing the service.
        """
        try:
            config = settings.three_tier
            if not config.enabled:
                return
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Three-tier memory init skipped: %s", exc)
            return

        async def _consolidation_callback(
            content: str,
            tags: list[str] | None,
            memory_type: str | None,
            metadata: dict[str, Any] | None,
        ) -> dict[str, Any]:
            """Store working memory item to LTM via the normal store pipeline."""
            meta = metadata.copy() if metadata else {}
            meta["source"] = "working_memory_consolidation"
            return await self.store_memory(
                content=content,
                tags=tags,
                memory_type=memory_type,
                metadata=meta,
            )

        try:
            self._three_tier = ThreeTierMemory(
                sensory_capacity=config.sensory_capacity,
                sensory_decay_ms=config.sensory_decay_ms,
                working_capacity=config.working_capacity,
                working_decay_minutes=config.working_decay_minutes,
                consolidation_callback=_consolidation_callback if config.auto_consolidate else None,
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Three-tier memory initialization failed (non-fatal): {e}")
            self._three_tier = None

    @property
    def three_tier(self) -> ThreeTierMemory | None:
        """Access the three-tier memory manager (None if disabled)."""
        return self._three_tier

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

    async def _compute_hebbian_boosts(self, result_hashes: list[str]) -> dict[str, float]:
        """
        Get Hebbian co-access boost weights for search results.

        Queries Hebbian edges where both endpoints are in the result set.
        Non-blocking, non-fatal — graph failures don't affect the read path.

        Returns:
            Dict of {content_hash: max_hebbian_weight} for connected results.
        """
        if self._graph is None or len(result_hashes) < 2:
            return {}

        if settings.falkordb.hebbian_boost <= 0:
            return {}

        try:
            return await self._graph.hebbian_boosts_within(result_hashes)
        except Exception as e:
            logger.warning(f"Hebbian boost query failed (non-fatal): {e}")
            return {}

    @staticmethod
    def _merge_tag_matches(exact: list[Memory], semantic: list[Memory]) -> list[Memory]:
        """Merge exact and semantic tag matches, deduplicating by content_hash."""
        if not semantic:
            return exact
        seen = {m.content_hash for m in exact}
        for m in semantic:
            if m.content_hash not in seen:
                exact.append(m)
                seen.add(m.content_hash)
        return exact

    async def _search_semantic_tags(self, query: str, fetch_size: int) -> list[Memory]:
        """Find memories via semantically similar tags (k-NN on tag embeddings).

        Non-fatal: returns empty list on any error.
        """
        if not settings.semantic_tag.enabled:
            return []

        try:
            from ..utils.tag_embeddings import find_semantic_tags

            # Get tag embeddings (cached in-memory with TTL)
            index = await self._get_tag_embedding_index()
            if not index["tags"]:
                return []

            # Get query embedding via public batch API
            query_embedding = (await self.storage.generate_embeddings_batch([query]))[0]

            # k-NN match
            matched_tags = find_semantic_tags(
                query_embedding,
                index,
                threshold=settings.semantic_tag.similarity_threshold,
                max_tags=settings.semantic_tag.max_tags,
            )

            if not matched_tags:
                return []

            logger.debug(f"Semantic tag matches: {matched_tags}")
            return await self.storage.search_by_tags(tags=matched_tags, match_all=False, limit=fetch_size)

        except Exception as e:
            logger.warning("Semantic tag matching failed (non-fatal): %s", e, exc_info=True)
            return []

    async def _get_tag_embedding_index(self) -> dict:
        """Get or build the tag embedding index, cached via CacheKit."""
        from ..utils.tag_embeddings import build_tag_embedding_index

        if _CACHEKIT_AVAILABLE:
            try:
                raw = await _cached_get_tag_embeddings()
                return build_tag_embedding_index(raw["tags"], raw["embeddings"])
            except Exception:
                logger.debug("CacheKit tag embedding cache miss/error, falling back to direct computation", exc_info=True)

        tags = sorted(await self._get_cached_tags())
        if not tags:
            return build_tag_embedding_index([], [])
        embeddings = await self.storage.generate_embeddings_batch(tags)
        return build_tag_embedding_index(tags, embeddings)

    async def _single_vector_search(
        self,
        query: str,
        keywords: list[str],
        fetch_size: int,
        memory_type: str | None,
        alpha: float,
    ) -> list[tuple[Memory, float, dict]]:
        """Single-vector search + tag RRF (shared by normal and fan-out fallback paths)."""
        vector_task = self.storage.retrieve(
            query=query,
            n_results=fetch_size,
            tags=None,
            memory_type=memory_type,
            min_similarity=0.0,
            offset=0,
        )
        tag_task = self.storage.search_by_tags(
            tags=keywords,
            match_all=False,
            limit=fetch_size,
        )
        semantic_tag_task = self._search_semantic_tags(query, fetch_size)

        vector_results, tag_matches, semantic_tag_matches = await asyncio.gather(vector_task, tag_task, semantic_tag_task)

        # Merge exact + semantic tag matches (deduplicate by content_hash)
        tag_matches = self._merge_tag_matches(tag_matches, semantic_tag_matches)

        return combine_results_rrf(vector_results, tag_matches, alpha)

    async def _inject_graph_neighbors(
        self,
        combined: list[tuple[Memory, float, dict]],
        seed_hashes: list[str],
        inject_limit: int = 10,
        min_activation: float = 0.05,
    ) -> list[tuple[Memory, float, dict]]:
        """Inject graph-activated neighbors as new candidates."""
        if self._graph is None or not seed_hashes:
            return combined

        try:
            graph_activation = await self._graph.spreading_activation(
                seed_hashes=seed_hashes[:5],
                max_hops=2,
                decay_factor=0.5,
                min_activation=min_activation,
            )

            if not graph_activation:
                return combined

            result_hashes = {m.content_hash for m, _, _ in combined}
            neighbor_hashes = [
                h
                for h, score in sorted(graph_activation.items(), key=lambda x: x[1], reverse=True)
                if h not in result_hashes and score >= min_activation
            ][:inject_limit]

            if not neighbor_hashes:
                return combined

            neighbors = await self.storage.get_memories_batch(neighbor_hashes)
            # Use minimum cosine score from existing results as display score floor,
            # so graph-injected entries don't get filtered by min_similarity thresholds.
            existing_scores = [s for _, s, _ in combined if s > 0]
            min_existing = min(existing_scores) if existing_scores else TAG_ONLY_BASE_SCORE
            for memory in neighbors:
                activation = graph_activation.get(memory.content_hash, 0.0)
                display_score = max(min_existing, activation)
                combined.append(
                    (
                        memory,
                        display_score,
                        {"source": "graph_injection", "activation": activation},
                    )
                )

            logger.debug(f"Graph injection: added {len(neighbors)} neighbors from {len(seed_hashes)} seeds")

        except Exception as e:
            logger.warning(f"Graph injection failed (non-fatal): {e}")

        return combined

    async def _fire_hebbian_co_access(self, content_hashes: list[str], spacing_qualities: list[float] | None = None) -> None:
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
                    confidence=signal.confidence,
                )
                logger.info(
                    f"Contradiction edge: {new_hash[:8]} -> {existing_hash[:8]} "
                    f"({signal.signal_type}, confidence={signal.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(f"Failed to create contradiction edge (non-fatal): {e}")

    async def supersede_memory(
        self,
        old_hash: str,
        new_hash: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Mark old_hash as superseded by new_hash.

        - Updates old memory's metadata: sets superseded_by and supersession_reason
        - Creates SUPERSEDES edge from new → old in the graph (audit trail)
        - Old memory is NOT deleted — it remains for historical reference
        - Superseded memories are excluded from default search results

        Args:
            old_hash: Content hash of the memory being superseded
            new_hash: Content hash of the newer memory that replaces it
            reason: Human-readable explanation for the supersession

        Returns:
            Dict with success status and details
        """
        if old_hash == new_hash:
            return {"success": False, "error": "old_id and new_id must be different memories"}

        # Verify both memories exist
        old_memory = await self.storage.get_memory_by_hash(old_hash)
        if old_memory is None:
            return {"success": False, "error": f"Memory not found: {old_hash}"}

        new_memory = await self.storage.get_memory_by_hash(new_hash)
        if new_memory is None:
            return {"success": False, "error": f"Memory not found: {new_hash}"}

        # Mark old memory as superseded in storage metadata
        try:
            metadata_update = {
                "metadata": {
                    **(old_memory.metadata or {}),
                    "superseded_by": new_hash,
                    "supersession_reason": reason,
                }
            }
            success, message = await self.storage.update_memory_metadata(old_hash, metadata_update, preserve_timestamps=True)
            if not success:
                return {"success": False, "error": f"Failed to mark memory as superseded: {message}"}
        except Exception as e:
            logger.error(f"Failed to update superseded metadata: {e}")
            return {"success": False, "error": f"Failed to update memory metadata: {e}"}

        # Create SUPERSEDES edge in graph (new → old, non-fatal)
        if self._graph is not None:
            try:
                await self._graph.ensure_memory_node(new_hash, new_memory.created_at or time.time())
                await self._graph.ensure_memory_node(old_hash, old_memory.created_at or time.time())
                await self._graph.create_typed_edge(
                    source_hash=new_hash,
                    target_hash=old_hash,
                    relation_type="SUPERSEDES",
                )
            except Exception as e:
                logger.warning(f"Graph SUPERSEDES edge creation failed (non-fatal): {e}")

        return {
            "success": True,
            "superseded": old_hash,
            "superseded_by": new_hash,
            "reason": reason,
        }

    async def get_contradictions_dashboard(self, limit: int = 20) -> dict[str, Any]:
        """
        Return all unresolved CONTRADICTS edges with memory content previews.

        Enables periodic review and resolution by human or gardener agent.

        Args:
            limit: Maximum number of contradiction pairs to return

        Returns:
            Dict with contradiction pairs and metadata
        """
        if self._graph is None:
            return {
                "success": False,
                "error": "Graph layer not enabled (set MCP_FALKORDB_ENABLED=true)",
                "pairs": [],
                "total": 0,
            }

        try:
            edges = await self._graph.get_all_contradictions(limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch contradiction edges: {e}")
            return {"success": False, "error": f"Failed to fetch contradictions: {e}", "pairs": [], "total": 0}

        pairs = []
        for edge in edges:
            pair: dict[str, Any] = {
                "memory_a_hash": edge["memory_a_hash"],
                "memory_b_hash": edge["memory_b_hash"],
                "confidence": edge.get("confidence"),
                "created_at": edge.get("created_at"),
            }

            # Fetch content previews for both memories
            for key, h in [("memory_a", edge["memory_a_hash"]), ("memory_b", edge["memory_b_hash"])]:
                try:
                    mem = await self.storage.get_memory_by_hash(h)
                    if mem:
                        content = mem.content or ""
                        pair[f"{key}_content"] = content[:200] + ("..." if len(content) > 200 else "")
                        pair[f"{key}_superseded"] = bool((mem.metadata or {}).get("superseded_by"))
                    else:
                        pair[f"{key}_content"] = None
                        pair[f"{key}_superseded"] = False
                except Exception:
                    pair[f"{key}_content"] = None
                    pair[f"{key}_superseded"] = False

            pairs.append(pair)

        return {"success": True, "pairs": pairs, "total": len(pairs), "limit": limit}

    async def _get_contradiction_warnings(self, result_hashes: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Batch-fetch contradiction warnings for a set of result hashes.

        For each hash in results, returns a list of contradicting memories
        with content previews. Non-blocking, non-fatal.

        Returns:
            Dict of {hash: [{"memory_id": str, "content": str, "edge_confidence": float | None}]}
        """
        if self._graph is None or not result_hashes:
            return {}

        try:
            edge_map = await self._graph.get_contradictions_for_hashes(result_hashes)
        except Exception as e:
            logger.warning(f"Contradiction warning fetch failed (non-fatal): {e}")
            return {}

        if not edge_map or not isinstance(edge_map, dict):
            return {}

        # Collect all unique hashes we need content for (those NOT in the result set)
        result_set = set(result_hashes)
        other_hashes: set[str] = set()
        for warnings in edge_map.values():
            for w in warnings:
                h = w["contradicts_hash"]
                if h not in result_set:
                    other_hashes.add(h)

        # Fetch content for contradicting memories not already in results
        content_map: dict[str, str] = {}
        for h in other_hashes:
            try:
                mem = await self.storage.get_memory_by_hash(h)
                if mem:
                    content_map[h] = mem.content or ""
            except Exception:
                pass

        # Build enriched warnings
        enriched: dict[str, list[dict[str, Any]]] = {}
        for result_hash, warnings in edge_map.items():
            enriched[result_hash] = [
                {
                    "memory_id": w["contradicts_hash"],
                    "content": content_map.get(w["contradicts_hash"], ""),
                    "edge_confidence": w["confidence"],
                }
                for w in warnings
            ]

        return enriched

    def _filter_superseded(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out superseded memories from a results list."""
        return [r for r in results if not (r.get("metadata") or {}).get("superseded_by")]

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

    def _log_search(
        self,
        query: str,
        start_time: float,
        result_count: int,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        hybrid_enabled: bool = True,
        keywords_extracted: list[str] | None = None,
        error: str | None = None,
        intent_enabled: bool = False,
        concepts_extracted: list[str] | None = None,
        sub_queries_count: int = 1,
    ) -> None:
        """
        Log a search query for analytics tracking.

        Non-blocking, stores in circular buffer (last 10K queries).
        Used for search analytics and query pattern analysis.
        """
        response_time_ms = (time.time() - start_time) * 1000
        log_entry = SearchLog(
            query=query,
            timestamp=time.time(),
            response_time_ms=response_time_ms,
            result_count=result_count,
            tags=tags,
            memory_type=memory_type,
            min_similarity=min_similarity,
            hybrid_enabled=hybrid_enabled,
            keywords_extracted=keywords_extracted or [],
            error=error,
            metadata={
                "intent_enabled": intent_enabled,
                "concepts_extracted": concepts_extracted or [],
                "sub_queries_count": sub_queries_count,
            },
        )
        self._search_logs.append(log_entry)

    def get_search_analytics(self, limit: int = 1000) -> dict[str, Any]:
        """
        Get aggregated search analytics from recent queries.

        Returns statistics about search patterns, performance, and popular queries.
        """
        if not self._search_logs:
            return {
                "total_searches": 0,
                "avg_response_time_ms": None,
                "popular_queries": [],
                "popular_tags": [],
                "search_types": {},
                "error_rate": 0.0,
                "queries_per_hour": 0.0,
            }

        # Get recent logs (up to limit)
        recent_logs = list(self._search_logs)[-limit:]
        total = len(recent_logs)

        # Calculate average response time
        response_times = [log.response_time_ms for log in recent_logs if log.error is None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None

        # Count query frequencies
        from collections import Counter

        query_counts = Counter(log.query for log in recent_logs)
        popular_queries = [{"query": q, "count": c} for q, c in query_counts.most_common(10)]

        # Count tag frequencies from search filters
        tag_counts = Counter()
        for log in recent_logs:
            if log.tags:
                tag_counts.update(log.tags)
        popular_tags = [{"tag": t, "count": c} for t, c in tag_counts.most_common(10)]

        # Count search types (hybrid vs vector-only)
        search_types = {
            "hybrid": sum(1 for log in recent_logs if log.hybrid_enabled),
            "vector_only": sum(1 for log in recent_logs if not log.hybrid_enabled),
        }

        # Calculate error rate
        errors = sum(1 for log in recent_logs if log.error is not None)
        error_rate = (errors / total * 100) if total > 0 else 0.0

        # Calculate queries per hour (based on time span)
        if total > 1:
            time_span_hours = (recent_logs[-1].timestamp - recent_logs[0].timestamp) / 3600
            queries_per_hour = total / time_span_hours if time_span_hours > 0 else 0.0
        else:
            queries_per_hour = 0.0

        return {
            "total_searches": total,
            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
            "popular_queries": popular_queries,
            "popular_tags": popular_tags,
            "search_types": search_types,
            "error_rate": round(error_rate, 2),
            "queries_per_hour": round(queries_per_hour, 2),
        }

    async def get_most_accessed_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get the most frequently accessed memories based on access_count.

        Returns a list of memory objects sorted by access frequency.
        """
        try:
            # Get a large sample of recent memories
            # TODO: Add dedicated storage method for sorting by access_count
            memories = await self.storage.get_recent_memories(n=1000)

            # Sort by access_count
            sorted_memories = sorted(memories, key=lambda m: m.access_count, reverse=True)

            # Limit and format
            result = []
            for memory in sorted_memories[:limit]:
                result.append(
                    {
                        "content_hash": memory.content_hash,
                        "content_preview": (memory.content or "")[:200] + ("..." if len(memory.content or "") > 200 else ""),
                        "access_count": memory.access_count,
                        "last_accessed": memory.access_timestamps[-1] if memory.access_timestamps else None,
                        "tags": memory.tags or [],
                        "memory_type": memory.memory_type,
                        "created_at": memory.created_at,
                        "salience_score": memory.salience_score,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error getting most accessed memories: {e}")
            return []

    def _log_audit(
        self,
        operation: str,
        content_hash: str,
        actor: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a memory operation for audit trail tracking.

        Non-blocking, stores in circular buffer (last 10K operations).
        Used for compliance, debugging, and change history analysis.

        Args:
            operation: Operation type (CREATE, DELETE, DELETE_RELATION)
            content_hash: Content hash of affected memory
            actor: Client hostname or user identifier
            memory_type: Memory type if applicable
            tags: Memory tags if applicable
            success: Whether operation succeeded
            error: Error message if operation failed
            metadata: Additional operation metadata
        """
        log_entry = AuditLog(
            operation=operation,
            content_hash=content_hash,
            timestamp=time.time(),
            actor=actor,
            memory_type=memory_type,
            tags=tags,
            success=success,
            error=error,
            metadata=metadata or {},
        )
        self._audit_logs.append(log_entry)

    def get_audit_trail(
        self,
        limit: int = 100,
        operation: str | None = None,
        actor: str | None = None,
        content_hash: str | None = None,
    ) -> dict[str, Any]:
        """
        Get audit trail of memory operations.

        Returns recent operations with optional filtering.

        Args:
            limit: Maximum number of log entries to return
            operation: Filter by operation type (CREATE, DELETE, DELETE_RELATION)
            actor: Filter by actor (client_hostname)
            content_hash: Filter by specific content hash

        Returns:
            Dictionary with audit log entries and statistics
        """
        if not self._audit_logs:
            return {
                "total_operations": 0,
                "operations": [],
                "operations_by_type": {},
                "operations_by_actor": {},
                "success_rate": 1.0,
            }

        # Apply filters
        filtered_logs = list(self._audit_logs)
        if operation:
            filtered_logs = [log for log in filtered_logs if log.operation == operation]
        if actor:
            filtered_logs = [log for log in filtered_logs if log.actor == actor]
        if content_hash:
            filtered_logs = [log for log in filtered_logs if log.content_hash == content_hash]

        # Sort by timestamp descending (newest first)
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)

        # Limit results
        limited_logs = filtered_logs[:limit]

        # Compute statistics from ALL filtered logs (not just limited)
        operations_by_type: dict[str, int] = {}
        operations_by_actor: dict[str, int] = {}
        success_count = 0

        for log in filtered_logs:
            # Count by operation type
            operations_by_type[log.operation] = operations_by_type.get(log.operation, 0) + 1

            # Count by actor
            if log.actor:
                operations_by_actor[log.actor] = operations_by_actor.get(log.actor, 0) + 1

            # Track success
            if log.success:
                success_count += 1

        total_ops = len(filtered_logs)
        success_rate = success_count / total_ops if total_ops > 0 else 1.0

        return {
            "total_operations": total_ops,
            "operations": [log.to_dict() for log in limited_logs],
            "operations_by_type": operations_by_type,
            "operations_by_actor": operations_by_actor,
            "success_rate": success_rate,
        }

    async def _get_cached_tags(self) -> set[str]:
        """Get all tags, cached via CacheKit (60s TTL, L1+L2)."""
        if _CACHEKIT_AVAILABLE:
            try:
                return await _cached_fetch_all_tags()
            except Exception:
                pass
        return set(await self.storage.get_all_tags())

    async def _retrieve_vector_only(
        self,
        query: str,
        page: int,
        page_size: int,
        tags: list[str] | None,
        memory_type: str | None,
        min_similarity: float | None,
        encoding_context: dict[str, Any] | None = None,
        include_superseded: bool = False,
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

        # Apply Hebbian co-access boost (within-result mutual edges)
        hebbian_boosts = await self._compute_hebbian_boosts(result_hashes)
        if hebbian_boosts:
            hb_weight = settings.falkordb.hebbian_boost
            for result in results:
                hw = hebbian_boosts.get(result["content_hash"], 0.0)
                if hw > 0:
                    result["similarity_score"] = result.get("similarity_score", 0.0) * (1 + hw * hb_weight)
                    result["hebbian_boost"] = hw
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

        # Apply encoding context boost (context-dependent retrieval)
        if settings.encoding_context.enabled and encoding_context:
            current_ctx = EncodingContext.from_dict(encoding_context)
            for result, mem in zip(results, result_memories):
                stored_ctx_data = mem.encoding_context
                if stored_ctx_data:
                    stored_ctx = EncodingContext.from_dict(stored_ctx_data)
                    ctx_sim = compute_context_similarity(stored_ctx, current_ctx)
                    if ctx_sim > 0:
                        result["similarity_score"] = apply_context_boost(
                            result.get("similarity_score", 0.0),
                            ctx_sim,
                            settings.encoding_context.boost_weight,
                        )
                        result["context_similarity"] = ctx_sim
            results.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

        # Apply temporal decay to vector-only results
        td_lambda = settings.hybrid_search.temporal_decay_lambda
        if td_lambda > 0:
            td_base = settings.hybrid_search.temporal_decay_base
            now = datetime.now(timezone.utc)
            for result, mem in zip(results, result_memories):
                try:
                    updated_at = datetime.fromisoformat(mem.updated_at_iso)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    days_old = (now - updated_at).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_old = 365
                factor = temporal_decay_factor(days_old, td_lambda, td_base)
                result["similarity_score"] = result.get("similarity_score", 0.0) * factor
                result["temporal_decay_factor"] = factor
            results.sort(key=lambda r: r.get("similarity_score", 0.0), reverse=True)

        # Cap scores at 1.0 (boosts can push above cosine range)
        for result in results:
            result["similarity_score"] = min(1.0, result.get("similarity_score", 0.0))

        # Re-filter by min_similarity after boosts (scores may have shifted)
        pre_filter_count = len(results)
        if min_similarity is not None and min_similarity > 0:
            results = [r for r in results if r.get("similarity_score", 0.0) >= min_similarity]
        filtered_below_threshold = pre_filter_count - len(results)

        # Filter out superseded memories unless explicitly requested
        if not include_superseded:
            results = self._filter_superseded(results)

        # Enrich results with contradiction warnings (non-blocking, non-fatal)
        warning_map = await self._get_contradiction_warnings([r["content_hash"] for r in results])
        if warning_map:
            for r in results:
                warnings = warning_map.get(r["content_hash"])
                if warnings:
                    r["contradictions"] = warnings

        # Fire Hebbian co-access events for co-retrieved memories
        await self._fire_hebbian_co_access(result_hashes, spacing_qualities or None)

        # Track access counts (fire-and-forget)
        self._fire_access_count_updates(result_hashes)

        response: dict[str, Any] = {
            "memories": results,
            "query": query,
            "hybrid_enabled": False,
            **self._build_pagination_metadata(total, page, page_size),
        }
        if filtered_below_threshold > 0:
            response["filtered_below_threshold"] = filtered_below_threshold
        return response

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
        summary: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a new memory with validation and content processing.

        Args:
            content: The memory content
            tags: Optional tags for the memory
            memory_type: Optional memory type classification
            metadata: Optional additional metadata
            client_hostname: Optional client hostname for source tagging
            summary: Optional client-provided summary (auto-generated if not provided)

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

            # Capture encoding context (environmental context at storage time)
            enc_context = None
            if settings.encoding_context.enabled:
                enc_context = capture_encoding_context(
                    tags=final_tags,
                    agent=client_hostname or "",
                ).to_dict()

            # Generate summary (client-provided takes precedence, mode from config)
            final_summary = await summarise(content, client_summary=summary, config=settings)

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
                        encoding_context=enc_context,
                        summary=await summarise(chunk, config=settings),
                    )

                    success, message = await self.storage.store(memory)
                    if success:
                        stored_memories.append(self._format_memory_response(memory))
                        # Log successful chunk create operation
                        self._log_audit(
                            operation="CREATE",
                            content_hash=chunk_hash,
                            actor=client_hostname,
                            memory_type=memory_type,
                            tags=final_tags,
                            success=True,
                            metadata={"chunk_index": i, "total_chunks": len(chunks), "original_hash": content_hash},
                        )

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
                            mem_dict["content_hash"],
                            interference.contradictions,
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
                    encoding_context=enc_context,
                    summary=final_summary,
                )

                success, message = await self.storage.store(memory)

                if success:
                    # Create corresponding graph node (non-blocking, non-fatal)
                    if self._graph is not None:
                        try:
                            await self._graph.ensure_memory_node(content_hash, memory.created_at)
                        except Exception as e:
                            logger.warning(f"Graph node creation failed (non-fatal): {e}")

                    # Log successful create operation
                    self._log_audit(
                        operation="CREATE",
                        content_hash=content_hash,
                        actor=client_hostname,
                        memory_type=memory_type,
                        tags=final_tags,
                        success=True,
                    )

                    result = {"success": True, "memory": self._format_memory_response(memory)}

                    # Detect contradictions with existing memories
                    interference = await self._detect_contradictions(content)
                    if interference.has_contradictions:
                        result["interference"] = interference.to_dict()
                        await self._create_contradiction_edges(
                            content_hash,
                            interference.contradictions,
                        )

                    return result
                else:
                    # Log failed create operation
                    self._log_audit(
                        operation="CREATE",
                        content_hash=content_hash,
                        actor=client_hostname,
                        memory_type=memory_type,
                        tags=final_tags,
                        success=False,
                        error=message,
                    )
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
        encoding_context: dict[str, Any] | None = None,
        include_superseded: bool = False,
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
            include_superseded: If True, include superseded memories in results

        Returns:
            Dictionary with search results and pagination metadata
        """
        start_time = time.time()
        result_count = 0
        keywords = []
        hybrid_enabled = True
        error_msg = None

        try:
            config = settings.hybrid_search

            # If tags explicitly provided (even empty list), skip hybrid and use pure vector search
            # Distinguishes "no tags" (None) from "explicit empty tags" ([])
            if tags is not None:
                return await self._retrieve_vector_only(
                    query,
                    page,
                    page_size,
                    tags,
                    memory_type,
                    min_similarity,
                    encoding_context,
                    include_superseded=include_superseded,
                )

            # Get cached tags for keyword extraction
            existing_tags = await self._get_cached_tags()

            # Extract potential tag keywords from query
            keywords = extract_query_keywords(query, existing_tags)

            # If no keywords match existing tags, fall back to vector-only
            if not keywords:
                return await self._retrieve_vector_only(
                    query,
                    page,
                    page_size,
                    None,
                    memory_type,
                    min_similarity,
                    encoding_context,
                    include_superseded=include_superseded,
                )

            # Determine alpha (explicit > env > adaptive)
            corpus_size = await self.storage.count()
            alpha = get_adaptive_alpha(corpus_size, len(keywords), config)

            # If alpha is 1.0, pure vector search (opt-out)
            if alpha >= 1.0:
                return await self._retrieve_vector_only(
                    query,
                    page,
                    page_size,
                    None,
                    memory_type,
                    min_similarity,
                    encoding_context,
                    include_superseded=include_superseded,
                )

            # Fetch larger result set for RRF combination
            # Must cover offset + page_size to support pagination beyond page 1
            offset = (page - 1) * page_size
            fetch_size = min(max(page_size * 3, offset + page_size), 100)

            # ── Fan-out: concept extraction + parallel search ──────────────
            intent_result = None
            if settings.intent.enabled:
                try:
                    analyzer = get_analyzer(
                        model_name=settings.intent.spacy_model,
                        max_sub_queries=settings.intent.max_sub_queries,
                        min_query_tokens=settings.intent.min_query_tokens,
                    )
                    intent_result = analyzer.analyze(query)
                except Exception as e:
                    logger.warning(f"Intent analysis failed (non-fatal): {e}")

            if intent_result and len(intent_result.sub_queries) > 1:
                # Multi-vector fan-out path (non-fatal: falls back to single-vector on error)
                try:
                    sub_queries = intent_result.sub_queries

                    # Stage 2: Batched embedding (single forward pass)
                    embeddings = await self.storage.generate_embeddings_batch(sub_queries)
                    if len(embeddings) != len(sub_queries):
                        raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(sub_queries)}")

                    # Stage 3: Parallel fan-out (asyncio.gather)
                    search_tasks = [
                        self.storage.search_by_vector(
                            embedding=emb,
                            n_results=fetch_size,
                            memory_type=memory_type,
                            min_similarity=0.0,
                            offset=0,
                        )
                        for emb in embeddings
                    ]
                    tag_task = self.storage.search_by_tags(tags=keywords, match_all=False, limit=fetch_size)
                    semantic_tag_task = self._search_semantic_tags(query, fetch_size)

                    all_results = await asyncio.gather(*search_tasks, tag_task, semantic_tag_task)
                    vector_result_sets = list(all_results[:-2])
                    tag_matches = all_results[-2]
                    semantic_tag_matches = all_results[-1]

                    # Merge exact + semantic tag matches
                    tag_matches = self._merge_tag_matches(tag_matches, semantic_tag_matches)

                    # Weights: original query gets 1.5x, concept sub-queries get 1.0x
                    weights = []
                    for sq in sub_queries:
                        if sq == intent_result.original_query:
                            weights.append(1.5)
                        else:
                            weights.append(1.0)

                    combined = combine_results_rrf_multi(
                        result_sets=vector_result_sets,
                        weights=weights,
                        tag_matches=tag_matches,
                        k=60,
                    )
                except Exception as e:
                    logger.warning(f"Fan-out search failed, falling back to single-vector: {e}")
                    intent_result = None  # Reset so we don't log fan-out analytics
                    combined = await self._single_vector_search(query, keywords, fetch_size, memory_type, alpha)
            else:
                # Single-vector path (existing behavior)
                combined = await self._single_vector_search(query, keywords, fetch_size, memory_type, alpha)

            # ── Graph injection: add associatively-connected neighbors ──
            if settings.intent.graph_inject and self._graph is not None:
                top_hashes = [m.content_hash for m, _, _ in combined[:5]]
                combined = await self._inject_graph_neighbors(
                    combined=combined,
                    seed_hashes=top_hashes,
                    inject_limit=settings.intent.graph_inject_limit,
                    min_activation=settings.intent.graph_inject_min_activation,
                )

            # Boost stages below operate on and re-sort by cosine display scores,
            # not RRF rank. RRF determines initial ordering; boosts refine from there.
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

            # Apply Hebbian co-access boost (within-result mutual edges)
            hebbian_boosts = await self._compute_hebbian_boosts(all_hashes)
            if hebbian_boosts:
                hb_weight = settings.falkordb.hebbian_boost
                hb_boosted = []
                for memory, score, debug_info in combined:
                    hw = hebbian_boosts.get(memory.content_hash, 0.0)
                    if hw > 0:
                        score = score * (1 + hw * hb_weight)
                        debug_info = {**debug_info, "hebbian_boost": hw}
                    hb_boosted.append((memory, score, debug_info))
                combined = sorted(hb_boosted, key=lambda x: x[1], reverse=True)

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

            # Apply encoding context boost (context-dependent retrieval)
            if settings.encoding_context.enabled and encoding_context:
                current_ctx = EncodingContext.from_dict(encoding_context)
                context_boosted = []
                for memory, score, debug_info in combined:
                    stored_ctx_data = memory.encoding_context
                    if stored_ctx_data:
                        stored_ctx = EncodingContext.from_dict(stored_ctx_data)
                        ctx_sim = compute_context_similarity(stored_ctx, current_ctx)
                        if ctx_sim > 0:
                            score = apply_context_boost(
                                score,
                                ctx_sim,
                                settings.encoding_context.boost_weight,
                            )
                            debug_info = {**debug_info, "context_similarity": ctx_sim}
                    context_boosted.append((memory, score, debug_info))
                combined = sorted(context_boosted, key=lambda x: x[1], reverse=True)

            # Apply temporal decay after all boosts (matches vector-only path order:
            # boosts represent intrinsic quality, decay represents freshness)
            if config.temporal_decay_lambda > 0:
                combined = apply_recency_decay(combined, config.temporal_decay_lambda, config.temporal_decay_base)
            elif config.recency_decay > 0:
                combined = apply_recency_decay(combined, config.recency_decay)

            # Cap scores at 1.0 and sync debug final_score after all boosts
            combined = [(m, min(1.0, s), {**d, "final_score": min(1.0, s)}) for m, s, d in combined]

            # Post-fusion threshold: apply min_similarity filter on final cosine-based scores
            pre_filter_count = len(combined)
            if min_similarity is not None and min_similarity > 0:
                combined = [(m, s, d) for m, s, d in combined if s >= min_similarity]
            filtered_below_threshold = pre_filter_count - len(combined)

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
                    compute_spacing_quality(memory.access_timestamps) if settings.spaced_repetition.enabled else 0.0
                )

            # Filter out superseded memories unless explicitly requested
            if not include_superseded:
                results = self._filter_superseded(results)

            # Enrich results with contradiction warnings (non-blocking, non-fatal)
            warning_map = await self._get_contradiction_warnings([r["content_hash"] for r in results])
            if warning_map:
                for r in results:
                    warnings = warning_map.get(r["content_hash"])
                    if warnings:
                        r["contradictions"] = warnings

            # Fire Hebbian co-access events with spacing qualities for LTP
            await self._fire_hebbian_co_access(result_hashes, result_spacing or None)

            # Track access counts (fire-and-forget)
            self._fire_access_count_updates(result_hashes)

            # Log search for analytics
            result_count = len(results)
            self._log_search(
                query=query,
                start_time=start_time,
                result_count=result_count,
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                hybrid_enabled=True,
                keywords_extracted=keywords,
                intent_enabled=bool(intent_result and len(intent_result.sub_queries) > 1),
                concepts_extracted=intent_result.concepts if intent_result else [],
                sub_queries_count=len(intent_result.sub_queries) if intent_result else 1,
            )

            response: dict[str, Any] = {
                "memories": results,
                "query": query,
                "hybrid_enabled": True,
                "alpha_used": alpha,
                "keywords_extracted": keywords,
                **self._build_pagination_metadata(total, page, page_size),
            }
            if filtered_below_threshold > 0:
                response["filtered_below_threshold"] = filtered_below_threshold
            return response

        except Exception as e:
            error_msg = f"Failed to retrieve memories: {str(e)}"
            logger.error(f"Error retrieving memories: {e}")

            # Log error for analytics
            self._log_search(
                query=query,
                start_time=start_time,
                result_count=0,
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                hybrid_enabled=hybrid_enabled,
                keywords_extracted=keywords,
                error=error_msg,
            )

            return {"memories": [], "query": query, "error": error_msg}

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

                # Log successful delete operation
                self._log_audit(
                    operation="DELETE",
                    content_hash=content_hash,
                    actor=None,  # TODO: Add client_hostname parameter to track actor
                    success=True,
                )
                return {"success": True, "content_hash": content_hash}
            else:
                # Log failed delete operation
                self._log_audit(
                    operation="DELETE",
                    content_hash=content_hash,
                    actor=None,
                    success=False,
                    error=message,
                )
                return {"success": False, "content_hash": content_hash, "error": message}

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            # Log exception during delete
            self._log_audit(
                operation="DELETE",
                content_hash=content_hash,
                actor=None,
                success=False,
                error=str(e),
            )
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
                        logger.info(f"Merged duplicate: kept {keeper.content_hash[:8]}, removed {victim.content_hash[:8]}")
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

            # Log relation delete operation
            self._log_audit(
                operation="DELETE_RELATION",
                content_hash=source_hash,
                actor=None,  # TODO: Add client_hostname parameter to track actor
                success=deleted,
                metadata={"target_hash": target_hash, "relation_type": relation_type.upper()},
            )

            return {
                "success": deleted,
                "source": source_hash,
                "target": target_hash,
                "relation_type": relation_type.upper(),
            }
        except ValueError as e:
            # Log failed relation delete
            self._log_audit(
                operation="DELETE_RELATION",
                content_hash=source_hash,
                actor=None,
                success=False,
                error=str(e),
                metadata={"target_hash": target_hash, "relation_type": relation_type},
            )
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to delete relation: {e}")
            # Log exception during relation delete
            self._log_audit(
                operation="DELETE_RELATION",
                content_hash=source_hash,
                actor=None,
                success=False,
                error=str(e),
                metadata={"target_hash": target_hash, "relation_type": relation_type},
            )
            return {"success": False, "error": f"Failed to delete relation: {e}"}

    async def scan_memories(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.5,
        output_format: str = "summary",
    ) -> dict[str, Any]:
        """
        Token-efficient memory scanning — returns summaries instead of full content.

        Uses the same Qdrant vector search as retrieve_memories but projects
        only lightweight fields (summary, tags, relevance, hash).

        Args:
            query: Semantic search query
            n_results: Maximum results (default 5)
            min_relevance: Minimum similarity threshold (default 0.5)
            output_format: Output format — "summary" (default), "full", or "both"

        Returns:
            Dictionary with scan results and metadata
        """
        valid_formats = {"summary", "full", "both"}
        if output_format not in valid_formats:
            return {
                "error": f"Invalid format '{output_format}'. Must be one of: {', '.join(sorted(valid_formats))}",
                "results": [],
                "count": 0,
            }

        try:
            results = await self.storage.retrieve(
                query=query,
                n_results=n_results,
                min_similarity=min_relevance,
            )

            scan_results = []
            for item in results:
                # Support both MemoryQueryResult-like objects and raw Memory instances
                if hasattr(item, "memory"):
                    mem = item.memory
                    relevance = getattr(item, "relevance_score", 1.0)
                else:
                    mem = item
                    relevance = 1.0

                entry: dict[str, Any] = {
                    "content_hash": mem.content_hash,
                    "relevance": round(relevance, 4),
                    "tags": mem.tags,
                    "created_at": mem.created_at,
                    "memory_type": mem.memory_type,
                }

                if output_format in ("summary", "both"):
                    # Use stored summary, or generate on-the-fly for old memories
                    entry["summary"] = mem.summary or await summarise(mem.content, config=settings)

                if output_format in ("full", "both"):
                    entry["content"] = mem.content

                scan_results.append(entry)

            return {
                "results": scan_results,
                "query": query,
                "count": len(scan_results),
                "format": output_format,
            }

        except Exception as e:
            logger.error(f"Error scanning memories: {e}")
            return {
                "results": [],
                "query": query,
                "count": 0,
                "format": output_format,
                "error": f"Failed to scan memories: {str(e)}",
            }

    async def find_similar_memories(
        self,
        query: str,
        k: int,
        distance_metric: str = "cosine",
    ) -> dict[str, Any]:
        """
        Find k most similar memories using pure vector similarity (k-nearest neighbors).

        Unlike retrieve_memories which uses hybrid search with tag boosting,
        this method performs pure k-NN vector similarity search.

        Args:
            query: Search query text
            k: Number of most similar memories to return
            distance_metric: Distance metric to use (default: "cosine")
                Supported: "cosine", "euclidean", "dot"

        Returns:
            Dictionary with:
            - count: Number of memories returned
            - memories: List of formatted memories with relevance_score
            - distance_metric: The distance metric used
            - k: The k value used
        """
        try:
            # Use storage.retrieve with n_results=k and no min_similarity threshold
            # This returns exactly k results ordered by similarity
            results = await self.storage.retrieve(
                query=query,
                n_results=k,
                min_similarity=None,  # No threshold - return top k
            )

            # Format memories for response
            memories = []
            for result in results:
                mem_dict = self._format_memory_response(result.memory)
                mem_dict["relevance_score"] = result.relevance_score
                memories.append(mem_dict)

            return {
                "count": len(memories),
                "memories": memories,
                "distance_metric": distance_metric,
                "k": k,
            }

        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return {
                "count": 0,
                "memories": [],
                "distance_metric": distance_metric,
                "k": k,
                "error": f"Failed to find similar memories: {str(e)}",
            }

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
            "encoding_context": memory.encoding_context,
        }
