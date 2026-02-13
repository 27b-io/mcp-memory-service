"""
FalkorDB graph client for cognitive memory knowledge graph.

CQRS read path: all service instances read concurrently via connection pool.
Write path is handled by HebbianWriteQueue (see queue.py).

This client is intentionally read-heavy. The only writes it performs are:
- ensure_memory_node(): idempotent MERGE on memory creation
- Schema initialization on startup

All Hebbian edge mutations go through the write queue.
"""

import logging
import time
from typing import Any

from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from .schema import RELATION_TYPES, SCHEMA_STATEMENTS

logger = logging.getLogger(__name__)


class GraphClient:
    """
    Async FalkorDB client for the memory knowledge graph.

    Manages a Redis connection pool shared between graph reads and
    the CQRS write queue (which uses the same FalkorDB/Redis instance).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        graph_name: str = "memory_graph",
        max_connections: int = 16,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.graph_name = graph_name
        self.max_connections = max_connections

        self._pool: BlockingConnectionPool | None = None
        self._db: FalkorDB | None = None
        self._graph = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool, select graph, and apply schema."""
        if self._initialized:
            return

        self._pool = BlockingConnectionPool(
            host=self.host,
            port=self.port,
            password=self.password,
            max_connections=self.max_connections,
            timeout=None,
            decode_responses=True,
        )

        self._db = FalkorDB(connection_pool=self._pool)
        self._graph = self._db.select_graph(self.graph_name)

        # Apply schema idempotently
        for stmt in SCHEMA_STATEMENTS:
            try:
                await self._graph.query(stmt)
            except Exception as e:
                # Index already exists is not an error
                if "already indexed" not in str(e).lower():
                    logger.warning(f"Schema statement warning: {stmt} -> {e}")

        self._initialized = True
        logger.info(f"GraphClient initialized: {self.host}:{self.port}/{self.graph_name}")

    @property
    def pool(self) -> BlockingConnectionPool:
        """Expose pool for write queue to share the same connection."""
        if self._pool is None:
            raise RuntimeError("GraphClient not initialized. Call initialize() first.")
        return self._pool

    @property
    def graph(self):
        """Expose graph for direct query access."""
        if self._graph is None:
            raise RuntimeError("GraphClient not initialized. Call initialize() first.")
        return self._graph

    # ── Node operations (idempotent, safe for concurrent calls) ──────────

    async def ensure_memory_node(self, content_hash: str, created_at: float) -> None:
        """
        Create a :Memory node if it doesn't exist (MERGE = idempotent).

        Called on memory store. Safe for concurrent execution because
        MERGE is atomic in FalkorDB.
        """
        await self._graph.query(
            "MERGE (m:Memory {content_hash: $hash}) ON CREATE SET m.created_at = $ts",
            params={"hash": content_hash, "ts": created_at},
        )

    async def delete_memory_node(self, content_hash: str) -> None:
        """Delete a :Memory node and all its edges."""
        await self._graph.query(
            "MATCH (m:Memory {content_hash: $hash}) DETACH DELETE m",
            params={"hash": content_hash},
        )

    # ── Read operations (concurrent, no locks) ──────────────────────────

    async def get_neighbors(
        self,
        content_hash: str,
        max_hops: int = 1,
        min_weight: float = 0.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get neighboring memories via Hebbian edges.

        Args:
            content_hash: Source memory hash
            max_hops: Maximum traversal depth (1-3)
            min_weight: Minimum edge weight threshold
            limit: Maximum results

        Returns:
            List of dicts with content_hash, weight, and hops
        """
        max_hops = min(max_hops, 3)  # Cap at 3 hops per acceptance criteria

        result = await self._graph.query(
            "MATCH (src:Memory {content_hash: $hash})"
            f"-[e:HEBBIAN*1..{max_hops}]->(dst:Memory) "
            "WHERE ALL(r IN e WHERE r.weight >= $min_w) "
            "WITH dst, e, length(e) AS hops "
            "RETURN DISTINCT dst.content_hash AS hash, "
            "reduce(w = 1.0, r IN e | w * r.weight) AS path_weight, "
            "hops "
            "ORDER BY path_weight DESC "
            "LIMIT $lim",
            params={"hash": content_hash, "min_w": min_weight, "lim": limit},
        )

        return [{"content_hash": row[0], "weight": float(row[1]), "hops": int(row[2])} for row in result.result_set]

    async def spreading_activation(
        self,
        seed_hashes: list[str],
        max_hops: int = 2,
        decay_factor: float = 0.5,
        min_activation: float = 0.01,
        limit: int = 50,
    ) -> dict[str, float]:
        """
        Multi-seed BFS with exponential hop decay (spreading activation).

        For each seed node, traverses up to max_hops via HEBBIAN edges.
        Activation = path_weight * decay_factor^hops.
        When a node is reachable from multiple seeds/paths, takes max activation.

        Args:
            seed_hashes: Content hashes of seed memories (from vector search)
            max_hops: Maximum traversal depth (capped at 3)
            decay_factor: Per-hop exponential decay
            min_activation: Minimum score to include a neighbor
            limit: Maximum activated neighbors to return

        Returns:
            Dict of {content_hash: activation_score}, sorted by score descending
        """
        if not seed_hashes:
            return {}

        max_hops = min(max_hops, 3)

        result = await self._graph.query(
            "MATCH (src:Memory)-"
            f"[e:HEBBIAN*1..{max_hops}]->"
            "(dst:Memory) "
            "WHERE src.content_hash IN $seeds "
            "AND NOT dst.content_hash IN $seeds "
            "WITH dst.content_hash AS hash, "
            "reduce(w = 1.0, r IN e | w * r.weight) AS path_weight, "
            "length(e) AS hops "
            "RETURN hash, path_weight, hops",
            params={"seeds": seed_hashes},
        )

        # Aggregate: for each destination, take max activation across all paths
        activations: dict[str, float] = {}
        for row in result.result_set:
            hash_val, path_weight, hops = row[0], float(row[1]), int(row[2])
            activation = path_weight * (decay_factor**hops)
            if activation >= min_activation:
                activations[hash_val] = max(activations.get(hash_val, 0.0), activation)

        # Sort by activation descending, apply limit
        sorted_items = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:limit]
        return dict(sorted_items)

    async def hebbian_boosts_within(self, hashes: list[str]) -> dict[str, float]:
        """
        Get max Hebbian edge weight for each node connected to another node in the set.

        For a set of search result hashes, finds all HEBBIAN edges where both
        endpoints are in the set. Returns {hash: max_weight} for nodes that have
        at least one such edge.

        Args:
            hashes: Content hashes of search results to check for mutual edges.

        Returns:
            Dict of {content_hash: max_hebbian_weight} for connected nodes.
        """
        if len(hashes) < 2:
            return {}

        result = await self._graph.query(
            "MATCH (a:Memory)-[e:HEBBIAN]->(b:Memory) "
            "WHERE a.content_hash IN $hashes AND b.content_hash IN $hashes "
            "RETURN a.content_hash AS hash, max(e.weight) AS max_weight",
            params={"hashes": hashes},
        )

        return {row[0]: float(row[1]) for row in result.result_set if float(row[1]) > 0}

    async def get_edge(self, source_hash: str, target_hash: str) -> dict[str, Any] | None:
        """Get a specific Hebbian edge between two memories."""
        result = await self._graph.query(
            "MATCH (a:Memory {content_hash: $src})"
            "-[e:HEBBIAN]->"
            "(b:Memory {content_hash: $dst}) "
            "RETURN e.weight, e.co_access_count, e.last_co_access",
            params={"src": source_hash, "dst": target_hash},
        )

        if not result.result_set:
            return None

        row = result.result_set[0]
        return {
            "weight": float(row[0]),
            "co_access_count": int(row[1]),
            "last_co_access": float(row[2]),
        }

    async def get_strongest_edges(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get the strongest Hebbian associations in the graph."""
        result = await self._graph.query(
            "MATCH (a:Memory)-[e:HEBBIAN]->(b:Memory) "
            "RETURN a.content_hash, b.content_hash, e.weight, e.co_access_count "
            "ORDER BY e.weight DESC "
            "LIMIT $lim",
            params={"lim": limit},
        )

        return [
            {
                "source": row[0],
                "target": row[1],
                "weight": float(row[2]),
                "co_access_count": int(row[3]),
            }
            for row in result.result_set
        ]

    # ── Typed relationship operations (explicit, low-frequency) ────────

    @staticmethod
    def _validate_relation_type(relation_type: str) -> str:
        """Validate and normalize a relation type against the whitelist."""
        normalized = relation_type.upper()
        if normalized not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation_type!r}. Must be one of: {', '.join(sorted(RELATION_TYPES))}")
        return normalized

    async def create_typed_edge(
        self,
        source_hash: str,
        target_hash: str,
        relation_type: str,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Create a typed relationship edge between two memories.

        Unlike Hebbian edges (auto-created on co-retrieval), typed edges are
        explicitly created by users/systems during storage. They are idempotent
        (MERGE) and carry no weight — they simply assert a relationship.

        Args:
            source_hash: Source memory content_hash
            target_hash: Target memory content_hash
            relation_type: One of RELATES_TO, PRECEDES, CONTRADICTS
            created_at: Optional timestamp (defaults to current time)
            metadata: Optional metadata dict (e.g., confidence, signal_type for CONTRADICTS)

        Returns:
            True if both source and target nodes exist and edge was created.

        Raises:
            ValueError: If relation_type is invalid or source == target.
        """
        if source_hash == target_hash:
            raise ValueError("Cannot create a relationship from a memory to itself")

        rel = self._validate_relation_type(relation_type)
        ts = created_at if created_at is not None else time.time()

        # Build SET clause for metadata
        set_clauses = ["e.created_at = $ts"]
        params = {"src": source_hash, "dst": target_hash, "ts": ts}

        if metadata:
            for key, value in metadata.items():
                set_clauses.append(f"e.{key} = ${key}")
                params[key] = value

        set_clause = ", ".join(set_clauses)

        # MATCH both nodes first — if either is missing, nothing happens.
        # This prevents dangling edges to non-existent memories.
        result = await self._graph.query(
            "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) "
            f"MERGE (a)-[e:{rel}]->(b) "
            f"ON CREATE SET {set_clause} "
            "RETURN count(e)",
            params=params,
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        return count > 0

    _VALID_DIRECTIONS = frozenset({"outgoing", "incoming", "both"})

    async def get_typed_edges(
        self,
        content_hash: str,
        relation_type: str | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get typed relationship edges for a memory.

        Args:
            content_hash: Memory to query edges for
            relation_type: Filter by type (None = all typed edges)
            direction: "outgoing", "incoming", or "both"
            limit: Maximum edges to return

        Returns:
            List of dicts with source, target, relation_type, created_at
        """
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction!r}. Must be one of: {', '.join(sorted(self._VALID_DIRECTIONS))}")

        if relation_type is not None:
            rel_types = [self._validate_relation_type(relation_type)]
        else:
            rel_types = sorted(RELATION_TYPES)

        results: list[dict[str, Any]] = []
        for rel in rel_types:
            if direction in ("outgoing", "both"):
                out = await self._graph.query(
                    f"MATCH (a:Memory {{content_hash: $hash}})-[e:{rel}]->(b:Memory) "
                    "RETURN a.content_hash, b.content_hash, e.created_at "
                    "LIMIT $lim",
                    params={"hash": content_hash, "lim": limit},
                )
                for row in out.result_set:
                    results.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "relation_type": rel,
                            "direction": "outgoing",
                            "created_at": float(row[2]) if row[2] is not None else None,
                        }
                    )

            if direction in ("incoming", "both"):
                inc = await self._graph.query(
                    f"MATCH (a:Memory)-[e:{rel}]->(b:Memory {{content_hash: $hash}}) "
                    "RETURN a.content_hash, b.content_hash, e.created_at "
                    "LIMIT $lim",
                    params={"hash": content_hash, "lim": limit},
                )
                for row in inc.result_set:
                    results.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "relation_type": rel,
                            "direction": "incoming",
                            "created_at": float(row[2]) if row[2] is not None else None,
                        }
                    )

        return results[:limit]

    async def delete_typed_edge(self, source_hash: str, target_hash: str, relation_type: str) -> bool:
        """
        Delete a typed relationship edge between two memories.

        Returns:
            True if an edge was deleted, False if none existed.
        """
        rel = self._validate_relation_type(relation_type)
        result = await self._graph.query(
            f"MATCH (a:Memory {{content_hash: $src}})-[e:{rel}]->(b:Memory {{content_hash: $dst}}) DELETE e RETURN count(e)",
            params={"src": source_hash, "dst": target_hash},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        return count > 0

    async def update_typed_edge_metadata(
        self, source_hash: str, target_hash: str, relation_type: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Update metadata on a typed relationship edge (e.g., mark as resolved).

        Useful for conflict resolution tracking:
        - resolved_at: timestamp when conflict was resolved
        - resolved_by: "auto" or "manual"
        - resolution_action: "keep_source", "keep_target", "merge", etc.

        Args:
            source_hash: Source memory content hash
            target_hash: Target memory content hash
            relation_type: Edge type (CONTRADICTS, RELATES_TO, PRECEDES)
            metadata: Dict of properties to set on the edge

        Returns:
            True if edge was found and updated, False if edge doesn't exist
        """
        rel = self._validate_relation_type(relation_type)

        # Build SET clause from metadata
        set_clauses = [f"e.{key} = ${key}" for key in metadata.keys()]
        if not set_clauses:
            return False  # No metadata to update

        set_clause = ", ".join(set_clauses)

        result = await self._graph.query(
            f"MATCH (a:Memory {{content_hash: $src}})-[e:{rel}]->(b:Memory {{content_hash: $dst}}) "
            f"SET {set_clause} "
            "RETURN count(e)",
            params={"src": source_hash, "dst": target_hash, **metadata},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        return count > 0

    async def list_unresolved_conflicts(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        List memories with unresolved CONTRADICTS edges.

        A conflict is considered unresolved if the CONTRADICTS edge
        does not have a resolved_at property.

        Returns:
            List of conflicts, each with:
            - source: source memory hash
            - target: target memory hash
            - confidence: contradiction confidence
            - signal_type: type of contradiction
            - created_at: when contradiction was detected
        """
        result = await self._graph.query(
            """
            MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory)
            WHERE e.resolved_at IS NULL
            RETURN a.content_hash, b.content_hash, e.confidence, e.signal_type, e.created_at
            LIMIT $limit
            """,
            params={"limit": limit},
        )

        conflicts = []
        if result.result_set:
            for row in result.result_set:
                conflicts.append(
                    {
                        "source": row[0],
                        "target": row[1],
                        "confidence": float(row[2]) if row[2] is not None else None,
                        "signal_type": row[3],
                        "created_at": float(row[4]) if row[4] is not None else None,
                    }
                )

        return conflicts

    # ── Consolidation operations (bulk, idempotent) ────────────────────

    async def decay_all_edges(self, decay_factor: float, limit: int = 10000) -> int:
        """
        Apply global weight decay to all Hebbian edges (synaptic homeostasis).

        This simulates biological sleep consolidation where all synapses
        are downscaled. Strong edges survive; weak ones decay toward zero.

        Idempotent: running twice applies double decay, which is expected.

        Args:
            decay_factor: Multiplicative decay (e.g., 0.9 = 10% reduction)
            limit: Max edges to process per call (safety cap)

        Returns:
            Number of edges decayed
        """
        result = await self._graph.query(
            "MATCH ()-[e:HEBBIAN]->() WITH e LIMIT $lim SET e.weight = toFloat(e.weight * $decay) RETURN count(e)",
            params={"decay": decay_factor, "lim": limit},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        logger.info(f"Decayed {count} edges by factor {decay_factor}")
        return count

    async def decay_stale_edges(self, stale_before: float, decay_factor: float, limit: int = 10000) -> int:
        """
        Apply extra decay to edges not co-accessed since the given timestamp.

        Targets edges that haven't been reinforced by recent retrieval activity.

        Args:
            stale_before: Unix timestamp — edges with last_co_access before this are stale
            decay_factor: Additional multiplicative decay for stale edges
            limit: Max edges to process

        Returns:
            Number of stale edges decayed
        """
        result = await self._graph.query(
            "MATCH ()-[e:HEBBIAN]->() "
            "WHERE e.last_co_access < $ts "
            "WITH e LIMIT $lim "
            "SET e.weight = toFloat(e.weight * $decay) "
            "RETURN count(e)",
            params={"ts": stale_before, "decay": decay_factor, "lim": limit},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        logger.info(f"Decayed {count} stale edges (before {stale_before}) by factor {decay_factor}")
        return count

    async def prune_weak_edges(self, threshold: float, limit: int = 10000) -> int:
        """
        Delete edges with weight below threshold.

        This is the pruning phase of consolidation — removes synapses too
        weak to be useful, reducing graph noise.

        Idempotent: pruning already-deleted edges is a no-op.

        Args:
            threshold: Weight below which edges are pruned
            limit: Max edges to delete per call

        Returns:
            Number of edges pruned
        """
        result = await self._graph.query(
            "MATCH ()-[e:HEBBIAN]->() WHERE e.weight < $thresh WITH e LIMIT $lim DELETE e RETURN count(e)",
            params={"thresh": threshold, "lim": limit},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        logger.info(f"Pruned {count} edges below weight {threshold}")
        return count

    async def get_orphan_nodes(self, limit: int = 1000) -> list[str]:
        """
        Find Memory nodes with no edges (orphans after pruning).

        Considers both Hebbian and typed relationship edges. A node is only
        orphaned if it has zero edges of ANY type.

        Returns:
            List of content_hash values for orphan nodes
        """
        # Build a WHERE clause that checks all edge types
        edge_checks = ["NOT (m)-[:HEBBIAN]-()"]
        for rel in sorted(RELATION_TYPES):
            edge_checks.append(f"NOT (m)-[:{rel}]-()")
        where_clause = " AND ".join(edge_checks)

        result = await self._graph.query(
            f"MATCH (m:Memory) WHERE {where_clause} RETURN m.content_hash LIMIT $lim",
            params={"lim": limit},
        )
        return [row[0] for row in result.result_set]

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics for health checks."""
        try:
            node_result = await self._graph.query("MATCH (m:Memory) RETURN count(m)")
            edge_result = await self._graph.query("MATCH ()-[e:HEBBIAN]->() RETURN count(e)")

            node_count = node_result.result_set[0][0] if node_result.result_set else 0
            hebbian_count = edge_result.result_set[0][0] if edge_result.result_set else 0

            # Count typed relationship edges
            typed_counts: dict[str, int] = {}
            for rel in sorted(RELATION_TYPES):
                rel_result = await self._graph.query(f"MATCH ()-[e:{rel}]->() RETURN count(e)")
                typed_counts[rel.lower()] = int(rel_result.result_set[0][0]) if rel_result.result_set else 0

            total_typed = sum(typed_counts.values())

            return {
                "graph_name": self.graph_name,
                "node_count": int(node_count),
                "edge_count": int(hebbian_count) + total_typed,
                "hebbian_edge_count": int(hebbian_count),
                "typed_edge_counts": typed_counts,
                "status": "operational",
            }
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {
                "graph_name": self.graph_name,
                "status": "error",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            try:
                await self._pool.aclose()
                logger.info("GraphClient connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing GraphClient pool: {e}")
            finally:
                self._pool = None
                self._db = None
                self._graph = None
                self._initialized = False
