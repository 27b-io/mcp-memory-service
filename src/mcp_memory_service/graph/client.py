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
from typing import Any

from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from .schema import SCHEMA_STATEMENTS

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
            "MERGE (m:Memory {content_hash: $hash}) "
            "ON CREATE SET m.created_at = $ts",
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

        return [
            {"content_hash": row[0], "weight": float(row[1]), "hops": int(row[2])}
            for row in result.result_set
        ]

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
            "MATCH ()-[e:HEBBIAN]->() "
            "WITH e LIMIT $lim "
            "SET e.weight = toFloat(e.weight * $decay) "
            "RETURN count(e)",
            params={"decay": decay_factor, "lim": limit},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        logger.info(f"Decayed {count} edges by factor {decay_factor}")
        return count

    async def decay_stale_edges(
        self, stale_before: float, decay_factor: float, limit: int = 10000
    ) -> int:
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
            "MATCH ()-[e:HEBBIAN]->() "
            "WHERE e.weight < $thresh "
            "WITH e LIMIT $lim "
            "DELETE e "
            "RETURN count(e)",
            params={"thresh": threshold, "lim": limit},
        )
        count = int(result.result_set[0][0]) if result.result_set else 0
        logger.info(f"Pruned {count} edges below weight {threshold}")
        return count

    async def get_orphan_nodes(self, limit: int = 1000) -> list[str]:
        """
        Find Memory nodes with no edges (orphans after pruning).

        These nodes still exist in Qdrant but have no graph relationships.
        They're not deleted — just returned for informational purposes.

        Returns:
            List of content_hash values for orphan nodes
        """
        result = await self._graph.query(
            "MATCH (m:Memory) "
            "WHERE NOT (m)-[:HEBBIAN]-() "
            "RETURN m.content_hash "
            "LIMIT $lim",
            params={"lim": limit},
        )
        return [row[0] for row in result.result_set]

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics for health checks."""
        try:
            node_result = await self._graph.query(
                "MATCH (m:Memory) RETURN count(m)"
            )
            edge_result = await self._graph.query(
                "MATCH ()-[e:HEBBIAN]->() RETURN count(e)"
            )

            node_count = node_result.result_set[0][0] if node_result.result_set else 0
            edge_count = edge_result.result_set[0][0] if edge_result.result_set else 0

            return {
                "graph_name": self.graph_name,
                "node_count": int(node_count),
                "edge_count": int(edge_count),
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
