"""
CQRS Hebbian write queue consumer.

Serializes all graph edge mutations through a single consumer using
Redis LPUSH/BRPOP on the same FalkorDB instance (free infrastructure).

Pattern:
    Producers (any service instance): LPUSH write commands as JSON
    Consumer (single instance):       BRPOP, deserialize, execute Cypher

Write operations:
    - strengthen: Increase Hebbian edge weight on co-access
    - weaken:     Decay edge weight (consolidation)
    - delete:     Remove edge (pruning)

The queue ensures:
    1. No race conditions on edge weight updates
    2. Batched writes for efficiency
    3. At-least-once delivery (BRPOP is atomic)
"""

import asyncio
import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Hebbian learning defaults (overridden by FalkorDBSettings)
INITIAL_WEIGHT = 0.1
STRENGTHEN_RATE = 0.15
MAX_WEIGHT = 1.0


class HebbianWriteQueue:
    """
    Redis-backed write queue for Hebbian edge mutations.

    Uses LPUSH to enqueue from any instance, BRPOP to dequeue
    from the single consumer. Both operations use the same
    FalkorDB/Redis connection since FalkorDB IS Redis.

    Weight updates are multiplicative: w *= (1 + rate), capped at max_weight.
    This matches biological Hebbian plasticity where stronger connections
    reinforce faster (Bi & Poo 1998).
    """

    def __init__(
        self,
        pool: aioredis.BlockingConnectionPool,
        graph,
        queue_key: str = "mcp:graph:write_queue",
        batch_size: int = 50,
        poll_interval: float = 0.5,
        initial_weight: float = INITIAL_WEIGHT,
        strengthen_rate: float = STRENGTHEN_RATE,
        max_weight: float = MAX_WEIGHT,
    ):
        """
        Args:
            pool: Shared Redis connection pool (from GraphClient)
            graph: FalkorDB graph instance (from GraphClient)
            queue_key: Redis key for the write queue
            batch_size: Max operations per consumer tick
            poll_interval: Seconds to wait on empty BRPOP
            initial_weight: Weight for newly created edges
            strengthen_rate: Multiplicative rate per co-access (w *= 1 + rate)
            max_weight: Maximum edge weight ceiling
        """
        self._pool = pool
        self._graph = graph
        self._queue_key = queue_key
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._initial_weight = initial_weight
        self._strengthen_rate = strengthen_rate
        self._max_weight = max_weight
        self._running = False
        self._consumer_task: asyncio.Task | None = None
        self._stats = {"enqueued": 0, "processed": 0, "errors": 0}

    # ── Producer methods (called from any service instance) ─────────────

    async def enqueue_strengthen(
        self, source_hash: str, target_hash: str
    ) -> None:
        """
        Enqueue a Hebbian strengthening event for co-accessed memories.

        Called when two memories are retrieved together in the same query result.
        """
        await self._enqueue({
            "op": "strengthen",
            "source": source_hash,
            "target": target_hash,
            "ts": time.time(),
        })

    async def enqueue_weaken(
        self, source_hash: str, target_hash: str, decay_factor: float = 0.95
    ) -> None:
        """Enqueue edge weight decay (used by consolidation)."""
        await self._enqueue({
            "op": "weaken",
            "source": source_hash,
            "target": target_hash,
            "decay": decay_factor,
            "ts": time.time(),
        })

    async def enqueue_delete_edge(
        self, source_hash: str, target_hash: str
    ) -> None:
        """Enqueue edge deletion (used by pruning)."""
        await self._enqueue({
            "op": "delete",
            "source": source_hash,
            "target": target_hash,
            "ts": time.time(),
        })

    async def _enqueue(self, payload: dict[str, Any]) -> None:
        """LPUSH a JSON-encoded write operation to the queue."""
        conn = aioredis.Redis(connection_pool=self._pool)
        try:
            await conn.lpush(self._queue_key, json.dumps(payload))
            self._stats["enqueued"] += 1
        finally:
            await conn.aclose()

    # ── Consumer (runs on single instance) ──────────────────────────────

    async def start_consumer(self) -> None:
        """Start the background consumer loop."""
        if self._running:
            logger.warning("Hebbian write consumer already running")
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info(f"Hebbian write consumer started (queue={self._queue_key})")

    async def stop_consumer(self) -> None:
        """Gracefully stop the consumer, draining remaining items."""
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        logger.info("Hebbian write consumer stopped")

    async def _consumer_loop(self) -> None:
        """Main consumer loop: BRPOP → process batch → repeat."""
        conn = aioredis.Redis(connection_pool=self._pool)

        try:
            while self._running:
                try:
                    batch = await self._pop_batch(conn)
                    if not batch:
                        continue

                    for op in batch:
                        try:
                            await self._execute_write(op)
                            self._stats["processed"] += 1
                        except Exception as e:
                            self._stats["errors"] += 1
                            logger.error(f"Hebbian write failed: {op.get('op')} -> {e}")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Consumer loop error: {e}")
                    await asyncio.sleep(1.0)  # Back off on unexpected errors
        finally:
            await conn.aclose()

    async def _pop_batch(self, conn: aioredis.Redis) -> list[dict[str, Any]]:
        """Pop up to batch_size items from the queue."""
        batch: list[dict[str, Any]] = []

        # First item: blocking wait
        result = await conn.brpop(self._queue_key, timeout=self._poll_interval)
        if result is None:
            return batch

        _, raw = result
        batch.append(json.loads(raw))

        # Drain up to batch_size - 1 more without blocking
        for _ in range(self._batch_size - 1):
            raw = await conn.rpop(self._queue_key)
            if raw is None:
                break
            batch.append(json.loads(raw))

        return batch

    async def _execute_write(self, op: dict[str, Any]) -> None:
        """Execute a single write operation against the graph."""
        op_type = op["op"]
        source = op["source"]
        target = op["target"]

        if op_type == "strengthen":
            await self._strengthen_edge(source, target, op["ts"])
        elif op_type == "weaken":
            await self._weaken_edge(source, target, op.get("decay", 0.95))
        elif op_type == "delete":
            await self._delete_edge(source, target)
        else:
            logger.warning(f"Unknown write op: {op_type}")

    async def _strengthen_edge(
        self, source: str, target: str, timestamp: float
    ) -> None:
        """
        Strengthen or create a Hebbian edge between two memories.

        Uses MERGE for idempotent edge creation, then applies multiplicative
        weight update: w *= (1 + rate), capped at max_weight.

        Multiplicative strengthening means stronger connections reinforce
        faster — matching biological Hebbian plasticity.
        """
        await self._graph.query(
            "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) "
            "MERGE (a)-[e:HEBBIAN]->(b) "
            "ON CREATE SET e.weight = $init_w, e.co_access_count = 1, "
            "  e.created_at = $ts, e.last_co_access = $ts "
            "ON MATCH SET e.weight = toFloat(CASE "
            "  WHEN e.weight * (1.0 + $rate) > $max_w THEN $max_w "
            "  ELSE e.weight * (1.0 + $rate) END), "
            "  e.co_access_count = e.co_access_count + 1, "
            "  e.last_co_access = $ts",
            params={
                "src": source,
                "dst": target,
                "init_w": self._initial_weight,
                "rate": self._strengthen_rate,
                "max_w": self._max_weight,
                "ts": timestamp,
            },
        )

    async def _weaken_edge(
        self, source: str, target: str, decay: float
    ) -> None:
        """Apply decay to an edge weight."""
        await self._graph.query(
            "MATCH (a:Memory {content_hash: $src})"
            "-[e:HEBBIAN]->"
            "(b:Memory {content_hash: $dst}) "
            "SET e.weight = toFloat(e.weight * $decay)",
            params={"src": source, "dst": target, "decay": decay},
        )

    async def _delete_edge(self, source: str, target: str) -> None:
        """Delete a specific Hebbian edge."""
        await self._graph.query(
            "MATCH (a:Memory {content_hash: $src})"
            "-[e:HEBBIAN]->"
            "(b:Memory {content_hash: $dst}) "
            "DELETE e",
            params={"src": source, "dst": target},
        )

    # ── Stats ───────────────────────────────────────────────────────────

    async def get_queue_depth(self) -> int:
        """Get current queue depth."""
        conn = aioredis.Redis(connection_pool=self._pool)
        try:
            return await conn.llen(self._queue_key)
        finally:
            await conn.aclose()

    def get_stats(self) -> dict[str, Any]:
        """Get consumer statistics."""
        return {
            "queue_key": self._queue_key,
            "running": self._running,
            **self._stats,
        }
