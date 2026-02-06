"""
Unit tests for HebbianWriteQueue.

Tests the CQRS write queue consumer with mocked Redis/FalkorDB.
Validates enqueue, batch processing, and edge mutation operations.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_pool():
    """Mock Redis connection pool."""
    return MagicMock()


@pytest.fixture
def mock_graph():
    """Mock FalkorDB graph."""
    return AsyncMock()


@pytest.fixture
def queue(mock_pool, mock_graph):
    """Create a HebbianWriteQueue with mocked dependencies."""
    from mcp_memory_service.graph.queue import HebbianWriteQueue

    return HebbianWriteQueue(
        pool=mock_pool,
        graph=mock_graph,
        queue_key="test:queue",
        batch_size=10,
        poll_interval=0.1,
    )


class TestEnqueue:
    """Test producer-side enqueue operations."""

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.queue.aioredis.Redis")
    async def test_enqueue_strengthen(self, mock_redis_cls, queue):
        mock_conn = AsyncMock()
        mock_redis_cls.return_value = mock_conn

        await queue.enqueue_strengthen("hash_a", "hash_b")

        mock_conn.lpush.assert_called_once()
        key, payload = mock_conn.lpush.call_args[0]
        assert key == "test:queue"

        data = json.loads(payload)
        assert data["op"] == "strengthen"
        assert data["source"] == "hash_a"
        assert data["target"] == "hash_b"
        assert "ts" in data

        assert queue._stats["enqueued"] == 1

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.queue.aioredis.Redis")
    async def test_enqueue_weaken(self, mock_redis_cls, queue):
        mock_conn = AsyncMock()
        mock_redis_cls.return_value = mock_conn

        await queue.enqueue_weaken("hash_a", "hash_b", decay_factor=0.9)

        data = json.loads(mock_conn.lpush.call_args[0][1])
        assert data["op"] == "weaken"
        assert data["decay"] == 0.9

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.queue.aioredis.Redis")
    async def test_enqueue_delete_edge(self, mock_redis_cls, queue):
        mock_conn = AsyncMock()
        mock_redis_cls.return_value = mock_conn

        await queue.enqueue_delete_edge("hash_a", "hash_b")

        data = json.loads(mock_conn.lpush.call_args[0][1])
        assert data["op"] == "delete"


class TestExecuteWrite:
    """Test write operation execution against the graph."""

    @pytest.mark.asyncio
    async def test_strengthen_edge_uses_multiplicative_update(self, queue, mock_graph):
        """Verify the Cypher uses multiplicative w * (1 + rate), not additive w + delta."""
        await queue._strengthen_edge("src", "dst", 1700000000.0)

        mock_graph.query.assert_called_once()
        query = mock_graph.query.call_args[0][0]
        params = mock_graph.query.call_args[1]["params"]

        # Core Cypher structure
        assert "MERGE" in query
        assert "HEBBIAN" in query
        assert "co_access_count" in query

        # Multiplicative formula: w * (1.0 + $rate)
        assert "e.weight * (1.0 + $rate)" in query

        # Must NOT contain additive formula
        assert "e.weight + $delta" not in query

        # Params use instance config values
        assert params["init_w"] == 0.1
        assert params["rate"] == 0.15
        assert params["max_w"] == 1.0

    @pytest.mark.asyncio
    async def test_strengthen_edge_uses_custom_config(self, mock_pool, mock_graph):
        """Verify custom Hebbian parameters are passed through to Cypher."""
        from mcp_memory_service.graph.queue import HebbianWriteQueue

        custom_queue = HebbianWriteQueue(
            pool=mock_pool,
            graph=mock_graph,
            initial_weight=0.2,
            strengthen_rate=0.25,
            max_weight=2.0,
        )
        await custom_queue._strengthen_edge("src", "dst", 1700000000.0)

        params = mock_graph.query.call_args[1]["params"]
        assert params["init_w"] == 0.2
        assert params["rate"] == 0.25
        assert params["max_w"] == 2.0

    @pytest.mark.asyncio
    async def test_weaken_edge(self, queue, mock_graph):
        await queue._weaken_edge("src", "dst", 0.95)

        mock_graph.query.assert_called_once()
        query = mock_graph.query.call_args[0][0]
        assert "weight * $decay" in query

    @pytest.mark.asyncio
    async def test_delete_edge(self, queue, mock_graph):
        await queue._delete_edge("src", "dst")

        mock_graph.query.assert_called_once()
        query = mock_graph.query.call_args[0][0]
        assert "DELETE e" in query

    @pytest.mark.asyncio
    async def test_execute_write_dispatches_correctly(self, queue, mock_graph):
        """Test the dispatcher routes to correct handler."""
        import time

        ts = time.time()
        await queue._execute_write({"op": "strengthen", "source": "a", "target": "b", "ts": ts})
        assert mock_graph.query.call_count == 1

        mock_graph.query.reset_mock()
        await queue._execute_write({"op": "weaken", "source": "a", "target": "b", "decay": 0.9, "ts": ts})
        assert mock_graph.query.call_count == 1

        mock_graph.query.reset_mock()
        await queue._execute_write({"op": "delete", "source": "a", "target": "b", "ts": ts})
        assert mock_graph.query.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_write_unknown_op(self, queue, mock_graph):
        """Unknown operations are logged but don't crash."""
        await queue._execute_write({"op": "explode", "source": "a", "target": "b", "ts": 0})
        mock_graph.query.assert_not_called()


class TestPopBatch:
    """Test batch dequeue from Redis."""

    @pytest.mark.asyncio
    async def test_pop_batch_single_item(self, queue):
        """BRPOP returns one item, RPOP returns None -> batch of 1."""
        mock_conn = AsyncMock()
        mock_conn.brpop.return_value = ("test:queue", json.dumps({"op": "strengthen", "source": "a", "target": "b", "ts": 0}))
        mock_conn.rpop.return_value = None

        batch = await queue._pop_batch(mock_conn)
        assert len(batch) == 1
        assert batch[0]["op"] == "strengthen"

    @pytest.mark.asyncio
    async def test_pop_batch_multiple_items(self, queue):
        """BRPOP returns one, then RPOP returns more."""
        items = [
            json.dumps({"op": "strengthen", "source": f"s{i}", "target": f"t{i}", "ts": 0})
            for i in range(5)
        ]

        mock_conn = AsyncMock()
        mock_conn.brpop.return_value = ("test:queue", items[0])
        mock_conn.rpop.side_effect = items[1:] + [None]

        batch = await queue._pop_batch(mock_conn)
        assert len(batch) == 5

    @pytest.mark.asyncio
    async def test_pop_batch_empty_queue(self, queue):
        """BRPOP timeout returns None -> empty batch."""
        mock_conn = AsyncMock()
        mock_conn.brpop.return_value = None

        batch = await queue._pop_batch(mock_conn)
        assert len(batch) == 0


class TestQueueStats:
    """Test stats and monitoring."""

    def test_get_stats(self, queue):
        stats = queue.get_stats()
        assert stats["queue_key"] == "test:queue"
        assert stats["running"] is False
        assert stats["enqueued"] == 0
        assert stats["processed"] == 0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    @patch("mcp_memory_service.graph.queue.aioredis.Redis")
    async def test_get_queue_depth(self, mock_redis_cls, queue):
        mock_conn = AsyncMock()
        mock_conn.llen.return_value = 42
        mock_redis_cls.return_value = mock_conn

        depth = await queue.get_queue_depth()
        assert depth == 42
