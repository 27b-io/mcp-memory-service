"""Tests for the three-tier memory model (sensory buffer + working memory)."""

import asyncio
import time
from unittest.mock import AsyncMock

from mcp_memory_service.memory_tiers import (
    SensoryBuffer,
    ThreeTierMemory,
    WorkingMemory,
)

# =============================================================================
# Sensory Buffer Tests
# =============================================================================


class TestSensoryBuffer:
    def test_push_and_size(self):
        buf = SensoryBuffer(capacity=7)
        buf.push("hello")
        buf.push("world")
        assert buf.size == 2

    def test_capacity_eviction(self):
        buf = SensoryBuffer(capacity=3)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        buf.push("d")  # Should evict "a"
        assert buf.size == 3
        contents = [item.content for item in buf.get_valid()]
        assert "a" not in contents
        assert "d" in contents

    def test_decay_expiry(self, monkeypatch):
        fake_time = [1000.0]
        monkeypatch.setattr(time, "time", lambda: fake_time[0])
        buf = SensoryBuffer(capacity=7, decay_ms=50)  # 50ms decay
        buf.push("expires-fast")
        fake_time[0] += 0.06  # Advance past expiry
        valid = buf.get_valid()
        assert len(valid) == 0

    def test_valid_items_not_expired(self):
        buf = SensoryBuffer(capacity=7, decay_ms=5000)  # 5s decay
        buf.push("still-valid")
        valid = buf.get_valid()
        assert len(valid) == 1
        assert valid[0].content == "still-valid"

    def test_flush_clears_buffer(self):
        buf = SensoryBuffer(capacity=7, decay_ms=5000)
        buf.push("item1")
        buf.push("item2")
        flushed = buf.flush()
        assert len(flushed) == 2
        assert buf.size == 0

    def test_flush_only_returns_valid(self, monkeypatch):
        fake_time = [1000.0]
        monkeypatch.setattr(time, "time", lambda: fake_time[0])
        buf = SensoryBuffer(capacity=7, decay_ms=50)
        buf.push("old-item")
        fake_time[0] += 0.06  # Expire old-item
        buf.push("new-item")
        flushed = buf.flush()
        assert len(flushed) == 1
        assert flushed[0].content == "new-item"

    def test_clear(self):
        buf = SensoryBuffer(capacity=7)
        buf.push("a")
        buf.push("b")
        buf.clear()
        assert buf.size == 0

    def test_push_with_metadata_and_tags(self):
        buf = SensoryBuffer()
        item = buf.push("test", metadata={"key": "val"}, tags=["tag1"], memory_type="note")
        assert item.metadata == {"key": "val"}
        assert item.tags == ["tag1"]
        assert item.memory_type == "note"

    def test_stats(self):
        buf = SensoryBuffer(capacity=7, decay_ms=5000)
        buf.push("a")
        buf.push("b")
        stats = buf.stats()
        assert stats["total_items"] == 2
        assert stats["valid_items"] == 2
        assert stats["capacity"] == 7
        assert stats["decay_ms"] == 5000


# =============================================================================
# Working Memory Tests
# =============================================================================


class TestWorkingMemory:
    def test_activate_new_chunk(self):
        wm = WorkingMemory(capacity=4)
        chunk = wm.activate("k1", "content1")
        assert chunk.content == "content1"
        assert chunk.access_count == 1
        assert wm.size == 1

    def test_activate_existing_increments_access(self):
        wm = WorkingMemory(capacity=4)
        wm.activate("k1", "content1")
        chunk = wm.activate("k1", "content1")
        assert chunk.access_count == 2

    def test_capacity_eviction_lru(self):
        wm = WorkingMemory(capacity=3)
        wm.activate("k1", "c1")
        wm.activate("k2", "c2")
        wm.activate("k3", "c3")
        # k1 is LRU
        wm.activate("k4", "c4")
        assert not wm.contains("k1")
        assert wm.contains("k4")
        assert wm.size == 3

    def test_lru_order_updated_on_access(self):
        wm = WorkingMemory(capacity=3)
        wm.activate("k1", "c1")
        wm.activate("k2", "c2")
        wm.activate("k3", "c3")
        # Touch k1 to make it MRU
        wm.get("k1")
        # Now k2 should be LRU
        wm.activate("k4", "c4")
        assert wm.contains("k1")
        assert not wm.contains("k2")

    def test_get_returns_none_for_missing(self):
        wm = WorkingMemory(capacity=4)
        assert wm.get("nonexistent") is None

    def test_peek_does_not_update_access(self):
        wm = WorkingMemory(capacity=4)
        wm.activate("k1", "c1")
        chunk = wm.peek("k1")
        assert chunk is not None
        assert chunk.access_count == 1  # Not incremented by peek

    def test_stale_eviction(self, monkeypatch):
        fake_time = [1000.0]
        monkeypatch.setattr(time, "time", lambda: fake_time[0])
        wm = WorkingMemory(capacity=4, decay_minutes=0.001)  # ~60ms decay
        wm.activate("k1", "c1")
        fake_time[0] += 0.08  # Advance past decay window
        # Activate a new item to trigger stale eviction
        wm.activate("k2", "c2")
        assert not wm.contains("k1")

    def test_get_consolidation_candidates(self):
        wm = WorkingMemory(capacity=4)
        wm.activate("k1", "c1")
        wm.activate("k1", "c1")  # 2nd access
        wm.activate("k2", "c2")  # Only 1 access
        candidates = wm.get_consolidation_candidates()
        assert len(candidates) == 1
        assert candidates[0][0] == "k1"

    def test_consolidation_callback(self):
        callback = AsyncMock(return_value={"success": True, "content_hash": "abc123"})
        wm = WorkingMemory(capacity=4, consolidation_callback=callback)
        wm.activate("k1", "c1", tags=["t1"], memory_type="note", metadata={"x": 1})
        wm.activate("k1", "c1")  # 2nd access

        results = asyncio.run(wm.consolidate())
        assert len(results) == 1
        callback.assert_called_once_with("c1", ["t1"], "note", {"x": 1})

    def test_consolidation_idempotent(self):
        callback = AsyncMock(return_value={"success": True})
        wm = WorkingMemory(capacity=4, consolidation_callback=callback)
        wm.activate("k1", "c1")
        wm.activate("k1", "c1")

        asyncio.run(wm.consolidate())
        asyncio.run(wm.consolidate())
        # Should only call once (second call skips already-consolidated)
        assert callback.call_count == 1

    def test_remove(self):
        wm = WorkingMemory(capacity=4)
        wm.activate("k1", "c1")
        removed = wm.remove("k1")
        assert removed is not None
        assert removed.content == "c1"
        assert wm.size == 0

    def test_clear(self):
        wm = WorkingMemory(capacity=4)
        wm.activate("k1", "c1")
        wm.activate("k2", "c2")
        wm.clear()
        assert wm.size == 0

    def test_stats(self):
        wm = WorkingMemory(capacity=4, decay_minutes=30.0)
        wm.activate("k1", "content1")
        stats = wm.stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 4
        assert stats["decay_minutes"] == 30.0
        assert len(stats["chunks"]) == 1


# =============================================================================
# ThreeTierMemory Integration Tests
# =============================================================================


class TestThreeTierMemory:
    def test_push_sensory(self):
        ttm = ThreeTierMemory()
        item = ttm.push_sensory("raw input", tags=["test"])
        assert item.content == "raw input"
        assert ttm.sensory.size == 1

    def test_attend(self):
        ttm = ThreeTierMemory()
        chunk = ttm.attend("key1", "active context")
        assert chunk.content == "active context"
        assert ttm.working.size == 1

    def test_flush_sensory_to_working(self):
        ttm = ThreeTierMemory(sensory_decay_ms=5000)
        ttm.push_sensory("item1", tags=["t1"])
        ttm.push_sensory("item2", tags=["t2"])
        activated = ttm.flush_sensory_to_working()
        assert len(activated) == 2
        assert ttm.sensory.size == 0
        assert ttm.working.size == 2

    def test_flush_sensory_increments_existing(self):
        ttm = ThreeTierMemory(sensory_decay_ms=5000)
        # Push same content twice — should get same key
        ttm.push_sensory("duplicate content")
        ttm.flush_sensory_to_working()
        ttm.push_sensory("duplicate content")
        ttm.flush_sensory_to_working()
        # Working memory should have 1 item with access_count=2
        assert ttm.working.size == 1
        chunks = ttm.working.get_all()
        key = list(chunks.keys())[0]
        assert chunks[key].access_count == 2

    def test_consolidate_with_callback(self):
        callback = AsyncMock(return_value={"success": True})
        ttm = ThreeTierMemory(consolidation_callback=callback)
        # Access twice to meet threshold
        ttm.attend("k1", "important content", tags=["important"])
        ttm.attend("k1", "important content", tags=["important"])

        results = asyncio.run(ttm.consolidate())
        assert len(results) == 1
        callback.assert_called_once()

    def test_consolidate_without_callback(self):
        ttm = ThreeTierMemory(consolidation_callback=None)
        ttm.attend("k1", "content")
        ttm.attend("k1", "content")
        results = asyncio.run(ttm.consolidate())
        assert results == []

    def test_stats(self):
        ttm = ThreeTierMemory()
        ttm.push_sensory("sense")
        ttm.attend("k1", "work")
        stats = ttm.stats()
        assert "sensory_buffer" in stats
        assert "working_memory" in stats
        assert stats["sensory_buffer"]["total_items"] == 1
        assert stats["working_memory"]["size"] == 1

    def test_clear(self):
        ttm = ThreeTierMemory()
        ttm.push_sensory("sense")
        ttm.attend("k1", "work")
        ttm.clear()
        assert ttm.sensory.size == 0
        assert ttm.working.size == 0

    def test_full_pipeline(self):
        """End-to-end: sensory → working → consolidation candidate."""
        callback = AsyncMock(return_value={"success": True, "content_hash": "abc"})
        ttm = ThreeTierMemory(sensory_decay_ms=5000, consolidation_callback=callback)

        # Step 1: Push to sensory
        ttm.push_sensory("important fact", tags=["fact"])

        # Step 2: Flush to working
        activated = ttm.flush_sensory_to_working()
        assert len(activated) == 1
        key = activated[0][0]

        # Step 3: Access again in working (2nd touch)
        ttm.working.get(key)

        # Step 4: Consolidate to LTM
        results = asyncio.run(ttm.consolidate())
        assert len(results) == 1
        callback.assert_called_once()


# =============================================================================
# Config Integration Tests
# =============================================================================


class TestThreeTierConfig:
    def test_settings_defaults(self):
        """Verify ThreeTierSettings has sensible defaults."""
        from mcp_memory_service.config import ThreeTierSettings

        s = ThreeTierSettings()
        assert s.enabled is True
        assert s.sensory_capacity == 7
        assert s.sensory_decay_ms == 1000
        assert s.working_capacity == 4
        assert s.working_decay_minutes == 30.0
        assert s.auto_consolidate is True

    def test_settings_env_override(self, monkeypatch):
        """Verify settings can be overridden via env vars."""
        from mcp_memory_service.config import ThreeTierSettings

        monkeypatch.setenv("MCP_THREE_TIER_ENABLED", "false")
        monkeypatch.setenv("MCP_THREE_TIER_SENSORY_CAPACITY", "10")
        monkeypatch.setenv("MCP_THREE_TIER_WORKING_CAPACITY", "8")
        s = ThreeTierSettings()
        assert s.enabled is False
        assert s.sensory_capacity == 10
        assert s.working_capacity == 8
