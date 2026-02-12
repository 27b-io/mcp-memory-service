"""
Three-tier memory model based on Cowan's embedded-processes model.

Implements session-scoped, in-process memory tiers:
- Sensory Buffer: Ring buffer (~7 items, 1s TTL) for raw input capture
- Working Memory: LRU cache (~4 chunks, 30min decay) for active task context
- Long-Term Memory: Existing Qdrant storage (not managed here)

Tier transitions:
- Sensory → Working: attention (items accessed/referenced promote)
- Working → LTM: consolidation (items accessed 2+ times in working memory)

These tiers are ephemeral and per-session. They do NOT touch persistent storage.
Only the consolidation callback writes to Qdrant via the existing store_memory path.

Reference: Cowan (2001), "The magical number 4 in short-term memory"
Reference: arxiv:2512.23343, "AI Meets Brain: Memory Systems"
"""

import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SensoryItem:
    """A single item in the sensory buffer."""

    content: str
    metadata: dict[str, Any]
    timestamp: float = field(default_factory=lambda: time.time())
    tags: list[str] = field(default_factory=list)
    memory_type: str | None = None


@dataclass
class WorkingChunk:
    """A chunk held in working memory."""

    content: str
    metadata: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    memory_type: str | None = None
    created_at: float = field(default_factory=lambda: time.time())
    last_access: float = field(default_factory=lambda: time.time())
    access_count: int = 1


# =============================================================================
# Sensory Buffer
# =============================================================================


class SensoryBuffer:
    """Ring buffer for raw input capture before processing.

    Items expire after `decay_ms` and are evicted when capacity is reached
    (oldest-first). This prevents flooding long-term memory with noise.

    Capacity defaults to 7 (Miller's magic number ±2).
    """

    def __init__(self, capacity: int = 7, decay_ms: int = 1000):
        self._capacity = capacity
        self._decay_ms = decay_ms
        self._items: deque[SensoryItem] = deque(maxlen=capacity)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> SensoryItem:
        """Add an item to the buffer. Oldest item is evicted if at capacity."""
        item = SensoryItem(
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            memory_type=memory_type,
        )
        self._items.append(item)  # deque(maxlen=N) auto-evicts oldest
        return item

    def get_valid(self) -> list[SensoryItem]:
        """Return non-expired items without clearing the buffer."""
        now = time.time()
        cutoff = now - (self._decay_ms / 1000.0)
        return [item for item in self._items if item.timestamp >= cutoff]

    def flush(self) -> list[SensoryItem]:
        """Remove and return all non-expired items. Clears the buffer."""
        valid = self.get_valid()
        self._items.clear()
        return valid

    def clear(self) -> None:
        """Discard all items."""
        self._items.clear()

    def stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        valid = self.get_valid()
        return {
            "total_items": len(self._items),
            "valid_items": len(valid),
            "capacity": self._capacity,
            "decay_ms": self._decay_ms,
        }


# =============================================================================
# Working Memory
# =============================================================================

# Type alias for the consolidation callback
ConsolidationCallback = Callable[
    [str, list[str] | None, str | None, dict[str, Any] | None],
    Coroutine[Any, Any, dict[str, Any]],
]


class WorkingMemory:
    """LRU cache for active task context.

    Holds the ~4 most relevant pieces for the current task (Cowan's limit).
    Items are keyed by a unique identifier (content hash or generated key).

    Eviction: LRU (least recently used) when capacity is exceeded.
    Decay: Items not accessed within `decay_minutes` are eligible for eviction.

    Consolidation: Items accessed 2+ times are candidates for long-term storage.
    A callback is invoked to store them via the existing memory service pipeline.
    """

    CONSOLIDATION_ACCESS_THRESHOLD = 2

    def __init__(
        self,
        capacity: int = 4,
        decay_minutes: float = 30.0,
        consolidation_callback: ConsolidationCallback | None = None,
    ):
        self._capacity = capacity
        self._decay_seconds = decay_minutes * 60.0
        self._chunks: OrderedDict[str, WorkingChunk] = OrderedDict()
        self._consolidation_callback = consolidation_callback
        self._consolidated_keys: set[str] = set()  # Track what's already been sent to LTM

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def capacity(self) -> int:
        return self._capacity

    def activate(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> WorkingChunk:
        """Bring a memory into (or refresh in) the active working set.

        If the key already exists, its access count and timestamp are updated.
        If at capacity, the least-recently-used item is evicted.

        Returns the working chunk (new or existing).
        """
        if key in self._chunks:
            chunk = self._chunks[key]
            chunk.access_count += 1
            chunk.last_access = time.time()
            # Move to end (most recently used)
            self._chunks.move_to_end(key)
            return chunk

        # Evict stale items first, then LRU if still at capacity
        self._evict_stale()
        while len(self._chunks) >= self._capacity:
            evicted_key, _ = self._chunks.popitem(last=False)
            logger.debug(f"Working memory evicted LRU: {evicted_key[:12]}")

        chunk = WorkingChunk(
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            memory_type=memory_type,
        )
        self._chunks[key] = chunk
        return chunk

    def get(self, key: str) -> WorkingChunk | None:
        """Get a chunk by key, updating access count and LRU position."""
        if key not in self._chunks:
            return None
        chunk = self._chunks[key]
        chunk.access_count += 1
        chunk.last_access = time.time()
        self._chunks.move_to_end(key)
        return chunk

    def peek(self, key: str) -> WorkingChunk | None:
        """Get a chunk without updating access count or LRU position."""
        return self._chunks.get(key)

    def contains(self, key: str) -> bool:
        return key in self._chunks

    def get_all(self) -> dict[str, WorkingChunk]:
        """Return all chunks (for inspection, not modification)."""
        return dict(self._chunks)

    def get_consolidation_candidates(self) -> list[tuple[str, WorkingChunk]]:
        """Return chunks accessed enough times to be consolidated to LTM.

        Only returns items not yet consolidated in this session.
        """
        return [
            (key, chunk)
            for key, chunk in self._chunks.items()
            if chunk.access_count >= self.CONSOLIDATION_ACCESS_THRESHOLD and key not in self._consolidated_keys
        ]

    async def consolidate(self) -> list[dict[str, Any]]:
        """Consolidate eligible chunks to long-term memory via callback.

        Returns results from the consolidation callback for each consolidated item.
        """
        if self._consolidation_callback is None:
            return []

        candidates = self.get_consolidation_candidates()
        results = []

        for key, chunk in candidates:
            try:
                result = await self._consolidation_callback(
                    chunk.content,
                    chunk.tags,
                    chunk.memory_type,
                    chunk.metadata,
                )
                if result.get("success", False):
                    self._consolidated_keys.add(key)
                    results.append(result)
                    logger.info(f"Consolidated working memory chunk to LTM: {key[:12]} (accesses={chunk.access_count})")
                else:
                    logger.warning(f"Consolidation callback reported failure for chunk {key[:12]}: {result}")
            except Exception as e:
                logger.warning(f"Failed to consolidate chunk {key[:12]}: {e}")

        return results

    def remove(self, key: str) -> WorkingChunk | None:
        """Remove and return a specific chunk."""
        return self._chunks.pop(key, None)

    def clear(self) -> None:
        """Clear all chunks and consolidation tracking."""
        self._chunks.clear()
        self._consolidated_keys.clear()

    def _evict_stale(self) -> int:
        """Remove chunks that haven't been accessed within the decay window."""
        cutoff = time.time() - self._decay_seconds
        stale_keys = [key for key, chunk in self._chunks.items() if chunk.last_access < cutoff]
        for key in stale_keys:
            del self._chunks[key]
            logger.debug(f"Working memory evicted stale: {key[:12]}")
        return len(stale_keys)

    def stats(self) -> dict[str, Any]:
        """Return working memory statistics."""
        now = time.time()
        chunks_info = []
        for key, chunk in self._chunks.items():
            chunks_info.append(
                {
                    "key": key[:12] + "..." if len(key) > 12 else key,
                    "access_count": chunk.access_count,
                    "age_seconds": round(now - chunk.created_at, 1),
                    "idle_seconds": round(now - chunk.last_access, 1),
                    "consolidated": key in self._consolidated_keys,
                }
            )
        return {
            "size": len(self._chunks),
            "capacity": self._capacity,
            "decay_minutes": self._decay_seconds / 60.0,
            "consolidated_count": len(self._consolidated_keys),
            "chunks": chunks_info,
        }


# =============================================================================
# Three-Tier Manager
# =============================================================================


class ThreeTierMemory:
    """Orchestrates the three-tier memory model.

    Provides a unified interface for:
    1. Pushing raw input to the sensory buffer
    2. Promoting attended items to working memory
    3. Consolidating rehearsed items to long-term memory

    Session-scoped: create one per MCP session. Not shared across clients.
    """

    def __init__(
        self,
        sensory_capacity: int = 7,
        sensory_decay_ms: int = 1000,
        working_capacity: int = 4,
        working_decay_minutes: float = 30.0,
        consolidation_callback: ConsolidationCallback | None = None,
    ):
        self.sensory = SensoryBuffer(
            capacity=sensory_capacity,
            decay_ms=sensory_decay_ms,
        )
        self.working = WorkingMemory(
            capacity=working_capacity,
            decay_minutes=working_decay_minutes,
            consolidation_callback=consolidation_callback,
        )

    def push_sensory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> SensoryItem:
        """Push raw input into the sensory buffer."""
        return self.sensory.push(content, metadata, tags, memory_type)

    def attend(self, key: str, content: str, **kwargs: Any) -> WorkingChunk:
        """Promote an item to working memory (attention gate).

        This is the sensory → working transition. Called when the system
        determines an item is worth holding in active context.
        """
        return self.working.activate(key, content, **kwargs)

    def flush_sensory_to_working(self) -> list[tuple[str, WorkingChunk]]:
        """Flush valid sensory items into working memory.

        Each sensory item is promoted using its content hash as key.
        Returns list of (key, chunk) pairs that were activated.
        """
        from .utils.hashing import generate_content_hash

        items = self.sensory.flush()
        activated = []
        for item in items:
            key = generate_content_hash(item.content)
            chunk = self.working.activate(
                key=key,
                content=item.content,
                metadata=item.metadata,
                tags=item.tags,
                memory_type=item.memory_type,
            )
            activated.append((key, chunk))
        return activated

    async def consolidate(self) -> list[dict[str, Any]]:
        """Consolidate working memory items to LTM.

        Items accessed 2+ times in working memory are promoted to
        long-term storage via the consolidation callback.
        """
        return await self.working.consolidate()

    def stats(self) -> dict[str, Any]:
        """Return combined tier statistics."""
        return {
            "sensory_buffer": self.sensory.stats(),
            "working_memory": self.working.stats(),
        }

    def clear(self) -> None:
        """Clear all tiers (session reset)."""
        self.sensory.clear()
        self.working.clear()
