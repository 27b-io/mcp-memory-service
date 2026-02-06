"""
Graph layer for MCP Memory Service.

Provides FalkorDB-backed knowledge graph with CQRS write pattern:
- Concurrent reads from all service instances
- Hebbian writes queued via Redis LPUSH to single consumer
- Typed relationship edges (RELATES_TO, PRECEDES, CONTRADICTS) via direct writes
"""

from .client import GraphClient
from .queue import HebbianWriteQueue
from .schema import RELATION_TYPES

__all__ = [
    "GraphClient",
    "HebbianWriteQueue",
    "RELATION_TYPES",
]
