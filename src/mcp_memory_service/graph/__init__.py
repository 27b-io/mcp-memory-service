"""
Graph layer for MCP Memory Service.

Provides FalkorDB-backed knowledge graph with CQRS write pattern:
- Concurrent reads from all service instances
- Hebbian writes queued via Redis LPUSH to single consumer
"""

from .client import GraphClient
from .queue import HebbianWriteQueue

__all__ = [
    "GraphClient",
    "HebbianWriteQueue",
]
