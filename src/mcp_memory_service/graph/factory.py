"""
Factory for creating and initializing the graph layer.

Creates GraphClient + HebbianWriteQueue from FalkorDBSettings config.
Returns None if graph layer is disabled (MCP_FALKORDB_ENABLED=false).
"""

import logging

from ..config import settings
from .client import GraphClient
from .queue import HebbianWriteQueue

logger = logging.getLogger(__name__)


async def create_graph_layer() -> tuple[GraphClient, HebbianWriteQueue] | None:
    """
    Create and initialize the FalkorDB graph layer if enabled.

    Returns:
        Tuple of (GraphClient, HebbianWriteQueue) if enabled, None otherwise.
    """
    config = settings.falkordb

    if not config.enabled:
        logger.info("FalkorDB graph layer disabled (MCP_FALKORDB_ENABLED=false)")
        return None

    password = config.password.get_secret_value() if config.password else None

    client = GraphClient(
        host=config.host,
        port=config.port,
        password=password,
        graph_name=config.graph_name,
        max_connections=config.max_connections,
    )

    await client.initialize()

    queue = HebbianWriteQueue(
        pool=client.pool,
        graph=client.graph,
        queue_key=config.write_queue_key,
        batch_size=config.write_queue_batch_size,
        poll_interval=config.write_queue_poll_interval,
        initial_weight=config.hebbian_initial_weight,
        strengthen_rate=config.hebbian_strengthen_rate,
        max_weight=config.hebbian_max_weight,
    )

    logger.info(f"Graph layer initialized: {config.host}:{config.port}/{config.graph_name}")
    return client, queue
