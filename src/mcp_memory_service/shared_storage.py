#!/usr/bin/env python3
"""
Shared storage manager for MCP Memory Service.

This module provides a singleton storage instance that can be shared between
HTTP and MCP servers, preventing duplicate model loading and initialization.

The storage is initialized once and reused by both servers, saving ~500MB RAM
per additional server instance and avoiding race conditions.
"""

import asyncio
import logging
from threading import Lock
from typing import Optional

from .graph.client import GraphClient
from .graph.factory import create_graph_layer
from .graph.queue import HebbianWriteQueue
from .storage.base import MemoryStorage
from .storage.factory import create_storage_instance

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages a singleton storage instance for shared access."""

    _instance: Optional["StorageManager"] = None
    _lock: Lock = Lock()

    def __init__(self):
        """Initialize storage manager."""
        self._storage: MemoryStorage | None = None
        self._graph_client: GraphClient | None = None
        self._write_queue: HebbianWriteQueue | None = None
        self._initialization_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    @classmethod
    def get_instance(cls) -> "StorageManager":
        """Get singleton instance of StorageManager.

        Thread-safe singleton pattern ensures only one instance exists.

        Returns:
            StorageManager: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Created new StorageManager singleton instance")
        return cls._instance

    async def get_storage(self) -> MemoryStorage:
        """Get or create the shared storage instance.

        This method is idempotent and thread-safe. Multiple concurrent calls
        will result in only one storage initialization.

        Returns:
            MemoryStorage: The shared storage instance
        """
        # Fast path - already initialized
        if self._initialized and self._storage is not None:
            return self._storage

        # Slow path - need to initialize
        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._initialized and self._storage is not None:
                return self._storage

            logger.info("Initializing shared storage instance...")

            # Create storage using factory
            self._storage = await create_storage_instance()

            # Initialize graph layer if enabled
            try:
                graph_result = await create_graph_layer()
                if graph_result is not None:
                    self._graph_client, self._write_queue = graph_result
                    await self._write_queue.start_consumer()
                    logger.info("Graph layer initialized with Hebbian write consumer")
            except Exception as e:
                logger.warning(f"Graph layer initialization failed (non-fatal): {e}")
                self._graph_client = None
                self._write_queue = None

            self._initialized = True

            logger.info(f"Shared storage initialized successfully: {type(self._storage).__name__}")

            return self._storage

    @property
    def graph_client(self) -> GraphClient | None:
        """Get the graph client if graph layer is enabled."""
        return self._graph_client

    @property
    def write_queue(self) -> HebbianWriteQueue | None:
        """Get the Hebbian write queue if graph layer is enabled."""
        return self._write_queue

    async def close(self) -> None:
        """Close all managed instances.

        Safe to call even if storage was never initialized.
        """
        # Close write queue consumer first
        if self._write_queue is not None:
            try:
                await self._write_queue.stop_consumer()
            except Exception as e:
                logger.warning(f"Error stopping write queue consumer: {e}")
            self._write_queue = None

        # Close graph client
        if self._graph_client is not None:
            try:
                await self._graph_client.close()
            except Exception as e:
                logger.warning(f"Error closing graph client: {e}")
            self._graph_client = None

        # Close storage
        if self._storage is not None:
            try:
                logger.info("Closing shared storage instance...")
                await self._storage.close()
                self._storage = None
                self._initialized = False
                logger.info("Shared storage closed successfully")
            except Exception as e:
                logger.error(f"Error closing shared storage: {e}")

    def is_initialized(self) -> bool:
        """Check if storage has been initialized.

        Returns:
            bool: True if storage is initialized, False otherwise
        """
        return self._initialized and self._storage is not None


# Module-level convenience functions
_manager = StorageManager.get_instance()


async def get_shared_storage() -> MemoryStorage:
    """Get the shared storage instance.

    Convenience function that uses the singleton StorageManager.

    Returns:
        MemoryStorage: The shared storage instance
    """
    return await _manager.get_storage()


async def close_shared_storage() -> None:
    """Close the shared storage instance.

    Convenience function that uses the singleton StorageManager.
    """
    await _manager.close()


def is_storage_initialized() -> bool:
    """Check if shared storage has been initialized.

    Convenience function that uses the singleton StorageManager.

    Returns:
        bool: True if storage is initialized, False otherwise
    """
    return _manager.is_initialized()


def get_graph_client() -> GraphClient | None:
    """Get the shared graph client if graph layer is enabled."""
    return _manager.graph_client


def get_write_queue() -> HebbianWriteQueue | None:
    """Get the shared Hebbian write queue if graph layer is enabled."""
    return _manager.write_queue
