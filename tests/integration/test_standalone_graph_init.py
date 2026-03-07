"""Test that standalone init paths properly initialize graph layer.

Verifies the fix for #151: standalone MCP and HTTP modes previously called
create_storage_instance() directly, bypassing StorageManager.get_storage()
which is the only path that initializes the FalkorDB graph layer.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import src.mcp_memory_service.shared_storage as _shared_mod
from src.mcp_memory_service.shared_storage import (
    StorageManager,
    close_shared_storage,
    get_graph_client,
    get_shared_storage,
    get_write_queue,
)

# All integration tests get 600s timeout (10 minutes for slow CI/startup)
pytestmark = pytest.mark.timeout(600)


def _reset_manager():
    """Reset both the singleton class var AND the module-level cached _manager."""
    StorageManager._instance = None
    _shared_mod._manager = StorageManager.get_instance()


@pytest.fixture(autouse=True)
async def _reset_storage_manager():
    """Reset StorageManager singleton before each test, clean up after."""
    _reset_manager()
    yield
    try:
        await close_shared_storage()
    except Exception:
        pass
    _reset_manager()


@pytest.mark.asyncio
async def test_standalone_init_initializes_graph():
    """get_shared_storage() initializes graph layer when FalkorDB is enabled.

    This tests the exact integration point that was broken in #151:
    standalone MCP/HTTP mode must go through StorageManager to get
    both storage AND graph client initialized.
    """
    mock_storage = AsyncMock()
    mock_storage.close = AsyncMock()

    mock_graph_client = MagicMock()
    mock_graph_client.close = AsyncMock()
    mock_write_queue = MagicMock()
    mock_write_queue.start_consumer = AsyncMock()
    mock_write_queue.stop_consumer = AsyncMock()

    with (
        patch(
            "src.mcp_memory_service.shared_storage.create_storage_instance",
            return_value=mock_storage,
        ),
        patch(
            "src.mcp_memory_service.shared_storage.create_graph_layer",
            return_value=(mock_graph_client, mock_write_queue),
        ),
    ):
        storage = await get_shared_storage()

        assert storage is mock_storage
        assert get_graph_client() is mock_graph_client
        assert get_write_queue() is mock_write_queue
        mock_write_queue.start_consumer.assert_called_once()


@pytest.mark.asyncio
async def test_standalone_init_graph_disabled():
    """Graph client is None when FalkorDB disabled — not an error."""
    mock_storage = AsyncMock()
    mock_storage.close = AsyncMock()

    with (
        patch(
            "src.mcp_memory_service.shared_storage.create_storage_instance",
            return_value=mock_storage,
        ),
        patch(
            "src.mcp_memory_service.shared_storage.create_graph_layer",
            return_value=None,
        ),
    ):
        storage = await get_shared_storage()

        assert storage is mock_storage
        assert get_graph_client() is None
        assert get_write_queue() is None


@pytest.mark.asyncio
async def test_standalone_init_idempotent():
    """Multiple calls to get_shared_storage() return same instance.

    Verifies the idempotency contract — whether called by unified_server
    first or by MCP lifespan first, subsequent calls return the cached
    instance without re-initializing.
    """
    mock_storage = AsyncMock()
    mock_storage.close = AsyncMock()

    mock_graph_client = MagicMock()
    mock_graph_client.close = AsyncMock()
    mock_write_queue = MagicMock()
    mock_write_queue.start_consumer = AsyncMock()
    mock_write_queue.stop_consumer = AsyncMock()

    with (
        patch(
            "src.mcp_memory_service.shared_storage.create_storage_instance",
            return_value=mock_storage,
        ) as mock_create,
        patch(
            "src.mcp_memory_service.shared_storage.create_graph_layer",
            return_value=(mock_graph_client, mock_write_queue),
        ) as mock_graph,
    ):
        # Simulate: unified_server calls first, then MCP lifespan calls
        storage1 = await get_shared_storage()
        storage2 = await get_shared_storage()

        assert storage1 is storage2
        mock_create.assert_called_once()
        mock_graph.assert_called_once()


@pytest.mark.asyncio
async def test_standalone_init_concurrent_safety():
    """Concurrent calls to get_shared_storage() only initialize once.

    Simulates the race condition where MCP lifespan and HTTP lifespan
    both call get_shared_storage() before either completes initialization.
    """
    call_count = 0

    async def slow_create_storage():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)  # Simulate slow init
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    with (
        patch(
            "src.mcp_memory_service.shared_storage.create_storage_instance",
            side_effect=slow_create_storage,
        ),
        patch(
            "src.mcp_memory_service.shared_storage.create_graph_layer",
            return_value=None,
        ),
    ):
        results = await asyncio.gather(*[get_shared_storage() for _ in range(10)])

        assert call_count == 1, f"Storage initialized {call_count} times, expected 1"
        assert all(r is results[0] for r in results)
