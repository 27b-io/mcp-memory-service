"""Tests for memory lifecycle hooks."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.hooks import (
    CreateEvent,
    DeleteEvent,
    HookRegistry,
    HookValidationError,
    RetrieveEvent,
    UpdateEvent,
)
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def registry():
    return HookRegistry()


# --- HookValidationError ---


def test_hook_validation_error_is_exception():
    err = HookValidationError("content too long")
    assert str(err) == "content too long"
    assert isinstance(err, Exception)


# --- HookRegistry.add / fire_pre ---


@pytest.mark.asyncio
async def test_fire_pre_calls_handler(registry):
    called = []

    async def handler(event):
        called.append(event)

    event = CreateEvent(
        content="hello",
        content_hash="abc123",
        tags=["x"],
        memory_type=None,
        metadata={},
        client_hostname=None,
    )
    registry.add("pre_create", handler)
    await registry.fire_pre("pre_create", event)
    assert called == [event]


@pytest.mark.asyncio
async def test_fire_pre_propagates_hook_validation_error(registry):
    async def rejector(event):
        raise HookValidationError("rejected")

    registry.add("pre_create", rejector)
    event = CreateEvent(content="c", content_hash="h")
    with pytest.raises(HookValidationError, match="rejected"):
        await registry.fire_pre("pre_create", event)


@pytest.mark.asyncio
async def test_fire_pre_propagates_arbitrary_exception(registry):
    async def broken(event):
        raise ValueError("something broke")

    registry.add("pre_create", broken)
    event = CreateEvent(content="c", content_hash="h")
    with pytest.raises(ValueError, match="something broke"):
        await registry.fire_pre("pre_create", event)


@pytest.mark.asyncio
async def test_fire_pre_multiple_handlers_in_order(registry):
    order = []

    async def first(event):
        order.append("first")

    async def second(event):
        order.append("second")

    registry.add("pre_create", first)
    registry.add("pre_create", second)
    await registry.fire_pre("pre_create", CreateEvent(content="c", content_hash="h"))
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_fire_pre_no_handlers_is_noop(registry):
    # Should not raise
    await registry.fire_pre("pre_create", CreateEvent(content="c", content_hash="h"))


# --- HookRegistry.fire_post ---


@pytest.mark.asyncio
async def test_fire_post_calls_handler(registry):
    called = []

    async def handler(event):
        called.append(event)

    event = DeleteEvent(content_hash="abc")
    registry.add("post_delete", handler)
    await registry.fire_post("post_delete", event)
    assert called == [event]


@pytest.mark.asyncio
async def test_fire_post_swallows_exceptions(registry, caplog):
    async def explodes(event):
        raise RuntimeError("boom")

    registry.add("post_create", explodes)
    # Must not raise
    await registry.fire_post("post_create", CreateEvent(content="c", content_hash="h"))
    assert "boom" in caplog.text


@pytest.mark.asyncio
async def test_fire_post_no_handlers_is_noop(registry):
    await registry.fire_post("post_delete", DeleteEvent(content_hash="abc"))


# --- Event dataclasses ---


def test_create_event_fields():
    ev = CreateEvent(content="x", content_hash="h", tags=["a"], memory_type="note", metadata={"k": "v"}, client_hostname="host1")
    assert ev.content == "x"
    assert ev.tags == ["a"]
    assert ev.client_hostname == "host1"


def test_delete_event_fields():
    ev = DeleteEvent(content_hash="abc123")
    assert ev.content_hash == "abc123"


def test_update_event_fields():
    ev = UpdateEvent(content_hash="abc", fields={"tags": ["x"]})
    assert ev.fields == {"tags": ["x"]}


def test_retrieve_event_fields():
    ev = RetrieveEvent(query="what is x", result_hashes=["h1", "h2"], result_count=2)
    assert ev.result_count == 2


# --- MemoryService integration ---


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "OK")
    storage.delete.return_value = (True, "Deleted")
    return storage


def test_memory_service_accepts_hooks_param(mock_storage):
    registry = HookRegistry()
    svc = MemoryService(storage=mock_storage, hooks=registry)
    assert svc._hooks is registry


def test_memory_service_default_hooks_is_empty_registry(mock_storage):
    svc = MemoryService(storage=mock_storage)
    assert isinstance(svc._hooks, HookRegistry)


# --- store_memory hooks ---


@pytest.mark.asyncio
async def test_pre_create_hook_receives_event(mock_storage):
    registry = HookRegistry()
    events = []

    async def capture(event: CreateEvent):
        events.append(event)

    registry.add("pre_create", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.store_memory("hello world", tags=["x"], memory_type="note")

    assert len(events) == 1
    assert events[0].content == "hello world"
    assert events[0].tags == ["x"]
    assert events[0].memory_type == "note"
    assert events[0].content_hash != ""


@pytest.mark.asyncio
async def test_pre_create_validation_error_aborts_store(mock_storage):
    registry = HookRegistry()

    async def rejector(event: CreateEvent):
        raise HookValidationError("content rejected")

    registry.add("pre_create", rejector)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.store_memory("hello", tags=[])

    assert result["success"] is False
    assert "content rejected" in result["error"]
    mock_storage.store.assert_not_called()


@pytest.mark.asyncio
async def test_post_create_hook_fires_on_success(mock_storage):
    registry = HookRegistry()
    post_events = []

    async def capture(event: CreateEvent):
        post_events.append(event)

    registry.add("post_create", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.store_memory("hello", tags=[])

    assert result["success"] is True
    assert len(post_events) == 1


@pytest.mark.asyncio
async def test_post_create_hook_not_fired_on_storage_failure(mock_storage):
    mock_storage.store.return_value = (False, "disk full")
    registry = HookRegistry()
    post_events = []

    async def capture(event):
        post_events.append(event)

    registry.add("post_create", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.store_memory("hello", tags=[])

    assert result["success"] is False
    assert len(post_events) == 0


@pytest.mark.asyncio
async def test_post_create_hook_error_is_non_fatal(mock_storage):
    registry = HookRegistry()

    async def explodes(event):
        raise RuntimeError("notification failed")

    registry.add("post_create", explodes)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.store_memory("hello", tags=[])

    # Service still returns success despite hook failure
    assert result["success"] is True


# --- delete_memory hooks ---


@pytest.mark.asyncio
async def test_pre_delete_hook_receives_event(mock_storage):
    registry = HookRegistry()
    events = []

    async def capture(event: DeleteEvent):
        events.append(event)

    registry.add("pre_delete", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.delete_memory("abc123")

    assert len(events) == 1
    assert events[0].content_hash == "abc123"


@pytest.mark.asyncio
async def test_pre_delete_validation_error_aborts_delete(mock_storage):
    registry = HookRegistry()

    async def rejector(event: DeleteEvent):
        raise HookValidationError("deletion blocked")

    registry.add("pre_delete", rejector)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.delete_memory("abc123")

    assert result["success"] is False
    assert "deletion blocked" in result["error"]
    mock_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_post_delete_hook_fires_on_success(mock_storage):
    registry = HookRegistry()
    post_events = []

    async def capture(event: DeleteEvent):
        post_events.append(event)

    registry.add("post_delete", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.delete_memory("abc123")

    assert result["success"] is True
    assert len(post_events) == 1
    assert post_events[0].content_hash == "abc123"


@pytest.mark.asyncio
async def test_post_delete_hook_not_fired_on_storage_failure(mock_storage):
    mock_storage.delete.return_value = (False, "not found")
    registry = HookRegistry()
    post_events = []

    async def capture(event):
        post_events.append(event)

    registry.add("post_delete", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.delete_memory("abc123")
    assert len(post_events) == 0


# --- batch_update_memory hooks ---


@pytest.mark.asyncio
async def test_pre_update_hook_fires_per_item(mock_storage):
    mock_storage.update_memory_metadata = AsyncMock(return_value=(True, "OK"))
    registry = HookRegistry()
    events = []

    async def capture(event: UpdateEvent):
        events.append(event)

    registry.add("pre_update", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)

    updates = [
        {"content_hash": "h1", "tags": ["a"]},
        {"content_hash": "h2", "memory_type": "note"},
    ]
    await svc.batch_update_memory(updates)

    assert len(events) == 2
    assert events[0].content_hash == "h1"
    assert events[0].fields == {"tags": ["a"]}
    assert events[1].content_hash == "h2"
    assert events[1].fields == {"memory_type": "note"}


@pytest.mark.asyncio
async def test_pre_update_validation_error_skips_item(mock_storage):
    mock_storage.update_memory_metadata = AsyncMock(return_value=(True, "OK"))
    registry = HookRegistry()

    async def rejector(event: UpdateEvent):
        if event.content_hash == "bad":
            raise HookValidationError("update blocked")

    registry.add("pre_update", rejector)
    svc = MemoryService(storage=mock_storage, hooks=registry)

    updates = [
        {"content_hash": "good", "tags": ["a"]},
        {"content_hash": "bad", "tags": ["b"]},
    ]
    result = await svc.batch_update_memory(updates)

    # "good" updated, "bad" failed due to hook veto
    assert result["updated"] == 1
    assert result["failed"] == 1
    # storage.update_memory_metadata called once (for "good" only)
    assert mock_storage.update_memory_metadata.call_count == 1


@pytest.mark.asyncio
async def test_post_update_hook_fires_on_success(mock_storage):
    mock_storage.update_memory_metadata = AsyncMock(return_value=(True, "OK"))
    registry = HookRegistry()
    post_events = []

    async def capture(event: UpdateEvent):
        post_events.append(event)

    registry.add("post_update", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.batch_update_memory([{"content_hash": "h1", "tags": ["x"]}])

    assert len(post_events) == 1
    assert post_events[0].content_hash == "h1"


# --- retrieve_memories post_retrieve hook ---


@pytest.mark.asyncio
async def test_post_retrieve_hook_fires_after_retrieve(mock_storage):
    # Set up mocks needed for _retrieve_vector_only path (keywords empty → vector-only)
    mock_storage.retrieve = AsyncMock(return_value=[])
    mock_storage.count_semantic_search = AsyncMock(return_value=0)

    registry = HookRegistry()
    retrieve_events = []

    async def capture(event: RetrieveEvent):
        retrieve_events.append(event)

    registry.add("post_retrieve", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    result = await svc.retrieve_memories("hello")

    assert "memories" in result
    assert len(retrieve_events) == 1
    assert retrieve_events[0].query == "hello"
    assert retrieve_events[0].result_count >= 0


@pytest.mark.asyncio
async def test_post_retrieve_hook_error_is_non_fatal(mock_storage):
    mock_storage.retrieve = AsyncMock(return_value=[])
    mock_storage.count_semantic_search = AsyncMock(return_value=0)

    registry = HookRegistry()

    async def explodes(event):
        raise RuntimeError("webhook down")

    registry.add("post_retrieve", explodes)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    # Must not raise
    result = await svc.retrieve_memories("something")
    assert "memories" in result


# --- search_by_tag post_retrieve hook ---


@pytest.mark.asyncio
async def test_post_retrieve_fires_on_search_by_tag(mock_storage):
    from mcp_memory_service.models.memory import Memory

    mem = Memory(
        content="tagged",
        content_hash="tag_hash",
        tags=["mytag"],
        memory_type=None,
        metadata={},
        created_at=1000.0,
        updated_at=1000.0,
    )
    mock_storage.count_tag_search = AsyncMock(return_value=1)
    mock_storage.search_by_tag = AsyncMock(return_value=[mem])

    registry = HookRegistry()
    events = []

    async def capture(event: RetrieveEvent):
        events.append(event)

    registry.add("post_retrieve", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.search_by_tag("mytag")

    assert len(events) == 1
    assert events[0].query == "mytag"
    assert events[0].result_count == 1
