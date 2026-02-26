# Memory Lifecycle Hooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pre/post lifecycle hooks to `MemoryService` for create, delete, update, and retrieve operations, enabling callers to inject validation, notifications, and side-effects without modifying the service.

**Architecture:** A new `hooks.py` module defines typed event dataclasses, `HookValidationError`, and a `HookRegistry` that stores async callable lists per hook name. `MemoryService.__init__` accepts an optional `HookRegistry`; each operation fires the appropriate pre/post hooks inline. Pre-hooks propagate exceptions (including `HookValidationError`) to abort the operation. Post-hooks are non-fatal.

**Tech Stack:** Python 3.11+, `asyncio`, `dataclasses`, `typing.Literal`. No new dependencies.

---

## Task 1: Create `hooks.py` — event dataclasses, `HookValidationError`, `HookRegistry`

**Files:**
- Create: `src/mcp_memory_service/hooks.py`
- Test: `tests/unit/test_lifecycle_hooks.py`

### Step 1: Write the failing tests for HookRegistry

Create `tests/unit/test_lifecycle_hooks.py`:

```python
"""Tests for memory lifecycle hooks."""

import pytest
from mcp_memory_service.hooks import (
    CreateEvent,
    DeleteEvent,
    HookRegistry,
    HookValidationError,
    RetrieveEvent,
    UpdateEvent,
)


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
    event = CreateEvent("c", "h", [], None, {}, None)
    with pytest.raises(HookValidationError, match="rejected"):
        await registry.fire_pre("pre_create", event)


@pytest.mark.asyncio
async def test_fire_pre_propagates_arbitrary_exception(registry):
    async def broken(event):
        raise ValueError("something broke")

    registry.add("pre_create", broken)
    event = CreateEvent("c", "h", [], None, {}, None)
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
    await registry.fire_pre("pre_create", CreateEvent("c", "h", [], None, {}, None))
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_fire_pre_no_handlers_is_noop(registry):
    # Should not raise
    await registry.fire_pre("pre_create", CreateEvent("c", "h", [], None, {}, None))


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
    await registry.fire_post("post_create", CreateEvent("c", "h", [], None, {}, None))
    assert "boom" in caplog.text


@pytest.mark.asyncio
async def test_fire_post_no_handlers_is_noop(registry):
    await registry.fire_post("post_delete", DeleteEvent("abc"))


# --- Event dataclasses ---

def test_create_event_fields():
    ev = CreateEvent(
        content="x", content_hash="h", tags=["a"],
        memory_type="note", metadata={"k": "v"}, client_hostname="host1"
    )
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
```

### Step 2: Run to verify all fail

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'mcp_memory_service.hooks'`

### Step 3: Create `src/mcp_memory_service/hooks.py`

```python
"""
Memory lifecycle hooks.

Provides HookRegistry for registering async pre/post callbacks on memory operations.
Pre-hooks can veto operations by raising HookValidationError.
Post-hooks are fire-and-forget: failures are logged but never propagate.
"""

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

HookName = Literal[
    "pre_create",
    "post_create",
    "pre_delete",
    "post_delete",
    "pre_update",
    "post_update",
    "post_retrieve",
]

AsyncHookFn = Callable[[Any], Awaitable[None]]


class HookValidationError(Exception):
    """Raised by a pre-hook to reject a memory operation.

    The operation will return {"success": False, "error": str(e)} to the caller.
    """


@dataclass
class CreateEvent:
    """Context for create lifecycle hooks."""

    content: str
    content_hash: str
    tags: list[str]
    memory_type: str | None
    metadata: dict[str, Any]
    client_hostname: str | None


@dataclass
class DeleteEvent:
    """Context for delete lifecycle hooks."""

    content_hash: str


@dataclass
class UpdateEvent:
    """Context for update lifecycle hooks."""

    content_hash: str
    fields: dict[str, Any]  # keys: tags, memory_type, metadata (whichever were updated)


@dataclass
class RetrieveEvent:
    """Context for retrieve lifecycle hooks."""

    query: str
    result_hashes: list[str]
    result_count: int


class HookRegistry:
    """Registry of async lifecycle hook callbacks.

    Usage::

        registry = HookRegistry()

        async def validate(event: CreateEvent) -> None:
            if len(event.content) > 10_000:
                raise HookValidationError("content exceeds 10KB limit")

        async def notify(event: CreateEvent) -> None:
            await slack.post(f"New memory: {event.content_hash}")

        registry.add("pre_create", validate)
        registry.add("post_create", notify)

        service = MemoryService(storage, hooks=registry)
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[AsyncHookFn]] = {}

    def add(self, name: HookName, fn: AsyncHookFn) -> None:
        """Register an async hook handler.

        Args:
            name: Hook event name (e.g. "pre_create", "post_delete").
            fn: Async callable receiving the event dataclass for this hook type.
        """
        self._hooks.setdefault(name, []).append(fn)

    async def fire_pre(self, name: str, event: Any) -> None:
        """Fire all pre-hooks for *name*.

        All exceptions propagate — callers must handle HookValidationError
        to abort the operation gracefully.
        """
        for handler in self._hooks.get(name, []):
            await handler(event)

    async def fire_post(self, name: str, event: Any) -> None:
        """Fire all post-hooks for *name*.

        Exceptions are caught and logged as WARNING; they never propagate.
        """
        for handler in self._hooks.get(name, []):
            try:
                await handler(event)
            except Exception as exc:
                logger.warning("Post-hook '%s' raised (non-fatal): %s", name, exc)
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -v
```

Expected: All tests pass.

### Step 5: Commit

```bash
git add src/mcp_memory_service/hooks.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add HookRegistry, event dataclasses, HookValidationError (mm-4td3b)"
```

---

## Task 2: Wire `HookRegistry` into `MemoryService.__init__`

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:126-131`
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing test (append to `test_lifecycle_hooks.py`)

```python
# --- MemoryService integration ---

from unittest.mock import AsyncMock
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


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
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py::test_memory_service_accepts_hooks_param tests/unit/test_lifecycle_hooks.py::test_memory_service_default_hooks_is_empty_registry -v
```

Expected: `TypeError: MemoryService.__init__() got an unexpected keyword argument 'hooks'`

### Step 3: Modify `MemoryService.__init__`

In `src/mcp_memory_service/services/memory_service.py`:

**Add import** (after existing imports, line ~26 area):
```python
from ..hooks import HookRegistry
```

**Change `__init__` signature** (line 126):
```python
    def __init__(
        self,
        storage: MemoryStorage,
        graph_client: GraphClient | None = None,
        write_queue: HebbianWriteQueue | None = None,
        hooks: HookRegistry | None = None,
    ):
```

**Add to `__init__` body** (after line 138, after `self._init_three_tier()`):
```python
        self._hooks = hooks if hooks is not None else HookRegistry()
```

### Step 4: Run tests

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -v
```

Expected: All tests pass.

### Step 5: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: wire HookRegistry into MemoryService (mm-4td3b)"
```

---

## Task 3: Instrument `store_memory` with pre/post create hooks

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (around line 1340–1495)
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing tests (append to `test_lifecycle_hooks.py`)

```python
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
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "create" -v
```

Expected: FAIL — hooks not wired yet.

### Step 3: Add hooks to `store_memory`

Locate the section in `store_memory` where `content_hash` is generated and `final_summary` is assigned (around line 1361). Add the pre-hook **after** `final_summary` is set (line 1361) and **before** the chunked-path check (line 1363):

```python
            # Fire pre-create hook (can raise HookValidationError to abort)
            try:
                await self._hooks.fire_pre(
                    "pre_create",
                    CreateEvent(
                        content=content,
                        content_hash=content_hash,
                        tags=list(final_tags),
                        memory_type=memory_type,
                        metadata=dict(final_metadata),
                        client_hostname=client_hostname,
                    ),
                )
            except HookValidationError as e:
                return {"success": False, "error": str(e)}
```

Also add the import at top of file:
```python
from ..hooks import CreateEvent, DeleteEvent, HookRegistry, HookValidationError, RetrieveEvent, UpdateEvent
```

For the **post-create hook**, find the success path in the non-chunked branch (after `result = {"success": True, "memory": ...}` at ~line 1477, before contradiction detection). Add:

```python
                    # Fire post-create hook (non-fatal)
                    _post_event = CreateEvent(
                        content=content,
                        content_hash=content_hash,
                        tags=list(final_tags),
                        memory_type=memory_type,
                        metadata=dict(final_metadata),
                        client_hostname=client_hostname,
                    )
                    await self._hooks.fire_post("post_create", _post_event)
```

> **Note:** The chunked path stores multiple chunks. The pre-hook fires once for the entire call (full content, root hash). The post-hook fires once after all chunks succeed. Add similar post-hook fire after the chunked path returns its result.

### Step 4: Run tests

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "create" -v
```

Expected: All pass.

### Step 5: Run full unit test suite to check for regressions

```bash
uv run pytest tests/unit/ -x -q
```

Expected: All pass.

### Step 6: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add pre/post create hooks to store_memory (mm-4td3b)"
```

---

## Task 4: Instrument `delete_memory` with pre/post delete hooks

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (around line 1976–2025)
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing tests (append to `test_lifecycle_hooks.py`)

```python
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
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "delete" -v
```

Expected: FAIL.

### Step 3: Add hooks to `delete_memory`

At line 1986, **before** `success, message = await self.storage.delete(content_hash)`:

```python
        try:
            # Fire pre-delete hook (can raise HookValidationError to abort)
            try:
                await self._hooks.fire_pre("pre_delete", DeleteEvent(content_hash=content_hash))
            except HookValidationError as e:
                return {"success": False, "content_hash": content_hash, "error": str(e)}

            success, message = await self.storage.delete(content_hash)
```

At line 2003, **after** `return {"success": True, "content_hash": content_hash}` is built but before returning:

```python
                # Fire post-delete hook (non-fatal)
                await self._hooks.fire_post("post_delete", DeleteEvent(content_hash=content_hash))
                return {"success": True, "content_hash": content_hash}
```

### Step 4: Run tests

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "delete" -v
uv run pytest tests/unit/ -x -q
```

Expected: All pass.

### Step 5: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add pre/post delete hooks to delete_memory (mm-4td3b)"
```

---

## Task 5: Instrument `batch_update_memory` with pre/post update hooks

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (around line 2680–2718)
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing tests (append to `test_lifecycle_hooks.py`)

```python
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
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "update" -v
```

Expected: FAIL.

### Step 3: Add hooks to `batch_update_memory`

The per-item loop in `batch_update_memory` (around line 2683) currently calls `storage.update_memory_metadata`. Wrap each item to add hook firing:

In the loop body (around line 2699), replace:
```python
            try:
                success, message = await self.storage.update_memory_metadata(...)
```

With:
```python
            try:
                # Fire pre-update hook; HookValidationError counts as failure for this item
                try:
                    await self._hooks.fire_pre("pre_update", UpdateEvent(content_hash=hash_, fields=update_fields))
                except HookValidationError as e:
                    results.append({"index": i, "success": False, "content_hash": hash_, "error": str(e)})
                    continue

                success, message = await self.storage.update_memory_metadata(
                    content_hash=hash_,
                    updates=update_fields,
                    preserve_timestamps=True,
                )
                if success:
                    updated_count += 1
                    results.append({"index": i, "success": True, "content_hash": hash_})
                    # Fire post-update hook (non-fatal)
                    await self._hooks.fire_post("post_update", UpdateEvent(content_hash=hash_, fields=update_fields))
                else:
                    results.append({"index": i, "success": False, "content_hash": hash_, "error": message})
```

### Step 4: Run tests

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "update" -v
uv run pytest tests/unit/ -x -q
```

Expected: All pass.

### Step 5: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add pre/post update hooks to batch_update_memory (mm-4td3b)"
```

---

## Task 6: Add `post_retrieve` hook to `retrieve_memories`

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (around line 1835–1845)
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing tests (append to `test_lifecycle_hooks.py`)

```python
# --- retrieve_memories post_retrieve hook ---

@pytest.mark.asyncio
async def test_post_retrieve_hook_fires_after_retrieve(mock_storage):
    from mcp_memory_service.models.memory import Memory, MemoryQueryResult

    mem = Memory(
        content="hello",
        content_hash="h1",
        tags=[],
        memory_type=None,
        metadata={},
        created_at=1000.0,
        updated_at=1000.0,
    )
    mock_storage.search.return_value = [MemoryQueryResult(memory=mem, score=0.9)]
    mock_storage.count.return_value = 1
    mock_storage.get_all_tags.return_value = []

    registry = HookRegistry()
    retrieve_events = []

    async def capture(event: RetrieveEvent):
        retrieve_events.append(event)

    registry.add("post_retrieve", capture)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    await svc.retrieve_memories("hello")

    assert len(retrieve_events) == 1
    assert retrieve_events[0].query == "hello"
    assert retrieve_events[0].result_count >= 0


@pytest.mark.asyncio
async def test_post_retrieve_hook_error_is_non_fatal(mock_storage):
    mock_storage.search.return_value = []
    mock_storage.count.return_value = 0
    mock_storage.get_all_tags.return_value = []

    registry = HookRegistry()

    async def explodes(event):
        raise RuntimeError("webhook down")

    registry.add("post_retrieve", explodes)
    svc = MemoryService(storage=mock_storage, hooks=registry)
    # Must not raise
    result = await svc.retrieve_memories("something")
    assert "memories" in result
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "retrieve" -v
```

Expected: FAIL — `post_retrieve` not yet wired.

### Step 3: Add `post_retrieve` to `retrieve_memories`

Locate the success return in `retrieve_memories` (around line 1835–1845). Before the `return response` statement, add:

```python
            # Fire post-retrieve hook (non-fatal)
            await self._hooks.fire_post(
                "post_retrieve",
                RetrieveEvent(
                    query=query,
                    result_hashes=[r["content_hash"] for r in results if "content_hash" in r],
                    result_count=len(results),
                ),
            )
            return response
```

> The `results` list is the formatted memory list assembled before building `response`. Confirm the variable name by checking that `results` is defined before `response` in that branch.

### Step 4: Run tests

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -k "retrieve" -v
uv run pytest tests/unit/ -x -q
```

Expected: All pass.

### Step 5: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add post_retrieve hook to retrieve_memories (mm-4td3b)"
```

---

## Task 7: Add `post_retrieve` hook to `search_by_tag`

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (around line 1938–1943)
- Test: `tests/unit/test_lifecycle_hooks.py` (append)

### Step 1: Write the failing tests (append to `test_lifecycle_hooks.py`)

```python
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
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py::test_post_retrieve_fires_on_search_by_tag -v
```

Expected: FAIL.

### Step 3: Add hook to `search_by_tag`

In `search_by_tag`, before the `return {...}` dict (around line 1938), add:

```python
            # Fire post-retrieve hook (non-fatal)
            tag_query = ", ".join(tags) if isinstance(tags, list) else tags
            await self._hooks.fire_post(
                "post_retrieve",
                RetrieveEvent(
                    query=tag_query,
                    result_hashes=[r["content_hash"] for r in results if "content_hash" in r],
                    result_count=len(results),
                ),
            )
            return {
                "memories": results,
                ...
            }
```

### Step 4: Run all lifecycle hook tests + full unit suite

```bash
uv run pytest tests/unit/test_lifecycle_hooks.py -v
uv run pytest tests/unit/ -x -q
```

Expected: All pass.

### Step 5: Run linter

```bash
uv run ruff check src/mcp_memory_service/hooks.py src/mcp_memory_service/services/memory_service.py
uv run ruff format --check src/mcp_memory_service/hooks.py src/mcp_memory_service/services/memory_service.py
```

Fix any issues, then commit.

### Step 6: Commit

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
git commit -m "feat: add post_retrieve hook to search_by_tag (mm-4td3b)"
```

---

## Task 8: Final validation + push

### Step 1: Run full test suite

```bash
uv run pytest tests/unit/ tests/integration/ -x -q --timeout=60 2>&1 | tail -20
```

Expected: All tests pass (or pre-existing integration failures only).

### Step 2: Run linter on all modified files

```bash
uv run ruff check src/mcp_memory_service/hooks.py src/mcp_memory_service/services/memory_service.py tests/unit/test_lifecycle_hooks.py
```

### Step 3: Push and signal completion

```bash
git push
gt done
```

---

## Acceptance Criteria

- `HookRegistry` with `add`, `fire_pre`, `fire_post`
- `HookValidationError` aborts `store_memory`, `delete_memory`, per-item in `batch_update_memory`
- `post_retrieve` fires on `retrieve_memories` and `search_by_tag`
- Post-hook failures are non-fatal (logged only)
- `MemoryService(storage)` works without `hooks` arg (backward compatible)
- All tests in `tests/unit/test_lifecycle_hooks.py` pass
- Ruff clean
