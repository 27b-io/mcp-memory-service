# Memory Lifecycle Hooks — Design Document

**Date**: 2026-02-26
**Bead**: mm-4td3b
**Status**: Approved

## Problem

The `MemoryService` provides no extension points. Consumers wanting to add validation (reject a store if content is too long), notifications (Slack on delete), or side-effects (sync to external system on update) must fork the service or wrap every method call. That's fragile and DRY-violating.

## Solution

Pre- and post-hooks for the four memory lifecycle events: create, delete, update, retrieve. Pre-hooks can veto operations by raising `HookValidationError`. Post-hooks observe without blocking.

## Architecture

### New file: `src/mcp_memory_service/hooks.py`

Three pieces:

**1. `HookValidationError`** — exception raised by pre-hooks to reject an operation.

**2. Event dataclasses** — typed context objects passed to every hook handler:

| Class | Fields |
|---|---|
| `CreateEvent` | `content`, `content_hash`, `tags`, `memory_type`, `metadata`, `client_hostname` |
| `DeleteEvent` | `content_hash` |
| `UpdateEvent` | `content_hash`, `fields` (dict of what changed) |
| `RetrieveEvent` | `query`, `result_hashes`, `result_count` |

**3. `HookRegistry`** — registers and fires handlers:

```python
HookName = Literal[
    "pre_create", "post_create",
    "pre_delete", "post_delete",
    "pre_update", "post_update",
    "post_retrieve",
]

class HookRegistry:
    def add(self, name: HookName, fn: Callable[[Event], Awaitable[None]]) -> None: ...
    async def fire_pre(self, name: str, event: Any) -> None: ...   # propagates all exceptions
    async def fire_post(self, name: str, event: Any) -> None: ...  # logs, never propagates
```

### Modified: `src/mcp_memory_service/services/memory_service.py`

`MemoryService.__init__` gains `hooks: HookRegistry | None = None` (backward-compatible default `None` → treated as empty registry).

### Integration points

| Method | Pre-hook | Post-hook |
|---|---|---|
| `store_memory` | `pre_create` (before `storage.store`) | `post_create` (after success) |
| `delete_memory` | `pre_delete` (before `storage.delete`) | `post_delete` (after success) |
| `batch_update_memory` | `pre_update` per item | `post_update` per item after success |
| `retrieve_memories` | — | `post_retrieve` (after results assembled) |
| `search_by_tag` | — | `post_retrieve` (after results assembled) |

### Error handling

- **Pre-hooks**: `HookValidationError` → operation returns `{"success": False, "error": str(e)}`. All other exceptions propagate to the caller's exception handler.
- **Post-hooks**: all exceptions caught and logged as `WARNING`, never surface.

## Testing

New file: `tests/unit/test_lifecycle_hooks.py`

Covers:
- `HookRegistry.add` / `fire_pre` / `fire_post`
- `HookValidationError` aborts `store_memory`, `delete_memory`, `batch_update_memory`
- Post-hook errors are non-fatal
- Multiple handlers fire in registration order
- `post_retrieve` fires on `retrieve_memories` and `search_by_tag`
- No hooks registered → service operates identically to before

## Out of Scope (YAGNI)

- `batch_store_memory`, `batch_delete_memory` hooks (can be added later)
- Hook removal / deregistration
- Hook priorities / ordering control
- Synchronous hooks
