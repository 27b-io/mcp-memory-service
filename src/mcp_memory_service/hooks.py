"""
Memory lifecycle hooks.

Provides HookRegistry for registering async pre/post callbacks on memory operations.
Pre-hooks can veto operations by raising HookValidationError.
Post-hooks are fire-and-forget: failures are logged but never propagate.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

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
