"""Memory lifecycle hooks.

Provides HookRegistry for registering async pre/post callbacks on memory operations.
Pre-hooks can veto operations by raising HookValidationError.
Post-hooks are fire-and-forget: failures are logged but never propagate.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models.validators import ContentHash, NonNegativeInt, Tags

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


class CreateEvent(BaseModel):
    """Context for create lifecycle hooks."""

    content: str = Field(min_length=1)
    content_hash: ContentHash
    tags: Tags = []
    memory_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    client_hostname: str | None = None


class DeleteEvent(BaseModel):
    """Context for delete lifecycle hooks."""

    content_hash: ContentHash


class UpdateEvent(BaseModel):
    """Context for update lifecycle hooks."""

    content_hash: ContentHash
    fields: dict[str, Any] = Field(default_factory=dict)


class RetrieveEvent(BaseModel):
    """Context for retrieve lifecycle hooks."""

    query: str
    result_hashes: list[ContentHash] = Field(default_factory=list)
    result_count: NonNegativeInt = 0


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
            fn: Async callable receiving the event model for this hook type.
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
