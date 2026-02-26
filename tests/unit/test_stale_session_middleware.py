"""Tests for StaleSessionMiddleware — converts stale-session 400 → 404."""

from http import HTTPStatus

import pytest

from mcp_memory_service.unified_server import StaleSessionMiddleware


def _make_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers or [],
    }


class _Captured:
    """Collects ASGI messages sent by the middleware."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)


def _fake_app(status: int, body: bytes):
    """Return a minimal ASGI app that sends a fixed response."""

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


def _fake_streamed_app(status: int, chunks: list[bytes]):
    """Return an ASGI app that sends the body in multiple chunks with more_body."""

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": not is_last,
                }
            )

    return app


@pytest.mark.asyncio
async def test_stale_session_400_rewritten_to_404():
    """A 400 with the stale-session message should become 404."""
    inner = _fake_app(400, b"Bad Request: No valid session ID provided")
    mw = StaleSessionMiddleware(inner)

    scope = _make_scope(headers=[(b"mcp-session-id", b"dead-session-abc")])
    captured = _Captured()
    await mw(scope, None, captured)

    assert len(captured.messages) == 2
    assert captured.messages[0]["status"] == HTTPStatus.NOT_FOUND
    assert b"Session expired" in captured.messages[1]["body"]


@pytest.mark.asyncio
async def test_non_session_400_passes_through():
    """A 400 for some other reason should NOT be rewritten."""
    inner = _fake_app(400, b"Bad Request: Invalid JSON")
    mw = StaleSessionMiddleware(inner)

    scope = _make_scope(headers=[(b"mcp-session-id", b"some-session")])
    captured = _Captured()
    await mw(scope, None, captured)

    assert len(captured.messages) == 2
    assert captured.messages[0]["status"] == HTTPStatus.BAD_REQUEST
    assert b"Invalid JSON" in captured.messages[1]["body"]


@pytest.mark.asyncio
async def test_200_passes_through_unchanged():
    """Successful responses are never touched."""
    inner = _fake_app(200, b'{"result": "ok"}')
    mw = StaleSessionMiddleware(inner)

    scope = _make_scope(headers=[(b"mcp-session-id", b"valid-session")])
    captured = _Captured()
    await mw(scope, None, captured)

    assert len(captured.messages) == 2
    assert captured.messages[0]["status"] == 200
    assert captured.messages[1]["body"] == b'{"result": "ok"}'


@pytest.mark.asyncio
async def test_no_session_header_fast_path():
    """Requests without mcp-session-id skip the middleware entirely."""
    inner = _fake_app(400, b"Bad Request: No valid session ID provided")
    mw = StaleSessionMiddleware(inner)

    scope = _make_scope(headers=[])
    captured = _Captured()
    await mw(scope, None, captured)

    # Even though the body matches, no rewrite — fast path skipped it
    assert captured.messages[0]["status"] == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_non_http_scope_passes_through():
    """Non-HTTP scopes (websocket, lifespan) are ignored."""
    inner = _fake_app(400, b"Bad Request: No valid session ID provided")
    mw = StaleSessionMiddleware(inner)

    scope = {"type": "lifespan"}
    captured = _Captured()
    await mw(scope, None, captured)

    # The inner app was called, messages pass through
    assert captured.messages[0]["status"] == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_streamed_400_passes_through():
    """A streamed 400 (more_body=True) is never our target — forward unchanged."""
    inner = _fake_streamed_app(400, [b"Bad Request: ", b"No valid session ID provided"])
    mw = StaleSessionMiddleware(inner)

    scope = _make_scope(headers=[(b"mcp-session-id", b"stale-session")])
    captured = _Captured()
    await mw(scope, None, captured)

    # Start message + 2 body chunks all forwarded, status untouched
    assert captured.messages[0]["status"] == HTTPStatus.BAD_REQUEST
    bodies = [m["body"] for m in captured.messages if m["type"] == "http.response.body"]
    assert bodies == [b"Bad Request: ", b"No valid session ID provided"]
