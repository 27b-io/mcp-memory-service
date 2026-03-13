"""Tests for the /health/ready readiness probe endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mcp_memory_service.web.api import health as health_module
from mcp_memory_service.web.api.health import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture(autouse=True)
def _clear_readiness_cache():
    """Reset the module-level readiness cache between tests."""
    health_module._readiness_cache["result"] = None
    health_module._readiness_cache["expires_at"] = 0.0
    yield
    health_module._readiness_cache["result"] = None
    health_module._readiness_cache["expires_at"] = 0.0


@pytest.fixture
def client():
    app = _make_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestReadinessProbe:
    @pytest.mark.asyncio
    async def test_ready_with_healthy_http_provider(self, client: AsyncClient):
        """Returns ready=True when HTTP provider health_check succeeds."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        type(mock_provider).__name__ = "OpenAICompatAdapter"

        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=mock_provider):
            resp = await client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["provider"] == "OpenAICompatAdapter"

    @pytest.mark.asyncio
    async def test_ready_no_provider(self, client: AsyncClient):
        """Returns 503 when embedding provider is not initialized."""
        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=None):
            resp = await client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert "not initialized" in data["reason"]

    @pytest.mark.asyncio
    async def test_ready_local_provider_always_ready(self, client: AsyncClient):
        """LocalProvider (no health_check method) is always ready."""

        class FakeLocalProvider:
            pass

        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=FakeLocalProvider()):
            resp = await client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["provider"] == "FakeLocalProvider"

    @pytest.mark.asyncio
    async def test_ready_unhealthy_provider(self, client: AsyncClient):
        """Returns 503 when health_check returns False."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(return_value=False)
        type(mock_provider).__name__ = "OpenAICompatAdapter"

        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=mock_provider):
            resp = await client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert "unhealthy" in data["reason"]

    @pytest.mark.asyncio
    async def test_ready_health_check_exception(self, client: AsyncClient):
        """Returns 503 when health_check raises an exception."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(side_effect=ConnectionError("connection refused"))
        type(mock_provider).__name__ = "OpenAICompatAdapter"

        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=mock_provider):
            resp = await client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert "connection refused" in data["reason"]

    @pytest.mark.asyncio
    async def test_ready_caches_result(self, client: AsyncClient):
        """Second call within TTL returns cached result without calling provider."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        type(mock_provider).__name__ = "OpenAICompatAdapter"

        with patch("mcp_memory_service.shared_storage.get_embedding_provider", return_value=mock_provider):
            resp1 = await client.get("/health/ready")
            resp2 = await client.get("/health/ready")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # health_check called only once — second hit was cached
        mock_provider.health_check.assert_awaited_once()
