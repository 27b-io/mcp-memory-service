"""Tests for OpenAICompatAdapter — HTTP embedding provider."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mcp_memory_service.embedding.http import OpenAICompatAdapter
from mcp_memory_service.embedding.protocol import EmbeddingProvider


class TestOpenAICompatAdapter:
    def test_satisfies_protocol(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="nomic-ai/nomic-embed-text-v1.5",
        )
        assert isinstance(adapter, EmbeddingProvider)

    def test_dimensions_property(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=768,
        )
        assert adapter.dimensions == 768

    def test_dimensions_default(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
        )
        assert adapter.dimensions == 768

    def test_model_name_property(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="nomic-ai/nomic-embed-text-v1.5",
        )
        assert adapter.model_name == "nomic-ai/nomic-embed-text-v1.5"

    @pytest.mark.asyncio
    async def test_embed_batch_sends_correct_request(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="nomic-ai/nomic-embed-text-v1.5",
            dimensions=768,
        )
        mock_response = httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1] * 768, "index": 0},
                    {"embedding": [0.2] * 768, "index": 1},
                ],
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "usage": {"total_tokens": 10},
            },
            request=httpx.Request("POST", "http://localhost:8080/v1/embeddings"),
        )
        with patch.object(adapter, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            result = await adapter.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 768

    @pytest.mark.asyncio
    @patch("mcp_memory_service.embedding.http.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_503(self, mock_sleep):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=768,
        )
        error_response = httpx.Response(503, request=httpx.Request("POST", "test"))
        ok_response = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 768, "index": 0}], "model": "test"},
            request=httpx.Request("POST", "test"),
        )
        with patch.object(adapter, "_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=[error_response, ok_response])
            result = await adapter.embed_batch(["hello"])

        assert len(result) == 1
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2**0 = 1

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=768,
        )
        error_response = httpx.Response(401, request=httpx.Request("POST", "test"))
        with patch.object(adapter, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=error_response)
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.embed_batch(["hello"])
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await OpenAICompatAdapter(
            base_url="http://x",
            model_name="x",
            dimensions=768,
        ).embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    @patch("mcp_memory_service.embedding.http.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_timeout(self, mock_sleep):
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=768,
        )
        ok_response = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 768, "index": 0}], "model": "test"},
            request=httpx.Request("POST", "test"),
        )
        with patch.object(adapter, "_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=[httpx.TimeoutException("timed out"), ok_response])
            result = await adapter.embed_batch(["hello"])

        assert len(result) == 1
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_index_ordering(self):
        """Embeddings returned out of order should be sorted by index."""
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=3,
        )
        mock_response = httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.3, 0.3, 0.3], "index": 2},
                    {"embedding": [0.1, 0.1, 0.1], "index": 0},
                    {"embedding": [0.2, 0.2, 0.2], "index": 1},
                ],
                "model": "test",
            },
            request=httpx.Request("POST", "test"),
        )
        with patch.object(adapter, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            result = await adapter.embed_batch(["a", "b", "c"])

        assert result[0] == [0.1, 0.1, 0.1]
        assert result[1] == [0.2, 0.2, 0.2]
        assert result[2] == [0.3, 0.3, 0.3]

    @pytest.mark.asyncio
    async def test_batch_splitting(self):
        """Texts exceeding max_batch are split into multiple POST requests."""
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=3,
            max_batch=2,
        )

        def make_response(n_items):
            return httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 3, "index": i} for i in range(n_items)],
                    "model": "test",
                },
                request=httpx.Request("POST", "test"),
            )

        with patch.object(adapter, "_client") as mock_client:
            # 5 texts with max_batch=2 → 3 POST calls (2+2+1)
            mock_client.post = AsyncMock(side_effect=[make_response(2), make_response(2), make_response(1)])
            result = await adapter.embed_batch(["a", "b", "c", "d", "e"])

        assert len(result) == 5
        assert mock_client.post.call_count == 3
