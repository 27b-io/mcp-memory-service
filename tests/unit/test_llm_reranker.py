"""Tests for optional LLM re-ranking of search results."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_anthropic_reranker_exists():
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    assert AnthropicReranker is not None


@pytest.mark.asyncio
async def test_reranker_returns_reordered_results():
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text='[{"hash": "hash_b", "score": 0.9}, {"hash": "hash_a", "score": 0.3}]')])
    )

    reranker = AnthropicReranker(client=mock_client, model="claude-haiku-4-5-20251001", timeout_ms=2000)
    scores = await reranker.rerank(
        query="find dream cycle",
        candidates=[
            {"content_hash": "hash_a", "summary": "email briefing"},
            {"content_hash": "hash_b", "summary": "dream cycle at 3AM"},
        ],
    )

    assert len(scores) == 2
    score_map = dict(scores)
    assert score_map["hash_b"] > score_map["hash_a"]


@pytest.mark.asyncio
async def test_reranker_nonfatal_on_error():
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

    reranker = AnthropicReranker(client=mock_client, model="test", timeout_ms=500)
    scores = await reranker.rerank(query="test", candidates=[{"content_hash": "h1", "summary": "s1"}])
    assert scores == []


@pytest.mark.asyncio
async def test_reranker_nonfatal_on_bad_json():
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text="this is not json")]))

    reranker = AnthropicReranker(client=mock_client, model="test", timeout_ms=500)
    scores = await reranker.rerank(query="test", candidates=[{"content_hash": "h1", "summary": "s1"}])
    assert scores == []
