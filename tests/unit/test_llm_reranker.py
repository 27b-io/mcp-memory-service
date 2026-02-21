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


@pytest.mark.asyncio
async def test_reranker_handles_markdown_fenced_json():
    """Reranker should extract JSON from markdown code fences."""
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = MagicMock()
    full_hash = "abcdef1234567890" + "a" * 16
    fenced = f'```json\n[{{"hash": "{full_hash[:16]}", "score": 0.9}}]\n```'
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=fenced)]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    reranker = AnthropicReranker(client=mock_client, model="test-model")
    candidates = [{"content_hash": full_hash, "summary": "test memory"}]
    results = await reranker.rerank("test query", candidates)

    assert len(results) == 1
    assert results[0][0] == full_hash
    assert results[0][1] == 0.9


@pytest.mark.asyncio
async def test_reranker_uses_16_char_hash_prefix():
    """Reranker uses 16-char hash prefixes so near-collisions at 12 chars are distinct."""
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = MagicMock()
    hash_a = "abcdef12345678AA" + "x" * 16
    hash_b = "abcdef12345678BB" + "y" * 16

    response_json = f'[{{"hash": "{hash_a[:16]}", "score": 0.9}}, {{"hash": "{hash_b[:16]}", "score": 0.3}}]'
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_json)]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    reranker = AnthropicReranker(client=mock_client, model="test-model")
    candidates = [
        {"content_hash": hash_a, "summary": "memory A"},
        {"content_hash": hash_b, "summary": "memory B"},
    ]
    results = await reranker.rerank("test", candidates)

    assert len(results) == 2
    result_map = dict(results)
    assert result_map[hash_a] == 0.9
    assert result_map[hash_b] == 0.3
