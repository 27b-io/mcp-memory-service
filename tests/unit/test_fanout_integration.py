"""Integration test: retrieve_memories uses fan-out when intent is enabled."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult


@pytest.fixture(autouse=True)
async def _clear_cachekit_caches():
    """Clear CacheKit L1 caches before and after tests."""
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags

        await _cached_fetch_all_tags.ainvalidate_cache()
    except Exception:  # noqa: BLE001
        pass  # Best-effort: cache functions may not exist or Redis may be unavailable
    yield
    try:
        from mcp_memory_service.services.memory_service import _cached_fetch_all_tags

        await _cached_fetch_all_tags.ainvalidate_cache()
    except Exception:  # noqa: BLE001
        pass  # Best-effort: cache functions may not exist or Redis may be unavailable


def _make_result(content_hash: str, score: float) -> MemoryQueryResult:
    return MemoryQueryResult(
        memory=Memory(
            content=f"content for {content_hash}",
            content_hash=content_hash,
            tags=["test"],
            memory_type="note",
            created_at=datetime.now().timestamp(),
        ),
        relevance_score=score,
        debug_info={},
    )


@pytest.mark.asyncio
async def test_fanout_calls_batch_embed_and_search_by_vector():
    """When intent extracts multiple concepts, uses batched embed + parallel search_by_vector."""
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    mock_storage = AsyncMock()
    mock_storage.count_all_memories = AsyncMock(return_value=1000)
    mock_storage.count = AsyncMock(return_value=1000)
    mock_storage.get_all_tags = AsyncMock(return_value={"dream", "openclaw"})
    mock_storage.search_by_tags = AsyncMock(return_value=[])
    mock_storage.search_by_tag = AsyncMock(return_value=[])
    mock_storage.generate_embeddings_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768])
    mock_storage.search_by_vector = AsyncMock(return_value=[_make_result("hash1", 0.7)])

    service = MemoryService(storage=mock_storage)

    mock_intent = QueryIntentResult(
        original_query="dream cycle OpenClaw consolidation",
        sub_queries=["dream cycle", "OpenClaw", "dream cycle OpenClaw consolidation"],
        concepts=["dream cycle", "OpenClaw"],
    )

    with (
        patch("mcp_memory_service.services.memory_service.get_analyzer") as mock_get_analyzer,
        patch("mcp_memory_service.services.memory_service.settings") as mock_settings,
    ):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_intent
        mock_get_analyzer.return_value = mock_analyzer

        mock_settings.intent.enabled = True
        mock_settings.intent.spacy_model = "en_core_web_sm"
        mock_settings.intent.min_query_tokens = 3
        mock_settings.intent.max_sub_queries = 4
        mock_settings.intent.graph_inject = False
        mock_settings.intent.llm_rerank = False
        mock_settings.hybrid_search.hybrid_alpha = None
        mock_settings.hybrid_search.recency_decay = 0.0
        mock_settings.hybrid_search.temporal_decay_lambda = 0.0
        mock_settings.hybrid_search.temporal_decay_base = None
        mock_settings.hybrid_search.adaptive_threshold_small = 500
        mock_settings.hybrid_search.adaptive_threshold_large = 5000
        mock_settings.falkordb.spreading_activation_boost = 0.0
        mock_settings.falkordb.hebbian_boost = 0.0
        mock_settings.falkordb.enabled = False
        mock_settings.salience.enabled = False
        mock_settings.spaced_repetition.enabled = False
        mock_settings.encoding_context.enabled = False
        mock_settings.semantic_tag.enabled = False

        await service.retrieve_memories(
            query="dream cycle OpenClaw consolidation",
            page=1,
            page_size=10,
        )

    # Verify: batch embed was called with all sub-queries
    mock_storage.generate_embeddings_batch.assert_called_once()
    batch_call_texts = mock_storage.generate_embeddings_batch.call_args[0][0]
    assert len(batch_call_texts) == 3

    # Verify: search_by_vector called for each embedding (parallel)
    assert mock_storage.search_by_vector.call_count == 3


@pytest.mark.asyncio
async def test_fanout_fallback_to_single_vector_on_error():
    """When fan-out search raises, should fall back to _single_vector_search."""
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    mock_storage = AsyncMock()
    mock_storage.count_all_memories = AsyncMock(return_value=100)
    mock_storage.count = AsyncMock(return_value=100)
    mock_storage.get_all_tags = AsyncMock(return_value={"concept", "full"})
    mock_storage.search_by_tags = AsyncMock(return_value=[])
    mock_storage.search_by_tag = AsyncMock(return_value=[])
    # Make batch embed raise to trigger fallback
    mock_storage.generate_embeddings_batch = AsyncMock(side_effect=RuntimeError("embedding failed"))

    fallback_result = _make_result("fallback_hash", 0.85)
    mock_storage.retrieve = AsyncMock(return_value=[fallback_result])
    mock_storage.search_by_vector = AsyncMock(return_value=[])

    mock_intent = QueryIntentResult(
        original_query="full query about concept one and concept two",
        sub_queries=["concept one", "concept two", "full query about concept one and concept two"],
        concepts=["concept one", "concept two"],
    )

    with (
        patch("mcp_memory_service.services.memory_service.get_analyzer") as mock_get_analyzer,
        patch("mcp_memory_service.services.memory_service.settings") as mock_settings,
    ):
        mock_settings.three_tier.enabled = False
        service = MemoryService(storage=mock_storage)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_intent
        mock_get_analyzer.return_value = mock_analyzer

        mock_settings.intent.enabled = True
        mock_settings.intent.spacy_model = "en_core_web_sm"
        mock_settings.intent.min_query_tokens = 3
        mock_settings.intent.max_sub_queries = 4
        mock_settings.intent.graph_inject = False
        mock_settings.intent.llm_rerank = False
        mock_settings.hybrid_search.hybrid_alpha = None
        mock_settings.hybrid_search.recency_decay = 0.0
        mock_settings.hybrid_search.temporal_decay_lambda = 0.0
        mock_settings.hybrid_search.temporal_decay_base = 0.0
        mock_settings.hybrid_search.adaptive_threshold_small = 500
        mock_settings.hybrid_search.adaptive_threshold_large = 5000
        mock_settings.falkordb.spreading_activation_boost = 0.0
        mock_settings.falkordb.hebbian_boost = 0.0
        mock_settings.falkordb.enabled = False
        mock_settings.salience.enabled = False
        mock_settings.spaced_repetition.enabled = False
        mock_settings.encoding_context.enabled = False
        mock_settings.semantic_tag.enabled = False

        result = await service.retrieve_memories(
            query="full query about concept one and concept two",
            page=1,
            page_size=10,
        )

    # Fallback path should have called storage.retrieve (single-vector)
    assert mock_storage.retrieve.called
    assert len(result.get("memories", [])) > 0
