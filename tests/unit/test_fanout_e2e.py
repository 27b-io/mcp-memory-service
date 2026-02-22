"""End-to-end test: full fan-out pipeline with all stages."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult


def _make_result(content_hash: str, score: float) -> MemoryQueryResult:
    return MemoryQueryResult(
        memory=Memory(
            content=f"content for {content_hash}",
            content_hash=content_hash,
            tags=["test"],
            memory_type="note",
            created_at=datetime.now().timestamp(),
            summary=f"summary of {content_hash}",
        ),
        relevance_score=score,
        debug_info={},
    )


@pytest.mark.asyncio
async def test_full_pipeline_returns_wider_results():
    """Multi-concept query returns more results than single-vector would."""
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    mock_storage = AsyncMock()
    mock_storage.count_all_memories = AsyncMock(return_value=5000)
    mock_storage.count = AsyncMock(return_value=5000)
    mock_storage.get_all_tags = AsyncMock(return_value={"dream", "openclaw", "cron"})
    mock_storage.search_by_tags = AsyncMock(return_value=[])
    mock_storage.search_by_tag = AsyncMock(return_value=[])
    mock_storage.generate_embeddings_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768])

    mock_storage.search_by_vector = AsyncMock(
        side_effect=[
            [_make_result("hash_dream", 0.7)],
            [_make_result("hash_claw", 0.65)],
            [_make_result("hash_dream", 0.6), _make_result("hash_consol", 0.55)],
        ]
    )
    mock_storage.get_memories_batch = AsyncMock(return_value=[])

    service = MemoryService(storage=mock_storage, graph_client=None)

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
        mock_settings.hybrid_search.hybrid_alpha = 0.7
        mock_settings.hybrid_search.recency_decay = 0.0
        mock_settings.hybrid_search.temporal_decay_lambda = 0.0
        mock_settings.hybrid_search.temporal_decay_base = 0.7
        mock_settings.falkordb.spreading_activation_boost = 0.0
        mock_settings.falkordb.hebbian_boost = 0.0
        mock_settings.falkordb.enabled = False
        mock_settings.salience.enabled = False
        mock_settings.spaced_repetition.enabled = False
        mock_settings.encoding_context.enabled = False
        mock_settings.semantic_tag.enabled = False

        result = await service.retrieve_memories(
            query="dream cycle OpenClaw consolidation",
            page=1,
            page_size=10,
            min_similarity=0.5,
        )

    memories = result.get("memories", [])
    hashes = {m["content_hash"] for m in memories}

    assert "hash_dream" in hashes, "dream cycle result missing"
    assert "hash_claw" in hashes, "OpenClaw result missing"
    assert "hash_consol" in hashes, "consolidation result missing"
    assert len(memories) == 3

    # hash_dream should rank highest (found by 2 of 3 sub-queries)
    assert memories[0]["content_hash"] == "hash_dream"
