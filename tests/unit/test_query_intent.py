"""Tests for query intent extraction and concept decomposition."""

import pytest

from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

# Check if spaCy model is available for tests that need it
_spacy_model_available = False
try:
    import spacy

    spacy.load("en_core_web_sm")
    _spacy_model_available = True
except (ImportError, OSError):
    pass

needs_spacy_model = pytest.mark.skipif(not _spacy_model_available, reason="en_core_web_sm not installed")


@pytest.fixture(autouse=True)
def _reset_spacy_singleton():
    """Reset spaCy singleton between tests to avoid stale state."""
    SpaCyAnalyzer._nlp = None
    yield
    SpaCyAnalyzer._nlp = None


def test_query_intent_result_dataclass():
    """QueryIntentResult holds original query, sub-queries, and concepts."""
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    r = QueryIntentResult(
        original_query="dream cycle 3AM OpenClaw consolidation",
        sub_queries=["dream cycle", "OpenClaw", "consolidation"],
        concepts=["dream cycle", "OpenClaw", "consolidation"],
    )
    assert r.original_query == "dream cycle 3AM OpenClaw consolidation"
    assert len(r.sub_queries) == 3


@needs_spacy_model
def test_spacy_analyzer_extracts_noun_phrases():
    """SpaCyAnalyzer extracts noun phrases and entities from multi-concept query."""
    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("dream cycle 3AM OpenClaw consolidation")
    assert len(result.sub_queries) >= 2
    assert result.original_query == "dream cycle 3AM OpenClaw consolidation"
    assert result.original_query in result.sub_queries


def test_spacy_analyzer_short_query_no_fanout():
    """Short queries (<= min tokens) return only the original as sub-query."""
    analyzer = SpaCyAnalyzer(min_query_tokens=3)
    result = analyzer.analyze("dream cycle")
    assert result.sub_queries == ["dream cycle"]
    assert result.concepts == []


def test_spacy_analyzer_single_word():
    """Single word query returns itself, no fan-out."""
    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("authentication")
    assert result.sub_queries == ["authentication"]


@needs_spacy_model
def test_spacy_analyzer_respects_max_sub_queries():
    """Sub-queries capped at max_sub_queries."""
    analyzer = SpaCyAnalyzer(max_sub_queries=2)
    result = analyzer.analyze("OpenClaw dream cycle memory consolidation graph injection system")
    concept_subs = [q for q in result.sub_queries if q != result.original_query]
    assert len(concept_subs) <= 2


@needs_spacy_model
def test_spacy_analyzer_deduplicates():
    """Duplicate concepts are deduplicated."""
    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("memory memory memory")
    assert len(result.sub_queries) == len(set(result.sub_queries))


def test_spacy_analyzer_graceful_without_model():
    """SpaCyAnalyzer falls back gracefully when model is not installed."""
    analyzer = SpaCyAnalyzer(model_name="nonexistent_model_xyz")
    result = analyzer.analyze("dream cycle 3AM OpenClaw consolidation")
    # Should return no-fanout result instead of crashing
    assert result.sub_queries == ["dream cycle 3AM OpenClaw consolidation"]
    assert result.concepts == []


def test_fallback_analyzer_when_spacy_unavailable():
    """FallbackAnalyzer uses keyword extraction when spaCy is not installed."""
    from mcp_memory_service.utils.query_intent import FallbackAnalyzer

    analyzer = FallbackAnalyzer(min_query_tokens=3)
    result = analyzer.analyze("dream cycle 3AM OpenClaw consolidation")
    assert result.original_query == "dream cycle 3AM OpenClaw consolidation"
    assert len(result.sub_queries) >= 1
    assert result.original_query in result.sub_queries


def test_get_analyzer_returns_instance():
    """get_analyzer() returns a working analyzer (spaCy or fallback)."""
    import mcp_memory_service.utils.query_intent as mod
    from mcp_memory_service.utils.query_intent import get_analyzer

    mod._analyzer = None

    analyzer = get_analyzer()
    result = analyzer.analyze("test query for analysis")
    assert result.original_query == "test query for analysis"
