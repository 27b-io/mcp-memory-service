# Query Intent Inference & Search Fan-Out — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pluggable query intent inference with NLP concept extraction, multi-vector parallel fan-out, graph neighbor injection, and optional LLM re-ranking to transform search from single-vector AND semantics to multi-concept OR semantics.

**Architecture:** Two-stage pipeline — pre-search spaCy concept extraction fans out parallel Qdrant searches with batched embeddings; post-search optional Haiku LLM re-ranks results. Graph layer injects associatively-connected neighbors as new candidates, not just score boosts. Every stage is non-fatal and independently toggleable.

**Tech Stack:** spaCy (en_core_web_sm), sentence-transformers (batched encode), asyncio.gather (parallel fan-out), Anthropic SDK (optional Haiku re-rank), pydantic-settings (config), FalkorDB (graph injection)

**Design Doc:** `docs/plans/2026-02-21-query-intent-inference-design.md`

**Issue:** #108

---

### Task 1: Add QueryIntentSettings to config

**Files:**
- Modify: `src/mcp_memory_service/config.py:681-811`
- Test: `tests/unit/test_config_intent.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_config_intent.py
"""Tests for QueryIntentSettings configuration."""
import os
import pytest


def test_default_settings():
    """QueryIntentSettings loads with sensible defaults."""
    from mcp_memory_service.config import QueryIntentSettings

    s = QueryIntentSettings()
    assert s.enabled is True
    assert s.spacy_model == "en_core_web_sm"
    assert s.max_sub_queries == 4
    assert s.min_query_tokens == 3
    assert s.graph_inject is True
    assert s.graph_inject_limit == 10
    assert s.graph_inject_min_activation == 0.05
    assert s.llm_rerank is False
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-haiku-4-5-20251001"
    assert s.llm_timeout_ms == 2000


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("MCP_INTENT_ENABLED", "false")
    monkeypatch.setenv("MCP_INTENT_MAX_SUB_QUERIES", "6")
    monkeypatch.setenv("MCP_INTENT_LLM_RERANK", "true")
    from mcp_memory_service.config import QueryIntentSettings

    s = QueryIntentSettings()
    assert s.enabled is False
    assert s.max_sub_queries == 6
    assert s.llm_rerank is True


def test_settings_on_main_settings():
    """QueryIntentSettings accessible as settings.intent."""
    from mcp_memory_service.config import Settings

    s = Settings()
    assert hasattr(s, "intent")
    assert s.intent.enabled is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_intent.py -v -x`
Expected: FAIL — `QueryIntentSettings` does not exist

**Step 3: Write minimal implementation**

In `src/mcp_memory_service/config.py`, add after `HybridSearchSettings` (after line ~681):

```python
class QueryIntentSettings(BaseSettings):
    """Query intent inference configuration for concept extraction and fan-out."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_INTENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Master switch
    enabled: bool = Field(default=True, description="Enable query intent inference and fan-out")

    # NLP concept extraction
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model for concept extraction")
    max_sub_queries: int = Field(default=4, ge=1, le=8, description="Maximum sub-queries from concept extraction")
    min_query_tokens: int = Field(
        default=3, ge=1, description="Minimum meaningful tokens to trigger fan-out (shorter queries use single-vector)"
    )

    # Graph injection
    graph_inject: bool = Field(default=True, description="Inject graph neighbors into candidate pool")
    graph_inject_limit: int = Field(default=10, ge=1, le=50, description="Maximum graph-injected neighbors")
    graph_inject_min_activation: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum spreading activation score for injection"
    )

    # LLM re-ranking (off by default)
    llm_rerank: bool = Field(default=False, description="Enable LLM-based re-ranking of results")
    llm_provider: str = Field(default="anthropic", description="LLM provider for re-ranking")
    llm_model: str = Field(default="claude-haiku-4-5-20251001", description="LLM model for re-ranking")
    llm_timeout_ms: int = Field(default=2000, ge=500, le=10000, description="LLM re-ranking timeout in milliseconds")
```

Then add to `Settings` class (after line ~811):

```python
    intent: QueryIntentSettings = Field(default_factory=QueryIntentSettings)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config_intent.py -v -x`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/config.py tests/unit/test_config_intent.py
git commit -m "feat(config): add QueryIntentSettings for intent inference"
```

---

### Task 2: Lower default min_similarity from 0.6 to 0.5

**Files:**
- Modify: `src/mcp_memory_service/mcp_server.py:273`
- Test: `tests/unit/test_mcp_server.py` (if exists, otherwise verify manually)

**Step 1: Write the failing test**

```python
# tests/unit/test_min_similarity_default.py
"""Verify default min_similarity is 0.5 after v11.10.0 cosine score fix."""
import inspect
from mcp_memory_service.mcp_server import search


def test_default_min_similarity_is_0_5():
    """Default min_similarity should be 0.5, not 0.6."""
    sig = inspect.signature(search)
    default = sig.parameters["min_similarity"].default
    assert default == 0.5, f"Expected 0.5, got {default}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_min_similarity_default.py -v -x`
Expected: FAIL — `default == 0.6`

**Step 3: Write minimal implementation**

In `src/mcp_memory_service/mcp_server.py` line 273, change:

```python
# OLD
    min_similarity: float = 0.6,
# NEW
    min_similarity: float = 0.5,
```

Also update the docstring at line ~293-294:

```python
# OLD
        min_similarity: Similarity threshold 0.0-1.0 (default: 0.6). Higher=stricter.
# NEW
        min_similarity: Similarity threshold 0.0-1.0 (default: 0.5). Higher=stricter.
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_min_similarity_default.py -v -x`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (no tests depend on 0.6 default)

**Step 6: Commit**

```bash
git add src/mcp_memory_service/mcp_server.py tests/unit/test_min_similarity_default.py
git commit -m "fix: lower default min_similarity from 0.6 to 0.5"
```

---

### Task 3: Add SpaCyAnalyzer for concept extraction

**Files:**
- Create: `src/mcp_memory_service/utils/query_intent.py`
- Test: `tests/unit/test_query_intent.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_query_intent.py
"""Tests for query intent extraction and concept decomposition."""
import pytest
from dataclasses import dataclass


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


def test_spacy_analyzer_extracts_noun_phrases():
    """SpaCyAnalyzer extracts noun phrases and entities from multi-concept query."""
    from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("dream cycle 3AM OpenClaw consolidation")
    # Should extract at least 2 concepts (noun phrases / compounds)
    assert len(result.sub_queries) >= 2
    assert result.original_query == "dream cycle 3AM OpenClaw consolidation"
    # Original query should always be included as a sub-query
    assert result.original_query in result.sub_queries


def test_spacy_analyzer_short_query_no_fanout():
    """Short queries (≤ min tokens) return only the original as sub-query."""
    from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

    analyzer = SpaCyAnalyzer(min_query_tokens=3)
    result = analyzer.analyze("dream cycle")
    assert result.sub_queries == ["dream cycle"]
    assert result.concepts == []


def test_spacy_analyzer_single_word():
    """Single word query returns itself, no fan-out."""
    from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("authentication")
    assert result.sub_queries == ["authentication"]


def test_spacy_analyzer_respects_max_sub_queries():
    """Sub-queries capped at max_sub_queries."""
    from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

    analyzer = SpaCyAnalyzer(max_sub_queries=2)
    result = analyzer.analyze("OpenClaw dream cycle memory consolidation graph injection system")
    # max_sub_queries=2 means 2 concept sub-queries + 1 original = 3 total,
    # but we cap concept sub-queries at max_sub_queries
    concept_subs = [q for q in result.sub_queries if q != result.original_query]
    assert len(concept_subs) <= 2


def test_spacy_analyzer_deduplicates():
    """Duplicate concepts are deduplicated."""
    from mcp_memory_service.utils.query_intent import SpaCyAnalyzer

    analyzer = SpaCyAnalyzer()
    result = analyzer.analyze("memory memory memory")
    # Should not produce 3 copies of "memory"
    assert len(result.sub_queries) == len(set(result.sub_queries))


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
    from mcp_memory_service.utils.query_intent import get_analyzer

    analyzer = get_analyzer()
    result = analyzer.analyze("test query for analysis")
    assert result.original_query == "test query for analysis"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_query_intent.py -v -x`
Expected: FAIL — `query_intent` module does not exist

**Step 3: Write implementation**

```python
# src/mcp_memory_service/utils/query_intent.py
"""
Query intent extraction for multi-vector search fan-out.

Decomposes natural language queries into concept sub-queries using spaCy NLP.
Falls back to simple keyword extraction if spaCy is unavailable.

Lazy-loaded singleton pattern (matches emotional_analysis.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Stop words for fallback analyzer (subset — full set in hybrid_search.py)
_FALLBACK_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no",
    "this", "that", "these", "those", "it", "its", "my", "your", "our",
    "their", "what", "which", "who", "how", "when", "where", "why",
    "about", "up", "out", "if", "then", "so", "than", "too", "very",
    "just", "also", "any", "each", "every", "all", "some", "me", "i",
})
_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class QueryIntentResult:
    """Result of query intent analysis."""

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)


@runtime_checkable
class QueryIntentAnalyzer(Protocol):
    """Protocol for pluggable query intent analyzers."""

    def analyze(self, query: str) -> QueryIntentResult: ...


class SpaCyAnalyzer:
    """
    spaCy-based concept extractor.

    Uses noun phrases, named entities, and compound nouns to decompose
    multi-concept queries into focused sub-queries for parallel fan-out.

    Lazy-loads the spaCy model on first use (singleton pattern).
    """

    _nlp = None

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        max_sub_queries: int = 4,
        min_query_tokens: int = 3,
    ):
        self._model_name = model_name
        self._max_sub_queries = max_sub_queries
        self._min_query_tokens = min_query_tokens

    def _ensure_model(self):
        """Lazy-load spaCy model (class-level singleton)."""
        if SpaCyAnalyzer._nlp is None:
            import spacy

            SpaCyAnalyzer._nlp = spacy.load(self._model_name)
            logger.info(f"Loaded spaCy model: {self._model_name}")

    def analyze(self, query: str) -> QueryIntentResult:
        """Extract concepts and generate sub-queries from a natural language query."""
        query = query.strip()
        if not query:
            return QueryIntentResult(original_query=query, sub_queries=[query])

        # Tokenize to count meaningful tokens
        tokens = [t for t in _TOKEN_PATTERN.split(query.lower()) if t and t not in _FALLBACK_STOP_WORDS and len(t) > 1]

        # Short-circuit: not enough tokens to warrant fan-out
        if len(tokens) < self._min_query_tokens:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        self._ensure_model()
        doc = SpaCyAnalyzer._nlp(query)

        concepts: list[str] = []
        seen: set[str] = set()

        # Extract noun phrases (includes compound nouns automatically)
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            lower = text.lower()
            if lower not in seen and len(text) > 1:
                seen.add(lower)
                concepts.append(text)

        # Extract named entities not already covered by noun phrases
        for ent in doc.ents:
            text = ent.text.strip()
            lower = text.lower()
            if lower not in seen and len(text) > 1:
                seen.add(lower)
                concepts.append(text)

        # Extract standalone tokens not covered by any concept
        covered_tokens = set()
        for c in concepts:
            for t in _TOKEN_PATTERN.split(c.lower()):
                if t:
                    covered_tokens.add(t)

        for token in doc:
            if (
                token.text.lower() not in covered_tokens
                and token.text.lower() not in _FALLBACK_STOP_WORDS
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 1
                and token.text.lower() not in seen
            ):
                seen.add(token.text.lower())
                concepts.append(token.text)

        # Cap concepts at max_sub_queries
        concepts = concepts[: self._max_sub_queries]

        # Build sub-queries: concepts + always include original
        sub_queries: list[str] = []
        seen_queries: set[str] = set()
        for concept in concepts:
            lower = concept.lower()
            if lower not in seen_queries:
                seen_queries.add(lower)
                sub_queries.append(concept)

        # Always include original query (deduplicated)
        if query.lower() not in seen_queries:
            sub_queries.append(query)

        # If we only got the original back, no point in fan-out
        if len(sub_queries) <= 1:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        return QueryIntentResult(
            original_query=query,
            sub_queries=sub_queries,
            concepts=concepts,
        )


class FallbackAnalyzer:
    """
    Simple keyword-based fallback when spaCy is unavailable.

    Uses regex tokenization and stop word removal (same as extract_query_keywords).
    """

    def __init__(self, min_query_tokens: int = 3, max_sub_queries: int = 4):
        self._min_query_tokens = min_query_tokens
        self._max_sub_queries = max_sub_queries

    def analyze(self, query: str) -> QueryIntentResult:
        query = query.strip()
        if not query:
            return QueryIntentResult(original_query=query, sub_queries=[query])

        tokens = [t for t in _TOKEN_PATTERN.split(query.lower()) if t and t not in _FALLBACK_STOP_WORDS and len(t) > 1]

        if len(tokens) < self._min_query_tokens:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        # Each token becomes a concept (simple approach)
        concepts = tokens[: self._max_sub_queries]

        sub_queries = list(dict.fromkeys(concepts))  # dedupe preserving order
        if query.lower() not in {q.lower() for q in sub_queries}:
            sub_queries.append(query)

        return QueryIntentResult(original_query=query, sub_queries=sub_queries, concepts=concepts)


# ── Singleton accessor ──────────────────────────────────────────────────

_analyzer: QueryIntentAnalyzer | None = None


def get_analyzer(
    model_name: str = "en_core_web_sm",
    max_sub_queries: int = 4,
    min_query_tokens: int = 3,
) -> QueryIntentAnalyzer:
    """
    Get or create the query intent analyzer singleton.

    Returns SpaCyAnalyzer if spaCy is available, FallbackAnalyzer otherwise.
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        import spacy  # noqa: F401

        _analyzer = SpaCyAnalyzer(
            model_name=model_name,
            max_sub_queries=max_sub_queries,
            min_query_tokens=min_query_tokens,
        )
        logger.info("Query intent: using SpaCyAnalyzer")
    except ImportError:
        _analyzer = FallbackAnalyzer(
            min_query_tokens=min_query_tokens,
            max_sub_queries=max_sub_queries,
        )
        logger.info("Query intent: spaCy not available, using FallbackAnalyzer")

    return _analyzer
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_query_intent.py -v -x`
Expected: PASS (9 tests). If spaCy not installed in dev env, install first:

```bash
uv add spacy --optional dev
uv run python -m spacy download en_core_web_sm
```

**Step 5: Lint**

Run: `uv run ruff check src/mcp_memory_service/utils/query_intent.py tests/unit/test_query_intent.py`
Run: `uv run ruff format src/mcp_memory_service/utils/query_intent.py tests/unit/test_query_intent.py`

**Step 6: Commit**

```bash
git add src/mcp_memory_service/utils/query_intent.py tests/unit/test_query_intent.py
git commit -m "feat: add SpaCyAnalyzer for query concept extraction"
```

---

### Task 4: Add batch embedding and vector search methods to storage

**Files:**
- Modify: `src/mcp_memory_service/storage/base.py:94-118`
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py`
- Test: `tests/unit/test_storage_fanout.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_storage_fanout.py
"""Tests for batch embedding and pre-embedded vector search."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


@pytest.fixture
def mock_qdrant_storage():
    """Create a QdrantStorage with mocked internals for testing new methods."""
    from mcp_memory_service.storage.qdrant_storage import QdrantStorage

    storage = QdrantStorage.__new__(QdrantStorage)
    storage.collection_name = "test_collection"
    storage._vector_size = 768
    storage.embedding_model = "test-model"
    storage._model_lock = __import__("threading").Lock()
    storage.client = MagicMock()
    storage._failure_count = 0
    storage._circuit_open_until = None
    storage._failure_threshold = 5
    storage._circuit_timeout = 60
    return storage


def test_generate_embeddings_batch_signature(mock_qdrant_storage):
    """generate_embeddings_batch exists and is async."""
    assert hasattr(mock_qdrant_storage, "generate_embeddings_batch")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.generate_embeddings_batch)


def test_search_by_vector_signature(mock_qdrant_storage):
    """search_by_vector exists and is async."""
    assert hasattr(mock_qdrant_storage, "search_by_vector")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.search_by_vector)


def test_get_memories_batch_signature(mock_qdrant_storage):
    """get_memories_batch exists and is async."""
    assert hasattr(mock_qdrant_storage, "get_memories_batch")
    assert asyncio.iscoroutinefunction(mock_qdrant_storage.get_memories_batch)


@pytest.mark.asyncio
async def test_generate_embeddings_batch_calls_encode(mock_qdrant_storage):
    """Batch embedding calls model.encode with all texts at once."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
    mock_model.prompts = {"query": "search_query: "}
    mock_qdrant_storage._embedding_model_instance = mock_model

    result = await mock_qdrant_storage.generate_embeddings_batch(["query one", "query two"])

    assert len(result) == 2
    assert len(result[0]) == 768
    # Verify encode was called ONCE with both texts (batched)
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args
    assert len(call_args[0][0]) == 2  # Two texts in single call


@pytest.mark.asyncio
async def test_search_by_vector_skips_embedding(mock_qdrant_storage):
    """search_by_vector uses provided embedding, does not call encode."""
    from qdrant_client.models import ScoredPoint

    mock_model = MagicMock()
    mock_qdrant_storage._embedding_model_instance = mock_model

    # Mock client.query_points to return empty
    mock_qdrant_storage.client.query_points.return_value = MagicMock(points=[])

    embedding = [0.1] * 768
    result = await mock_qdrant_storage.search_by_vector(
        embedding=embedding, n_results=10, min_similarity=0.0
    )

    assert result == []
    # Critically: encode was NEVER called
    mock_model.encode.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_storage_fanout.py -v -x`
Expected: FAIL — methods don't exist

**Step 3: Write implementation**

Add abstract methods to `src/mcp_memory_service/storage/base.py` (after `retrieve` at line ~118):

```python
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batched forward pass.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Note:
            Default implementation calls retrieve's embedding path sequentially.
            Qdrant backend overrides with true batched encode.
        """
        raise NotImplementedError("Subclass must implement generate_embeddings_batch")

    async def search_by_vector(
        self,
        embedding: list[float],
        n_results: int = 10,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """Search using a pre-computed embedding vector (skips embedding generation).

        Args:
            embedding: Pre-computed query embedding vector
            n_results: Maximum results
            tags: Optional tag filter
            memory_type: Optional type filter
            min_similarity: Optional similarity threshold
            offset: Pagination offset

        Returns:
            List of MemoryQueryResult sorted by similarity
        """
        raise NotImplementedError("Subclass must implement search_by_vector")

    async def get_memories_batch(self, content_hashes: list[str]) -> list[Memory]:
        """Fetch multiple memories by content hash in a single operation.

        Args:
            content_hashes: List of content hashes to fetch

        Returns:
            List of Memory objects (order not guaranteed, missing hashes skipped)
        """
        raise NotImplementedError("Subclass must implement get_memories_batch")
```

Add implementations to `src/mcp_memory_service/storage/qdrant_storage.py`. Place after `_generate_query_embedding` (after line ~900):

```python
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batched forward pass."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()

        def _batch_encode():
            # Ensure model is loaded (reuse lazy-loading from _generate_embedding)
            if not hasattr(self, "_embedding_model_instance"):
                with self._model_lock:
                    if not hasattr(self, "_embedding_model_instance"):
                        from .qdrant_storage import SentenceTransformer
                        from ..utils.system_detection import get_torch_device

                        device = get_torch_device()
                        self._embedding_model_instance = SentenceTransformer(
                            self.embedding_model, device=device, trust_remote_code=True
                        )

            model = self._embedding_model_instance
            prompts = getattr(model, "prompts", None) or {}
            if "query" in prompts:
                try:
                    embeddings = model.encode(texts, prompt_name="query", convert_to_numpy=True)
                except TypeError:
                    prefix = prompts["query"]
                    prefixed = [f"{prefix}{t}" for t in texts]
                    embeddings = model.encode(prefixed, convert_to_numpy=True)
            else:
                embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        return await loop.run_in_executor(None, _batch_encode)

    async def search_by_vector(
        self,
        embedding: list[float],
        n_results: int = 10,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """Search using a pre-computed embedding vector (no embedding generation)."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        # Build filter (same logic as retrieve)
        query_filter = None
        must_conditions = []
        should_conditions = []
        if tags:
            should_conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
        if memory_type:
            must_conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
        if must_conditions or should_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None,
            )

        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                query_filter=query_filter,
                limit=n_results,
                offset=offset,
                score_threshold=min_similarity if min_similarity and min_similarity > 0 else None,
                with_payload=True,
            ),
        )

        # Parse results (reuse existing parsing logic)
        results = []
        for scored_point in search_results.points:
            try:
                memory = Memory.from_payload(scored_point.payload, scored_point.id)
                relevance_score = float(scored_point.score)
                if min_similarity is not None and relevance_score < min_similarity:
                    continue
                results.append(
                    MemoryQueryResult(
                        memory=memory,
                        similarity_score=relevance_score,
                        debug_info={"score": scored_point.score, "backend": "qdrant", "pre_embedded": True},
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")
        return results

    async def get_memories_batch(self, content_hashes: list[str]) -> list[Memory]:
        """Fetch multiple memories by content hash."""
        if not content_hashes:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchAny

        loop = asyncio.get_event_loop()
        scroll_result = await loop.run_in_executor(
            None,
            lambda: self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="content_hash", match=MatchAny(any=content_hashes))]
                ),
                limit=len(content_hashes),
                with_payload=True,
            ),
        )

        memories = []
        for point in scroll_result[0]:
            try:
                memories.append(Memory.from_payload(point.payload, point.id))
            except Exception as e:
                logger.warning(f"Failed to parse memory in batch fetch: {e}")
        return memories
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_storage_fanout.py -v -x`
Expected: PASS (5 tests)

**Step 5: Lint**

Run: `uv run ruff check src/mcp_memory_service/storage/ tests/unit/test_storage_fanout.py`
Run: `uv run ruff format src/mcp_memory_service/storage/ tests/unit/test_storage_fanout.py`

**Step 6: Commit**

```bash
git add src/mcp_memory_service/storage/base.py src/mcp_memory_service/storage/qdrant_storage.py tests/unit/test_storage_fanout.py
git commit -m "feat: add batch embedding, vector search, and batch fetch to storage"
```

---

### Task 5: Add multi-set RRF merge function

**Files:**
- Modify: `src/mcp_memory_service/utils/hybrid_search.py:199-281`
- Test: `tests/unit/test_hybrid_search.py` (append to existing)

**Step 1: Write the failing tests**

```python
# Append to tests/unit/test_hybrid_search.py (or create test_rrf_multi.py)
# tests/unit/test_rrf_multi.py
"""Tests for multi-set RRF merge."""
import pytest
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from datetime import datetime


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    """Helper to create a Memory with minimal fields."""
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=[],
        memory_type="note",
        created_at=datetime.now().timestamp(),
    )


def _make_result(content_hash: str, score: float) -> MemoryQueryResult:
    """Helper to create a MemoryQueryResult."""
    return MemoryQueryResult(
        memory=_make_memory(content_hash),
        similarity_score=score,
        debug_info={},
    )


def test_combine_results_rrf_multi_exists():
    """combine_results_rrf_multi function exists."""
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    assert callable(combine_results_rrf_multi)


def test_single_result_set_matches_original():
    """Single result set should behave like original combine_results_rrf."""
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    results = [_make_result("hash1", 0.9), _make_result("hash2", 0.7)]
    combined = combine_results_rrf_multi(
        result_sets=[results],
        weights=[1.0],
        tag_matches=[],
    )
    assert len(combined) == 2
    # First result should be hash1 (higher score)
    assert combined[0][0].content_hash == "hash1"
    assert combined[0][1] == 0.9  # cosine score preserved


def test_overlapping_results_ranked_higher():
    """Memories appearing in multiple result sets get higher RRF scores."""
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    # hash_overlap appears in both sets, hash_a and hash_b in one each
    set_a = [_make_result("hash_overlap", 0.8), _make_result("hash_a", 0.7)]
    set_b = [_make_result("hash_overlap", 0.75), _make_result("hash_b", 0.6)]

    combined = combine_results_rrf_multi(
        result_sets=[set_a, set_b],
        weights=[1.0, 1.0],
        tag_matches=[],
    )

    hashes = [m.content_hash for m, _, _ in combined]
    # hash_overlap should be ranked first (appears in both)
    assert hashes[0] == "hash_overlap"
    assert len(combined) == 3  # 3 unique memories


def test_cosine_score_is_max_across_sets():
    """Display score should be the maximum cosine score across result sets."""
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    set_a = [_make_result("hash1", 0.6)]
    set_b = [_make_result("hash1", 0.8)]

    combined = combine_results_rrf_multi(
        result_sets=[set_a, set_b],
        weights=[1.0, 1.0],
        tag_matches=[],
    )

    assert combined[0][1] == 0.8  # Max of 0.6 and 0.8


def test_weights_affect_rrf_ordering():
    """Higher-weighted result sets contribute more to RRF ordering."""
    from mcp_memory_service.utils.hybrid_search import combine_results_rrf_multi

    set_a = [_make_result("hash_a", 0.9)]  # weight 0.1
    set_b = [_make_result("hash_b", 0.5)]  # weight 2.0

    combined = combine_results_rrf_multi(
        result_sets=[set_a, set_b],
        weights=[0.1, 2.0],
        tag_matches=[],
    )

    hashes = [m.content_hash for m, _, _ in combined]
    # hash_b should rank higher due to 20x weight advantage
    assert hashes[0] == "hash_b"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_rrf_multi.py -v -x`
Expected: FAIL — `combine_results_rrf_multi` does not exist

**Step 3: Write implementation**

Add to `src/mcp_memory_service/utils/hybrid_search.py` after `combine_results_rrf` (after line ~281):

```python
def combine_results_rrf_multi(
    result_sets: list[list[MemoryQueryResult]],
    weights: list[float],
    tag_matches: list[Memory],
    k: int = 60,
) -> list[tuple[Memory, float, dict]]:
    """
    Combine N vector search result sets + tag matches using weighted RRF.

    Each result set gets its own RRF ranking, weighted by the corresponding
    weight. Memories appearing in multiple sets accumulate RRF scores (OR
    semantics). The returned display score is the max cosine similarity
    across all sets (not the RRF score).

    Args:
        result_sets: List of ranked result lists from parallel vector searches
        weights: Per-set weight (higher = more influence on final ranking)
        tag_matches: Memories matching extracted tags (unranked)
        k: RRF smoothing constant

    Returns:
        List of (memory, max_cosine_score, debug_info) tuples, sorted by RRF rank desc
    """
    if len(result_sets) != len(weights):
        raise ValueError(f"result_sets ({len(result_sets)}) and weights ({len(weights)}) must have same length")

    rrf_scores: dict[str, float] = {}
    cosine_scores: dict[str, float] = {}  # max cosine across sets
    memories: dict[str, Memory] = {}
    debug: dict[str, dict] = {}

    total_weight = sum(weights) or 1.0

    for set_idx, (results, weight) in enumerate(zip(result_sets, weights)):
        normalized_weight = weight / total_weight
        for rank, result in enumerate(results, start=1):
            content_hash = result.memory.content_hash
            contribution = normalized_weight * rrf_score(rank, k)

            if content_hash not in memories:
                memories[content_hash] = result.memory
                rrf_scores[content_hash] = 0.0
                cosine_scores[content_hash] = 0.0
                debug[content_hash] = {
                    "vector_sets_matched": 0,
                    "vector_scores": {},
                    "tag_boost": 0.0,
                    "tag_matches": [],
                }

            rrf_scores[content_hash] += contribution
            cosine_scores[content_hash] = max(cosine_scores[content_hash], result.similarity_score)
            debug[content_hash]["vector_sets_matched"] += 1
            debug[content_hash]["vector_scores"][f"set_{set_idx}"] = result.similarity_score

    # Process tag matches (same logic as original)
    tag_weight = max(1.0 - (sum(weights) / total_weight), 0.1)  # Remaining weight for tags
    tag_rrf = rrf_score(1, k)
    tag_contribution = tag_weight * tag_rrf

    for memory in tag_matches:
        content_hash = memory.content_hash
        if content_hash in rrf_scores:
            rrf_scores[content_hash] += tag_contribution
            debug[content_hash]["tag_boost"] = tag_contribution
            debug[content_hash]["tag_matches"].append("matched")
        else:
            memories[content_hash] = memory
            rrf_scores[content_hash] = tag_contribution
            cosine_scores[content_hash] = TAG_ONLY_BASE_SCORE
            debug[content_hash] = {
                "vector_sets_matched": 0,
                "vector_scores": {},
                "tag_boost": tag_contribution,
                "tag_matches": ["matched"],
            }

    # Build final results
    results_list: list[tuple[Memory, float, dict]] = []
    for content_hash in rrf_scores:
        info = debug[content_hash]
        display_score = cosine_scores[content_hash]
        info["final_score"] = display_score
        info["rrf_score"] = rrf_scores[content_hash]
        results_list.append((memories[content_hash], display_score, info))

    results_list.sort(key=lambda x: x[2]["rrf_score"], reverse=True)
    return results_list
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_rrf_multi.py -v -x`
Expected: PASS (5 tests)

**Step 5: Run full hybrid search tests**

Run: `uv run pytest tests/unit/test_hybrid_search.py tests/unit/test_rrf_multi.py -v -x`
Expected: All pass (existing + new)

**Step 6: Commit**

```bash
git add src/mcp_memory_service/utils/hybrid_search.py tests/unit/test_rrf_multi.py
git commit -m "feat: add multi-set RRF merge for parallel fan-out"
```

---

### Task 6: Integrate fan-out into retrieve_memories hybrid path

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:1095-1160`
- Test: `tests/unit/test_fanout_integration.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fanout_integration.py
"""Integration test: retrieve_memories uses fan-out when intent is enabled."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from datetime import datetime


def _make_result(content_hash: str, score: float) -> MemoryQueryResult:
    return MemoryQueryResult(
        memory=Memory(
            content=f"content for {content_hash}",
            content_hash=content_hash,
            tags=["test"],
            memory_type="note",
            created_at=datetime.now().timestamp(),
        ),
        similarity_score=score,
        debug_info={},
    )


@pytest.mark.asyncio
async def test_fanout_calls_batch_embed_and_search_by_vector():
    """When intent extracts multiple concepts, uses batched embed + parallel search_by_vector."""
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    # Create service with mocked storage
    mock_storage = AsyncMock()
    mock_storage.count_all_memories = AsyncMock(return_value=1000)
    mock_storage.get_all_tags = AsyncMock(return_value={"dream", "openclaw"})
    mock_storage.search_by_tags = AsyncMock(return_value=[])
    mock_storage.generate_embeddings_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768])
    mock_storage.search_by_vector = AsyncMock(return_value=[_make_result("hash1", 0.7)])

    service = MemoryService(storage=mock_storage)

    # Mock the analyzer to return known concepts
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
        mock_settings.intent.min_query_tokens = 3
        mock_settings.intent.max_sub_queries = 4
        mock_settings.intent.graph_inject = False
        mock_settings.intent.llm_rerank = False
        mock_settings.hybrid_search.hybrid_alpha = None
        mock_settings.hybrid_search.recency_decay = 0.0
        mock_settings.hybrid_search.temporal_decay_lambda = 0.0
        mock_settings.falkordb.spreading_activation_boost = 0.0
        mock_settings.falkordb.hebbian_boost = 0.0
        mock_settings.salience.enabled = False
        mock_settings.spaced_repetition.enabled = False
        mock_settings.encoding_context.enabled = False

        result = await service.retrieve_memories(
            query="dream cycle OpenClaw consolidation",
            page=1,
            page_size=10,
        )

    # Verify: batch embed was called with all sub-queries
    mock_storage.generate_embeddings_batch.assert_called_once()
    batch_call_texts = mock_storage.generate_embeddings_batch.call_args[0][0]
    assert len(batch_call_texts) == 3  # 2 concepts + original

    # Verify: search_by_vector called for each embedding (parallel)
    assert mock_storage.search_by_vector.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fanout_integration.py -v -x`
Expected: FAIL — service doesn't call generate_embeddings_batch yet

**Step 3: Write implementation**

Modify `src/mcp_memory_service/services/memory_service.py`. This is the core integration.

At the top of the file, add import (around line ~45):

```python
from ..utils.query_intent import get_analyzer
```

In the `retrieve_memories` method, replace the hybrid fetch block (lines ~1130-1155) with the fan-out logic. The new code goes in place of the existing single vector + tag fetch:

```python
            # ── Fan-out: concept extraction + parallel search ──────────────
            intent_result = None
            if settings.intent.enabled:
                try:
                    analyzer = get_analyzer(
                        model_name=settings.intent.spacy_model,
                        max_sub_queries=settings.intent.max_sub_queries,
                        min_query_tokens=settings.intent.min_query_tokens,
                    )
                    intent_result = analyzer.analyze(query)
                except Exception as e:
                    logger.warning(f"Intent analysis failed (non-fatal): {e}")

            if intent_result and len(intent_result.sub_queries) > 1:
                # Multi-vector fan-out path
                sub_queries = intent_result.sub_queries

                # Stage 2: Batched embedding (single forward pass)
                embeddings = await self.storage.generate_embeddings_batch(sub_queries)

                # Stage 3: Parallel fan-out (asyncio.gather)
                search_tasks = [
                    self.storage.search_by_vector(
                        embedding=emb,
                        n_results=fetch_size,
                        memory_type=memory_type,
                        min_similarity=0.0,
                        offset=0,
                    )
                    for emb in embeddings
                ]
                tag_task = self.storage.search_by_tags(
                    tags=keywords, match_all=False, limit=fetch_size
                )

                all_results = await asyncio.gather(*search_tasks, tag_task)
                vector_result_sets = list(all_results[:-1])
                tag_matches = all_results[-1]

                # Weights: original query gets 1.5x, concept sub-queries get 1.0x
                weights = []
                for sq in sub_queries:
                    if sq == intent_result.original_query:
                        weights.append(1.5)
                    else:
                        weights.append(1.0)

                combined = combine_results_rrf_multi(
                    result_sets=vector_result_sets,
                    weights=weights,
                    tag_matches=tag_matches,
                    k=60,
                )
            else:
                # Single-vector path (existing behavior)
                vector_task = self.storage.retrieve(
                    query=query,
                    n_results=fetch_size,
                    tags=None,
                    memory_type=memory_type,
                    min_similarity=0.0,
                    offset=0,
                )
                tag_task = self.storage.search_by_tags(
                    tags=keywords,
                    match_all=False,
                    limit=fetch_size,
                )
                vector_results, tag_matches = await asyncio.gather(vector_task, tag_task)
                combined = combine_results_rrf(vector_results, tag_matches, alpha)
```

Also add import for `combine_results_rrf_multi` at the top:

```python
from ..utils.hybrid_search import combine_results_rrf, combine_results_rrf_multi, ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_fanout_integration.py -v -x`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py
git add tests/unit/test_fanout_integration.py
git commit -m "feat: integrate multi-vector fan-out into hybrid search pipeline"
```

---

### Task 7: Add graph neighbor injection

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (in hybrid path, after RRF merge)
- Test: `tests/unit/test_graph_injection.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_graph_injection.py
"""Tests for graph neighbor injection into search results."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_memory_service.models.memory import Memory
from datetime import datetime


def _make_memory(content_hash: str, content: str = "test") -> Memory:
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=["test"],
        memory_type="note",
        created_at=datetime.now().timestamp(),
    )


@pytest.mark.asyncio
async def test_graph_injection_adds_new_candidates():
    """Graph injection fetches neighbor memories not in result set and injects them."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    mock_graph = AsyncMock()

    # spreading_activation returns hash_neighbor (not in results)
    mock_graph.spreading_activation = AsyncMock(return_value={"hash_neighbor": 0.8})
    mock_graph.hebbian_boosts_within = AsyncMock(return_value={})

    # Batch fetch returns the neighbor memory
    neighbor = _make_memory("hash_neighbor", "neighbor content")
    mock_storage.get_memories_batch = AsyncMock(return_value=[neighbor])

    service = MemoryService(storage=mock_storage, graph_client=mock_graph)

    # Simulate: combined has hash_a, graph finds hash_neighbor
    existing_combined = [
        (_make_memory("hash_a"), 0.7, {"source": "vector"}),
    ]

    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 2  # hash_a + hash_neighbor
    injected_hashes = {m.content_hash for m, _, _ in result}
    assert "hash_neighbor" in injected_hashes


@pytest.mark.asyncio
async def test_graph_injection_skips_existing_results():
    """Graph injection does not duplicate memories already in results."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    mock_graph = AsyncMock()

    # spreading_activation returns hash_a which is already in results
    mock_graph.spreading_activation = AsyncMock(return_value={"hash_a": 0.9})
    mock_graph.hebbian_boosts_within = AsyncMock(return_value={})

    service = MemoryService(storage=mock_storage, graph_client=mock_graph)

    existing_combined = [
        (_make_memory("hash_a"), 0.7, {"source": "vector"}),
    ]

    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 1  # No injection, hash_a already present
    mock_storage.get_memories_batch.assert_not_called()


@pytest.mark.asyncio
async def test_graph_injection_noop_when_no_graph():
    """Graph injection is a no-op when graph layer is disabled."""
    from mcp_memory_service.services.memory_service import MemoryService

    mock_storage = AsyncMock()
    service = MemoryService(storage=mock_storage, graph_client=None)

    existing_combined = [(_make_memory("hash_a"), 0.7, {})]
    result = await service._inject_graph_neighbors(
        combined=existing_combined,
        seed_hashes=["hash_a"],
        inject_limit=10,
        min_activation=0.05,
    )

    assert len(result) == 1  # Unchanged
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_graph_injection.py -v -x`
Expected: FAIL — `_inject_graph_neighbors` does not exist

**Step 3: Write implementation**

Add to `MemoryService` in `src/mcp_memory_service/services/memory_service.py` (after `_compute_hebbian_boosts`):

```python
    async def _inject_graph_neighbors(
        self,
        combined: list[tuple[Memory, float, dict]],
        seed_hashes: list[str],
        inject_limit: int = 10,
        min_activation: float = 0.05,
    ) -> list[tuple[Memory, float, dict]]:
        """
        Inject graph neighbors into the candidate pool.

        Runs spreading activation from seed hashes, identifies activated
        neighbors NOT already in the result set, fetches their full Memory
        objects, and appends them to the candidate pool.

        Non-blocking, non-fatal — graph failures don't affect results.

        Args:
            combined: Current candidate pool (mutated in-place if injection occurs)
            seed_hashes: Content hashes to use as activation seeds
            inject_limit: Maximum neighbors to inject
            min_activation: Minimum activation score for injection

        Returns:
            Updated candidate pool with injected neighbors
        """
        if self._graph is None or not seed_hashes:
            return combined

        try:
            graph_activation = await self._graph.spreading_activation(
                seed_hashes=seed_hashes[:5],  # Top 5 seeds
                max_hops=2,
                decay_factor=0.5,
                min_activation=min_activation,
            )

            if not graph_activation:
                return combined

            # Find neighbors NOT already in the result set
            result_hashes = {m.content_hash for m, _, _ in combined}
            neighbor_hashes = [
                h for h, score in sorted(graph_activation.items(), key=lambda x: x[1], reverse=True)
                if h not in result_hashes and score >= min_activation
            ][:inject_limit]

            if not neighbor_hashes:
                return combined

            # Batch fetch neighbor memories
            neighbors = await self.storage.get_memories_batch(neighbor_hashes)

            for memory in neighbors:
                activation = graph_activation.get(memory.content_hash, 0.0)
                combined.append((
                    memory,
                    activation,  # Use activation as initial score
                    {"source": "graph_injection", "activation": activation},
                ))

            logger.debug(f"Graph injection: added {len(neighbors)} neighbors from {len(seed_hashes)} seeds")

        except Exception as e:
            logger.warning(f"Graph injection failed (non-fatal): {e}")

        return combined
```

Then wire it into the hybrid path in `retrieve_memories`, after the RRF merge and before the existing boost pipeline:

```python
            # ── Graph injection: add associatively-connected neighbors ──
            if settings.intent.graph_inject and self._graph is not None:
                all_hashes = [m.content_hash for m, _, _ in combined]
                combined = await self._inject_graph_neighbors(
                    combined=combined,
                    seed_hashes=all_hashes[:5],
                    inject_limit=settings.intent.graph_inject_limit,
                    min_activation=settings.intent.graph_inject_min_activation,
                )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_graph_injection.py -v -x`
Expected: PASS (3 tests)

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_graph_injection.py
git commit -m "feat: add graph neighbor injection to search pipeline"
```

---

### Task 8: Add optional LLM re-ranking

**Files:**
- Create: `src/mcp_memory_service/utils/llm_reranker.py`
- Test: `tests/unit/test_llm_reranker.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_llm_reranker.py
"""Tests for optional LLM re-ranking of search results."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_anthropic_reranker_exists():
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    assert AnthropicReranker is not None


@pytest.mark.asyncio
async def test_reranker_returns_reordered_results():
    """Reranker calls Anthropic API and returns relevance-scored results."""
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text='[{"hash": "hash_b", "score": 0.9}, {"hash": "hash_a", "score": 0.3}]')]
    ))

    reranker = AnthropicReranker(client=mock_client, model="claude-haiku-4-5-20251001", timeout_ms=2000)
    scores = await reranker.rerank(
        query="find dream cycle",
        candidates=[
            {"content_hash": "hash_a", "summary": "email briefing"},
            {"content_hash": "hash_b", "summary": "dream cycle at 3AM"},
        ],
    )

    assert len(scores) == 2
    # hash_b should score higher
    score_map = dict(scores)
    assert score_map["hash_b"] > score_map["hash_a"]


@pytest.mark.asyncio
async def test_reranker_nonfatal_on_error():
    """Reranker returns empty list on API error (non-fatal)."""
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

    reranker = AnthropicReranker(client=mock_client, model="test", timeout_ms=500)
    scores = await reranker.rerank(query="test", candidates=[{"content_hash": "h1", "summary": "s1"}])

    assert scores == []  # Non-fatal: empty on error


@pytest.mark.asyncio
async def test_reranker_nonfatal_on_bad_json():
    """Reranker handles malformed JSON response gracefully."""
    from mcp_memory_service.utils.llm_reranker import AnthropicReranker

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="this is not json")]
    ))

    reranker = AnthropicReranker(client=mock_client, model="test", timeout_ms=500)
    scores = await reranker.rerank(query="test", candidates=[{"content_hash": "h1", "summary": "s1"}])

    assert scores == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_reranker.py -v -x`
Expected: FAIL — module does not exist

**Step 3: Write implementation**

```python
# src/mcp_memory_service/utils/llm_reranker.py
"""
Optional LLM-based re-ranking for search results.

Uses Anthropic Haiku via anthropic-lb to re-score search candidates
by inferred relevance to the user's query intent.

Non-fatal: all errors return empty results, letting the existing
ranking pass through unmodified.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMReranker(Protocol):
    """Protocol for pluggable LLM re-rankers."""

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[tuple[str, float]]: ...


class AnthropicReranker:
    """
    Re-rank search results using Anthropic Claude (Haiku).

    Sends query + candidate summaries to Haiku, asks for relevance scores.
    Returns list of (content_hash, relevance_score) tuples.

    Non-fatal: any error returns empty list.
    """

    def __init__(self, client: Any, model: str, timeout_ms: int = 2000):
        self._client = client
        self._model = model
        self._timeout_ms = timeout_ms

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by LLM-inferred relevance."""
        if not candidates:
            return []

        try:
            # Build candidate descriptions
            candidate_text = "\n".join(
                f'- [{c["content_hash"][:12]}]: {c.get("summary", c.get("content", "")[:100])}'
                for c in candidates
            )

            prompt = (
                f"Given this search query: \"{query}\"\n\n"
                f"Rate the relevance of each memory (0.0 = irrelevant, 1.0 = highly relevant):\n\n"
                f"{candidate_text}\n\n"
                f"Return ONLY a JSON array of objects with 'hash' and 'score' fields. "
                f"Use the hash prefixes shown in brackets. Example: "
                f'[{{"hash": "abc123", "score": 0.8}}]'
            )

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            text = response.content[0].text.strip()

            # Extract JSON array (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            scores_data = json.loads(text)

            # Map short hashes back to full hashes
            hash_map = {c["content_hash"][:12]: c["content_hash"] for c in candidates}

            results = []
            for item in scores_data:
                short_hash = item.get("hash", "")
                score = float(item.get("score", 0.0))
                full_hash = hash_map.get(short_hash)
                if full_hash:
                    results.append((full_hash, min(1.0, max(0.0, score))))

            return results

        except Exception as e:
            logger.warning(f"LLM re-ranking failed (non-fatal): {e}")
            return []
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_reranker.py -v -x`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/mcp_memory_service/utils/llm_reranker.py tests/unit/test_llm_reranker.py
git commit -m "feat: add optional Anthropic Haiku LLM re-ranker"
```

---

### Task 9: Add spaCy dependency and Dockerfile update

**Files:**
- Modify: `pyproject.toml`
- Modify: `Dockerfile`

**Step 1: Add spaCy dependency**

```bash
uv add spacy
uv run python -m spacy download en_core_web_sm
uv lock
```

**Step 2: Update Dockerfile**

Add spaCy model download after the pip install stage. Find the existing `RUN uv pip install` block and add after it:

```dockerfile
RUN uv run python -m spacy download en_core_web_sm
```

**Step 3: Verify import works**

```bash
uv run python -c "from mcp_memory_service.utils.query_intent import get_analyzer; a = get_analyzer(); print(a.analyze('dream cycle 3AM OpenClaw consolidation'))"
```

Expected: prints QueryIntentResult with sub_queries

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 5: Lint everything**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock Dockerfile
git commit -m "build: add spacy dependency and model download"
```

---

### Task 10: Add search analytics fields and final integration test

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py` (SearchLog dataclass / dict)
- Test: `tests/unit/test_fanout_e2e.py`

**Step 1: Write the end-to-end test**

```python
# tests/unit/test_fanout_e2e.py
"""End-to-end test: full fan-out pipeline with all stages."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from datetime import datetime


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
        similarity_score=score,
        debug_info={},
    )


@pytest.mark.asyncio
async def test_full_pipeline_returns_wider_results():
    """Multi-concept query returns more results than single-vector would."""
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.utils.query_intent import QueryIntentResult

    mock_storage = AsyncMock()
    mock_storage.count_all_memories = AsyncMock(return_value=5000)
    mock_storage.get_all_tags = AsyncMock(return_value={"dream", "openclaw", "cron"})
    mock_storage.search_by_tags = AsyncMock(return_value=[])
    mock_storage.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768]
    )

    # Each sub-query finds different results (the whole point)
    mock_storage.search_by_vector = AsyncMock(side_effect=[
        [_make_result("hash_dream", 0.7)],    # "dream cycle"
        [_make_result("hash_claw", 0.65)],     # "OpenClaw"
        [_make_result("hash_dream", 0.6), _make_result("hash_consol", 0.55)],  # original
    ])
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

        # Configure all settings
        mock_settings.intent.enabled = True
        mock_settings.intent.min_query_tokens = 3
        mock_settings.intent.max_sub_queries = 4
        mock_settings.intent.graph_inject = False
        mock_settings.intent.llm_rerank = False
        mock_settings.intent.spacy_model = "en_core_web_sm"
        mock_settings.hybrid_search.hybrid_alpha = None
        mock_settings.hybrid_search.recency_decay = 0.0
        mock_settings.hybrid_search.temporal_decay_lambda = 0.0
        mock_settings.hybrid_search.temporal_decay_base = 0.7
        mock_settings.falkordb.spreading_activation_boost = 0.0
        mock_settings.falkordb.hebbian_boost = 0.0
        mock_settings.salience.enabled = False
        mock_settings.spaced_repetition.enabled = False
        mock_settings.encoding_context.enabled = False

        result = await service.retrieve_memories(
            query="dream cycle OpenClaw consolidation",
            page=1,
            page_size=10,
            min_similarity=0.5,
        )

    memories = result.get("memories", [])
    hashes = {m["content_hash"] for m in memories}

    # All three unique memories should be present
    assert "hash_dream" in hashes, "dream cycle result missing"
    assert "hash_claw" in hashes, "OpenClaw result missing"
    assert "hash_consol" in hashes, "consolidation result missing"
    assert len(memories) == 3

    # hash_dream should rank highest (found by 2 of 3 sub-queries)
    assert memories[0]["content_hash"] == "hash_dream"
```

**Step 2: Run test to verify it passes (integration of all previous tasks)**

Run: `uv run pytest tests/unit/test_fanout_e2e.py -v -x`
Expected: PASS

**Step 3: Update search analytics logging**

In `retrieve_memories`, update the search log dict to include intent fields. Find where `_log_search` is called and add:

```python
    "intent_enabled": bool(intent_result and len(intent_result.sub_queries) > 1),
    "concepts_extracted": intent_result.concepts if intent_result else [],
    "sub_queries_count": len(intent_result.sub_queries) if intent_result else 1,
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 5: Final lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_fanout_e2e.py
git commit -m "feat: add search analytics for intent and end-to-end fan-out test"
```

---

### Task 11: Final quality gate and PR

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -x -v
```

Expected: All pass

**Step 2: Lint and format**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: Clean

**Step 3: Push and create PR**

```bash
git push -u origin docs/query-intent-design
```

Create PR against `main` with title: `feat: query intent inference & multi-vector search fan-out`

Reference issue #108.

---

## Summary

| Task | Component | New files | Modified files |
|------|-----------|-----------|----------------|
| 1 | QueryIntentSettings config | `tests/unit/test_config_intent.py` | `config.py` |
| 2 | Lower min_similarity default | `tests/unit/test_min_similarity_default.py` | `mcp_server.py` |
| 3 | SpaCyAnalyzer + FallbackAnalyzer | `utils/query_intent.py`, `tests/unit/test_query_intent.py` | — |
| 4 | Batch embed + vector search + batch fetch | `tests/unit/test_storage_fanout.py` | `base.py`, `qdrant_storage.py` |
| 5 | Multi-set RRF merge | `tests/unit/test_rrf_multi.py` | `hybrid_search.py` |
| 6 | Fan-out integration in retrieve_memories | `tests/unit/test_fanout_integration.py` | `memory_service.py` |
| 7 | Graph neighbor injection | `tests/unit/test_graph_injection.py` | `memory_service.py` |
| 8 | LLM re-ranker (optional) | `utils/llm_reranker.py`, `tests/unit/test_llm_reranker.py` | — |
| 9 | spaCy dependency + Dockerfile | — | `pyproject.toml`, `Dockerfile`, `uv.lock` |
| 10 | Analytics + E2E test | `tests/unit/test_fanout_e2e.py` | `memory_service.py` |
| 11 | Quality gate + PR | — | — |

**Total: 11 tasks, ~8 new files, ~5 modified files, ~40 test cases**
