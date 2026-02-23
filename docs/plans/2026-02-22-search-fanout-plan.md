# Search Fan-Out Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic tag matching (k-NN against cached tag embeddings) and summary vector search (second Qdrant named vector) to the hybrid search pipeline, improving recall for short/fuzzy queries.

**Architecture:** Two independent features that each produce an additional result set fed into the existing `combine_results_rrf_multi` pipeline. Tag embeddings are cached via CacheKit (Redis-backed). Summary vectors require a Qdrant collection migration from unnamed to named vectors.

**Tech Stack:** CacheKit (Redis cache), Qdrant named vectors, numpy (k-NN), asyncio (background tasks)

---

## PR 1: Semantic Tag Matching + CacheKit Integration

### Task 1: Add CacheKit dependency

**Files:**
- Modify: `pyproject.toml` (dependencies section, lines 37-65)

**Step 1: Add cachekit to dependencies**

```bash
uv add cachekit
```

**Step 2: Verify installation**

```bash
uv run python -c "from cachekit import cache; print('CacheKit available')"
```

Expected: `CacheKit available`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add cachekit dependency"
```

---

### Task 2: Add SemanticTagSettings to config

**Files:**
- Modify: `src/mcp_memory_service/config.py:721` (after QueryIntentSettings)
- Test: `tests/unit/test_config_semantic_tags.py`

**Step 1: Write the failing test**

```python
"""Tests for SemanticTagSettings configuration."""

from mcp_memory_service.config import SemanticTagSettings


class TestSemanticTagSettings:
    def test_defaults(self):
        """All defaults should be sane out of the box."""
        s = SemanticTagSettings()
        assert s.enabled is True
        assert s.similarity_threshold == 0.5
        assert s.max_tags == 10
        assert s.rrf_weight == 0.5
        assert s.cache_ttl == 3600

    def test_env_override(self, monkeypatch):
        """Env vars with MCP_SEMANTIC_TAG_ prefix should override defaults."""
        monkeypatch.setenv("MCP_SEMANTIC_TAG_ENABLED", "false")
        monkeypatch.setenv("MCP_SEMANTIC_TAG_SIMILARITY_THRESHOLD", "0.7")
        monkeypatch.setenv("MCP_SEMANTIC_TAG_MAX_TAGS", "5")
        s = SemanticTagSettings()
        assert s.enabled is False
        assert s.similarity_threshold == 0.7
        assert s.max_tags == 5

    def test_threshold_bounds(self):
        """Threshold should be clamped to 0.0-1.0."""
        s = SemanticTagSettings(similarity_threshold=0.0)
        assert s.similarity_threshold == 0.0
        s = SemanticTagSettings(similarity_threshold=1.0)
        assert s.similarity_threshold == 1.0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_config_semantic_tags.py -v
```

Expected: FAIL with `ImportError: cannot import name 'SemanticTagSettings'`

**Step 3: Write minimal implementation**

Add after `QueryIntentSettings` class (line 721) in `config.py`:

```python
class SemanticTagSettings(BaseSettings):
    """Semantic tag matching configuration for k-NN tag embedding search."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_SEMANTIC_TAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable semantic tag matching via embedding k-NN")
    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum cosine similarity to consider a tag match"
    )
    max_tags: int = Field(default=10, ge=1, le=50, description="Maximum semantically matched tags to fan out")
    rrf_weight: float = Field(default=0.5, ge=0.0, le=5.0, description="RRF weight for semantic tag results")
    cache_ttl: int = Field(default=3600, ge=60, description="Tag embedding cache TTL in seconds")
```

Then add field to `Settings` class (after line 847):

```python
    semantic_tag: SemanticTagSettings = Field(default_factory=SemanticTagSettings)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_config_semantic_tags.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp_memory_service/config.py tests/unit/test_config_semantic_tags.py
git commit -m "feat: add SemanticTagSettings configuration"
```

---

### Task 3: Implement tag embedding cache with CacheKit

**Files:**
- Create: `src/mcp_memory_service/utils/tag_embeddings.py`
- Test: `tests/unit/test_tag_embeddings.py`

**Step 1: Write the failing test**

```python
"""Tests for semantic tag matching via embedding k-NN."""

import numpy as np
import pytest
from unittest.mock import AsyncMock

from mcp_memory_service.utils.tag_embeddings import (
    find_semantic_tags,
    build_tag_embedding_index,
)


class TestBuildTagEmbeddingIndex:
    def test_builds_index_from_tags_and_embeddings(self):
        """Index should contain tag names and a normalized embedding matrix."""
        tags = ["python", "redis", "docker"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        index = build_tag_embedding_index(tags, embeddings)
        assert index["tags"] == ["python", "redis", "docker"]
        assert index["matrix"].shape == (3, 3)

    def test_empty_tags_returns_empty_index(self):
        """Empty input should produce an empty index."""
        index = build_tag_embedding_index([], [])
        assert index["tags"] == []
        assert index["matrix"].shape[0] == 0


class TestFindSemanticTags:
    def test_finds_similar_tags(self):
        """Query embedding close to a tag embedding should return that tag."""
        # "python" embedding is [1, 0, 0], query is [0.9, 0.1, 0]
        tags = ["python", "redis", "docker"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [0.9, 0.1, 0.0]

        matches = find_semantic_tags(query_embedding, index, threshold=0.5, max_tags=5)
        assert "python" in matches
        assert "docker" not in matches

    def test_respects_threshold(self):
        """Tags below threshold should be excluded."""
        tags = ["python", "redis"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [0.7, 0.7]  # equidistant

        # High threshold: nothing matches
        matches = find_semantic_tags(query_embedding, index, threshold=0.99, max_tags=5)
        assert len(matches) == 0

    def test_respects_max_tags(self):
        """Should return at most max_tags results."""
        tags = [f"tag{i}" for i in range(20)]
        # All embeddings point in same direction — all will match
        embeddings = [[1.0, 0.0] for _ in range(20)]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [1.0, 0.0]

        matches = find_semantic_tags(query_embedding, index, threshold=0.0, max_tags=5)
        assert len(matches) == 5

    def test_empty_index_returns_empty(self):
        """Empty index should return no matches."""
        index = build_tag_embedding_index([], [])
        matches = find_semantic_tags([1.0, 0.0], index, threshold=0.0, max_tags=5)
        assert matches == []
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_tag_embeddings.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Semantic tag matching via embedding k-NN.

Builds an in-memory index of tag embeddings and finds semantically
similar tags for a given query embedding using cosine similarity.
The index is intended to be cached (via CacheKit) and rebuilt on
tag set changes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_tag_embedding_index(
    tags: list[str],
    embeddings: list[list[float]],
) -> dict:
    """Build a normalised tag embedding index for k-NN search.

    Args:
        tags: Tag names (parallel to embeddings list).
        embeddings: Embedding vectors for each tag.

    Returns:
        Dict with 'tags' (list[str]) and 'matrix' (NDArray, L2-normalised rows).
    """
    if not tags:
        return {"tags": [], "matrix": np.empty((0, 0))}

    matrix = np.array(embeddings, dtype=np.float32)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # guard zero vectors
    matrix = matrix / norms

    return {"tags": list(tags), "matrix": matrix}


def find_semantic_tags(
    query_embedding: list[float] | NDArray,
    index: dict,
    threshold: float = 0.5,
    max_tags: int = 10,
) -> list[str]:
    """Find tags semantically similar to a query embedding.

    Args:
        query_embedding: Query vector (same dimensionality as index).
        index: Output of build_tag_embedding_index().
        threshold: Minimum cosine similarity to include a tag.
        max_tags: Maximum number of tags to return.

    Returns:
        List of matching tag names, sorted by similarity descending.
    """
    if not index["tags"]:
        return []

    query = np.array(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    query = query / norm

    # Cosine similarities via dot product (both sides L2-normalised)
    similarities = index["matrix"] @ query

    # Filter and sort
    mask = similarities >= threshold
    if not mask.any():
        return []

    indices = np.where(mask)[0]
    scores = similarities[indices]
    top_indices = indices[np.argsort(-scores)[:max_tags]]

    return [index["tags"][i] for i in top_indices]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_tag_embeddings.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp_memory_service/utils/tag_embeddings.py tests/unit/test_tag_embeddings.py
git commit -m "feat: add tag embedding k-NN matching"
```

---

### Task 4: Replace _tag_cache with CacheKit

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:97,879-890`
- Test: `tests/unit/test_tag_cache_cachekit.py`

**Step 1: Write the failing test**

```python
"""Tests for CacheKit-backed tag cache replacing manual _tag_cache."""

import pytest
from unittest.mock import AsyncMock, patch

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.get_all_tags.return_value = ["python", "redis", "docker"]
    return storage


@pytest.fixture
def memory_service(mock_storage):
    return MemoryService(storage=mock_storage)


class TestCacheKitTagCache:
    @pytest.mark.asyncio
    async def test_get_cached_tags_returns_tags(self, memory_service, mock_storage):
        """Should return tags from storage."""
        tags = await memory_service._get_cached_tags()
        assert tags == {"python", "redis", "docker"}

    @pytest.mark.asyncio
    async def test_get_cached_tags_calls_storage(self, memory_service, mock_storage):
        """Should call storage.get_all_tags on cache miss."""
        await memory_service._get_cached_tags()
        mock_storage.get_all_tags.assert_called_once()
```

**Step 2: Run test to verify it passes** (existing behavior)

```bash
uv run pytest tests/unit/test_tag_cache_cachekit.py -v
```

Expected: PASS (these tests verify the interface contract, not the cache backend)

**Step 3: Replace _tag_cache implementation**

In `memory_service.py`, replace the manual cache at line 97 and method at lines 879-890:

Remove from `__init__`:
```python
self._tag_cache: tuple[float, set[str]] | None = None
```

Replace `_get_cached_tags` method with:
```python
    async def _get_cached_tags(self) -> set[str]:
        """Get all tags, cached via CacheKit (Redis L2 + in-memory L1).

        Cache is shared across pods via Redis. TTL of 60s matches previous behavior.
        Falls back to in-memory only if Redis is unavailable (CacheKit graceful degradation).
        """
        from cachekit import cache

        @cache.minimal(ttl=60)
        async def _fetch_tags(storage_id: str) -> set[str]:
            all_tags = await self.storage.get_all_tags()
            return set(all_tags)

        # storage_id makes cache key unique per storage instance
        return await _fetch_tags(id(self.storage))
```

> **Note to implementer:** The `@cache.minimal` decorator is defined inside the method to capture `self.storage`. The `storage_id` param ensures different storage instances get different cache keys. CacheKit handles TTL, L1/L2 tiering, and Redis fallback automatically.

**Step 4: Run test to verify it still passes**

```bash
uv run pytest tests/unit/test_tag_cache_cachekit.py -v
```

Expected: PASS

**Step 5: Run full unit suite to check for regressions**

```bash
uv run pytest tests/unit/ -x -q
```

Expected: All pass

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_tag_cache_cachekit.py
git commit -m "refactor: replace manual _tag_cache with CacheKit"
```

---

### Task 5: Integrate semantic tag matching into retrieve_memories

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:199-222` (single-vector path)
- Modify: `src/mcp_memory_service/services/memory_service.py:1446-1493` (fan-out path)
- Test: `tests/unit/test_semantic_tag_search.py`

**Step 1: Write the failing test**

```python
"""Tests for semantic tag matching integration in retrieve_memories."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.models.memory import Memory, MemoryQueryResult


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.get_all_tags.return_value = ["proton-bridge", "imap", "email", "redis", "docker"]
    storage.count.return_value = 100
    storage.count_all_memories.return_value = 100

    # Return empty results for all searches by default
    storage.retrieve.return_value = []
    storage.search_by_tags.return_value = []
    storage.search_by_tag.return_value = []

    # generate_embeddings_batch returns fake embeddings
    storage.generate_embeddings_batch = AsyncMock(
        return_value=[[1.0, 0.0, 0.0] for _ in range(5)]  # one per tag
    )

    return storage


@pytest.fixture
def memory_service(mock_storage):
    return MemoryService(storage=mock_storage)


class TestSemanticTagSearchIntegration:
    @pytest.mark.asyncio
    async def test_semantic_tags_trigger_additional_tag_search(self, memory_service, mock_storage):
        """When semantic tag matching finds tags, search_by_tags should be called with them."""
        # Mock the tag embedding lookup to return "proton-bridge" as a semantic match
        with patch(
            "mcp_memory_service.services.memory_service.find_semantic_tags",
            return_value=["proton-bridge"],
        ), patch(
            "mcp_memory_service.services.memory_service.build_tag_embedding_index",
            return_value={"tags": ["proton-bridge"], "matrix": MagicMock()},
        ):
            await memory_service._single_vector_search(
                query="proton bridge",
                keywords=[],  # no exact keyword matches
                fetch_size=10,
                memory_type=None,
                alpha=0.5,
            )

            # search_by_tags should be called with the semantic matches
            calls = mock_storage.search_by_tags.call_args_list
            assert any("proton-bridge" in call.kwargs.get("tags", []) for call in calls)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_semantic_tag_search.py -v
```

Expected: FAIL

**Step 3: Implement semantic tag matching in _single_vector_search**

Modify `_single_vector_search` (lines 199-222) to add a semantic tag search task:

```python
    async def _single_vector_search(
        self,
        query: str,
        keywords: list[str],
        fetch_size: int,
        memory_type: str | None,
        alpha: float,
    ) -> list[tuple[Memory, float, dict]]:
        """Single-vector search + tag RRF (shared by normal and fan-out fallback paths)."""
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

        # Semantic tag matching: find additional tags via embedding k-NN
        semantic_tag_task = self._search_semantic_tags(query, fetch_size)

        vector_results, tag_matches, semantic_tag_matches = await asyncio.gather(
            vector_task, tag_task, semantic_tag_task
        )

        # Merge exact + semantic tag matches (deduplicate by content_hash)
        all_tag_matches = tag_matches
        if semantic_tag_matches:
            seen = {m.content_hash for m in tag_matches}
            for m in semantic_tag_matches:
                if m.content_hash not in seen:
                    all_tag_matches.append(m)
                    seen.add(m.content_hash)

        return combine_results_rrf(vector_results, all_tag_matches, alpha)
```

Add the `_search_semantic_tags` helper method to `MemoryService`:

```python
    async def _search_semantic_tags(self, query: str, fetch_size: int) -> list[Memory]:
        """Find memories via semantically similar tags (k-NN on tag embeddings).

        Non-fatal: returns empty list on any error.
        """
        if not settings.semantic_tag.enabled:
            return []

        try:
            from mcp_memory_service.utils.tag_embeddings import (
                build_tag_embedding_index,
                find_semantic_tags,
            )

            # Get tag embeddings (CacheKit-cached)
            index = await self._get_tag_embedding_index()
            if not index["tags"]:
                return []

            # Get query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, lambda: self.storage._generate_embedding(query, prompt_name="query")
            )

            # k-NN match
            matched_tags = find_semantic_tags(
                query_embedding,
                index,
                threshold=settings.semantic_tag.similarity_threshold,
                max_tags=settings.semantic_tag.max_tags,
            )

            if not matched_tags:
                return []

            return await self.storage.search_by_tags(
                tags=matched_tags, match_all=False, limit=fetch_size
            )

        except Exception as e:
            logger.warning(f"Semantic tag matching failed (non-fatal): {e}")
            return []

    async def _get_tag_embedding_index(self) -> dict:
        """Get or build the tag embedding index, cached via CacheKit."""
        from cachekit import cache
        from mcp_memory_service.utils.tag_embeddings import build_tag_embedding_index

        @cache.minimal(ttl=settings.semantic_tag.cache_ttl)
        async def _build_index(storage_id: str) -> dict:
            tags = list(await self._get_cached_tags())
            if not tags:
                return {"tags": [], "matrix": []}
            embeddings = await self.storage.generate_embeddings_batch(tags)
            return {"tags": tags, "embeddings": embeddings}

        raw = await _build_index(str(id(self.storage)))

        # Rebuild numpy index from cached data (numpy arrays aren't serialisable)
        if not raw["tags"]:
            return build_tag_embedding_index([], [])
        return build_tag_embedding_index(raw["tags"], raw["embeddings"])
```

> **Note to implementer:** The CacheKit cache stores the raw tags + embeddings (JSON-serialisable lists). The numpy index is rebuilt from cached data on each call — this is ~0.1ms for 1000 tags and avoids numpy serialisation issues. The query embedding reuses storage's `_generate_embedding` which already handles model loading and prompt_name prefixing.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_semantic_tag_search.py -v
```

Expected: PASS

**Step 5: Run full unit suite**

```bash
uv run pytest tests/unit/ -x -q
```

Expected: All pass

**Step 6: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_semantic_tag_search.py
git commit -m "feat: integrate semantic tag matching into hybrid search pipeline"
```

---

### Task 6: Wire semantic tags into fan-out path

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py:1446-1493`

The fan-out path (lines 1446-1493) already uses `asyncio.gather` for parallel searches. Add the semantic tag search as another parallel task and merge its results into `tag_matches` before passing to `combine_results_rrf_multi`.

**Step 1: Modify the fan-out path**

At line 1467 (where `tag_task` is defined), add:

```python
                    semantic_tag_task = self._search_semantic_tags(query, fetch_size)
```

At line 1469 (the asyncio.gather), add `semantic_tag_task`:

```python
                    all_results = await asyncio.gather(*search_tasks, tag_task, semantic_tag_task)
                    vector_result_sets = list(all_results[:-2])
                    tag_matches = all_results[-2]
                    semantic_tag_matches = all_results[-1]
```

After line 1471 (tag_matches assignment), merge semantic matches:

```python
                    # Merge exact + semantic tag matches
                    if semantic_tag_matches:
                        seen = {m.content_hash for m in tag_matches}
                        for m in semantic_tag_matches:
                            if m.content_hash not in seen:
                                tag_matches.append(m)
                                seen.add(m.content_hash)
```

**Step 2: Run full unit suite**

```bash
uv run pytest tests/unit/ -x -q
```

Expected: All pass

**Step 3: Commit**

```bash
git add src/mcp_memory_service/services/memory_service.py
git commit -m "feat: wire semantic tags into intent fan-out path"
```

---

### Task 7: Lint, format, and quality gate

**Step 1: Run quality gates**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/unit/ -x -q
```

Expected: All pass, 0 errors

**Step 2: Fix any issues found**

**Step 3: Final commit if any fixes needed**

```bash
git add -u
git commit -m "chore: lint and format fixes"
```

---

## PR 2: Named Vector Migration + Summary Vector Search

> **Prerequisite:** PR 1 merged (CacheKit available)

### Task 8: Add SummarySearchSettings to config

**Files:**
- Modify: `src/mcp_memory_service/config.py` (after SemanticTagSettings)
- Test: `tests/unit/test_config_summary_search.py`

**Step 1: Write the failing test**

```python
"""Tests for SummarySearchSettings configuration."""

from mcp_memory_service.config import SummarySearchSettings


class TestSummarySearchSettings:
    def test_defaults(self):
        s = SummarySearchSettings()
        assert s.enabled is True
        assert s.rrf_weight == 0.8

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MCP_SUMMARY_SEARCH_ENABLED", "false")
        monkeypatch.setenv("MCP_SUMMARY_SEARCH_RRF_WEIGHT", "0.3")
        s = SummarySearchSettings()
        assert s.enabled is False
        assert s.rrf_weight == 0.3
```

**Step 2: Run to verify fail, then implement**

```python
class SummarySearchSettings(BaseSettings):
    """Summary vector search configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_SUMMARY_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable summary vector search")
    rrf_weight: float = Field(default=0.8, ge=0.0, le=5.0, description="RRF weight for summary vector results")
```

Add to Settings class:
```python
    summary_search: SummarySearchSettings = Field(default_factory=SummarySearchSettings)
```

**Step 3: Verify pass, commit**

---

### Task 9: Migrate QdrantStorage to named vectors

**Files:**
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:348-413` (collection creation)
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:646-724` (store — PointStruct creation)
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py:732-868` (retrieve — query)
- Test: `tests/unit/test_named_vectors.py`

**Key changes:**

1. **Collection creation** (line 375): Change from `VectorParams(size=..., distance=Distance.COSINE)` to:
   ```python
   vectors_config={
       "content": VectorParams(size=self._vector_size, distance=Distance.COSINE),
       "summary": VectorParams(size=self._vector_size, distance=Distance.COSINE),
   }
   ```

2. **Store** (PointStruct creation): Change `vector=memory.embedding` to:
   ```python
   vector={
       "content": memory.embedding,
       "summary": memory.summary_embedding or memory.embedding,  # fallback to content if no summary
   }
   ```

3. **Retrieve** (query_points): Change to use `using="content"` named vector parameter.

4. **Add new method** `search_by_summary_vector`:
   ```python
   async def search_by_summary_vector(
       self, query_embedding: list[float], n_results: int = 10, min_similarity: float | None = None
   ) -> list[MemoryQueryResult]:
       """Search against summary named vectors."""
       # Same as retrieve but uses using="summary"
   ```

> **Note to implementer:** The Memory model needs a `summary_embedding` field added. On store, generate summary embedding alongside content embedding. The `_generate_embedding` method handles the model loading.

---

### Task 10: Add CLI migration command

**Files:**
- Modify: `src/mcp_memory_service/cli/main.py` (add `migrate-vectors` command)
- Create: `src/mcp_memory_service/cli/migrate.py` (migration logic)

**Step 1: Implement migration command**

```python
@cli.command("migrate-vectors")
@click.option("--batch-size", default=100, help="Points per batch")
@click.option("--dry-run", is_flag=True, help="Preview without migrating")
def migrate_vectors(batch_size: int, dry_run: bool):
    """Migrate from unnamed to named vectors (content + summary)."""
    asyncio.run(_migrate_vectors(batch_size, dry_run))
```

Migration logic:
1. Connect to existing collection, read all points via scroll
2. Create v2 collection with named vectors config
3. For each batch: extract content vector, generate summary embedding, upsert to v2
4. Verify counts match
5. Print summary

---

### Task 11: Integrate summary vector search into retrieve_memories

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py`

Add summary vector search as another result set in both `_single_vector_search` and the fan-out path, following the same pattern as semantic tag matching (parallel asyncio.gather, merge into RRF with configurable weight).

---

### Task 12: Quality gate and PR

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/unit/ -x -q
```

---

## PR 3: Backfill + Async Task Infrastructure

> **Prerequisite:** PR 2 merged

### Task 13: Add asyncio background task manager

Lightweight task manager for in-process background work. Pattern from `graph/queue.py` — CacheKit distributed lock for multi-pod safety.

### Task 14: Implement summary vector backfill

Background task that scrolls all existing memories, generates summary embeddings, and upserts to the summary named vector. Progress tracked, resumable.

### Task 15: Auto-detect and trigger backfill on startup

On server startup, check if summary vectors exist. If not, schedule backfill as background task with CacheKit distributed lock.
