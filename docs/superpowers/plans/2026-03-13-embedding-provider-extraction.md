# Embedding Provider Extraction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract embedding generation from QdrantStorage into a pluggable `EmbeddingProvider` protocol so API and embedding services scale independently.

**Architecture:** Ports & adapters. `EmbeddingProvider` protocol with two adapters: `LocalProvider` (in-process SentenceTransformer, backward compat) and `OpenAICompatAdapter` (HTTP to TEI/vLLM/Ollama/OpenAI). `CachedEmbeddingProvider` decorator wraps either with CacheKit L1/L2. Single instance shared by both `QdrantStorage` and `MemoryService`.

**Tech Stack:** Python 3.13, pydantic-settings, CacheKit, httpx (already a dependency), SentenceTransformer (LocalProvider only)

**Spec:** `docs/superpowers/specs/2026-03-13-horizontal-scale-out-design.md`

**Test runner:** `pytest tests/ -x -m "not slow"` (asyncio_mode=auto, 300s timeout)

**Lint:** `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `src/mcp_memory_service/embedding/__init__.py` | Package init, re-exports |
| `src/mcp_memory_service/embedding/protocol.py` | `EmbeddingProvider` protocol definition |
| `src/mcp_memory_service/embedding/local.py` | `LocalProvider` — wraps SentenceTransformer |
| `src/mcp_memory_service/embedding/http.py` | `OpenAICompatAdapter` — HTTP to /v1/embeddings |
| `src/mcp_memory_service/embedding/cached.py` | `CachedEmbeddingProvider` — CacheKit L1/L2 wrapper |
| `src/mcp_memory_service/embedding/factory.py` | `create_embedding_provider()` factory |
| `tests/unit/test_embedding_protocol.py` | Protocol + LocalProvider tests |
| `tests/unit/test_embedding_http.py` | OpenAICompatAdapter tests |
| `tests/unit/test_embedding_cached.py` | CachedEmbeddingProvider tests |
| `tests/unit/test_embedding_factory.py` | Factory tests |
| `docker-compose.dev.yml` | Two-service dev deployment example |

### Modified Files

| File | What Changes |
|------|-------------|
| `src/mcp_memory_service/config.py` | Add `EmbeddingSettings` class |
| `src/mcp_memory_service/storage/qdrant_storage.py` | Accept `EmbeddingProvider`, remove model loading |
| `src/mcp_memory_service/storage/base.py` | Deprecate `generate_embeddings_batch()` |
| `src/mcp_memory_service/services/memory_service.py` | Accept `EmbeddingProvider`, simplify `_get_embeddings()` |
| `src/mcp_memory_service/shared_storage.py` | Create provider in factory, pass to storage + service |
| `src/mcp_memory_service/mcp_server.py` | Pass provider through lifecycle |
| `src/mcp_memory_service/unified_server.py` | Uses `get_shared_storage()` — gets provider transitively |
| `src/mcp_memory_service/web/app.py` | Pass provider through lifespan |
| `Dockerfile` | Multi-stage: `base`, `full`, `api` targets |
| `.github/workflows/qa.yml` | Build both Docker targets |
| `tests/unit/conftest.py` | Add `mock_embedding_provider` fixture |

### Deleted Files (Phase 5)

| File | Why |
|------|-----|
| `src/mcp_memory_service/web/sse.py` | Stateful, unused |
| `src/mcp_memory_service/web/write_queue.py` | Replaced by semaphore |

---

## Chunk 1: EmbeddingProvider Protocol + LocalProvider (Phase 1)

This is the hard phase. Every other phase depends on this being right.

### Task 1.1: Define the EmbeddingProvider Protocol

**Files:**
- Create: `src/mcp_memory_service/embedding/__init__.py`
- Create: `src/mcp_memory_service/embedding/protocol.py`
- Test: `tests/unit/test_embedding_protocol.py`

- [ ] **Step 1: Write the protocol test**

```python
# tests/unit/test_embedding_protocol.py
"""Tests for EmbeddingProvider protocol."""
import pytest
from mcp_memory_service.embedding.protocol import EmbeddingProvider


class FakeProvider:
    """Minimal implementation for protocol conformance testing."""

    async def embed_batch(self, texts: list[str], prompt_name: str = "query") -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return "fake-model"


def test_fake_provider_is_embedding_provider():
    """A class implementing the protocol methods satisfies runtime_checkable."""
    provider = FakeProvider()
    assert isinstance(provider, EmbeddingProvider)


def test_dimensions_property():
    provider = FakeProvider()
    assert provider.dimensions == 768


def test_model_name_property():
    provider = FakeProvider()
    assert provider.model_name == "fake-model"


@pytest.mark.asyncio
async def test_embed_batch_returns_correct_shape():
    provider = FakeProvider()
    result = await provider.embed_batch(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 768


@pytest.mark.asyncio
async def test_embed_batch_prompt_name_default():
    """Default prompt_name is 'query'."""
    provider = FakeProvider()
    # Should not raise — default kwarg works
    result = await provider.embed_batch(["test"])
    assert len(result) == 1


@pytest.mark.asyncio
async def test_embed_batch_passage_prompt():
    """passage prompt_name is accepted."""
    provider = FakeProvider()
    result = await provider.embed_batch(["test"], prompt_name="passage")
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedding_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_memory_service.embedding'`

- [ ] **Step 3: Write the protocol**

```python
# src/mcp_memory_service/embedding/__init__.py
"""Embedding provider protocol and adapters."""
from .protocol import EmbeddingProvider

__all__ = ["EmbeddingProvider"]
```

```python
# src/mcp_memory_service/embedding/protocol.py
"""EmbeddingProvider protocol — ports & adapters boundary for embedding generation."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural typing protocol for embedding generation.

    Implementations: LocalProvider (in-process), OpenAICompatAdapter (HTTP),
    CachedEmbeddingProvider (L1/L2 cache wrapper).
    """

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            prompt_name: "query" for search queries, "passage" for stored documents.
                         Adapters translate to model-specific prefixes.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Embedding dimensionality (e.g., 768 for nomic-embed-text-v1.5)."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier. Used in cache namespace for auto-invalidation."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedding_protocol.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
uv run ruff format src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
git add src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
git commit -m "feat: define EmbeddingProvider protocol"
```

---

### Task 1.2: Implement LocalProvider

**Files:**
- Create: `src/mcp_memory_service/embedding/local.py`
- Modify: `src/mcp_memory_service/embedding/__init__.py`
- Test: `tests/unit/test_embedding_protocol.py` (extend)

**Context:** `LocalProvider` wraps the existing SentenceTransformer code from `qdrant_storage.py` lines 209-226 (`_ensure_model_loaded`) and lines 685-753 (`_generate_embedding`). It must handle:
- Thread-safe lazy model loading (double-checked locking)
- CUDA capability detection (GTX 1060 → CPU fallback, see `utils/system_detection.py`)
- prompt_name translation for sentence-transformers (v2.2.2+ with `model.prompts`, fallback to manual prefix)
- `asyncio.get_running_loop().run_in_executor()` for non-blocking model.encode()

- [ ] **Step 1: Write LocalProvider tests**

```python
# Append to tests/unit/test_embedding_protocol.py
import asyncio
from unittest.mock import MagicMock, patch

from mcp_memory_service.embedding.local import LocalProvider


class TestLocalProvider:
    """Tests for LocalProvider wrapping SentenceTransformer."""

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self):
        """LocalProvider is an EmbeddingProvider."""
        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [[0.1] * 768]
            mock_st.return_value = mock_model

            provider = LocalProvider(model_name="test-model")
            assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embed_batch_calls_model_encode(self):
        """embed_batch delegates to model.encode via executor."""
        import numpy as np

        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
            mock_model.prompts = {"query": "search_query: ", "passage": "search_document: "}
            mock_st.return_value = mock_model

            provider = LocalProvider(model_name="test-model")
            result = await provider.embed_batch(["hello", "world"], prompt_name="query")

            assert len(result) == 2
            assert len(result[0]) == 768
            mock_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_dimensions_from_model(self):
        """dimensions property reflects actual model dimension."""
        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            provider = LocalProvider(model_name="small-model")
            assert provider.dimensions == 384

    @pytest.mark.asyncio
    async def test_model_loaded_once(self):
        """Model is loaded lazily on first call, not re-loaded."""
        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [[0.1] * 768]
            mock_st.return_value = mock_model

            provider = LocalProvider(model_name="test-model")
            # First access triggers load
            _ = provider.dimensions
            # Second access should not re-load
            _ = provider.dimensions
            mock_st.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_name_fallback_no_prompts(self):
        """Models without .prompts attribute fall back to manual prefix."""
        import numpy as np

        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock(spec=[])  # no .prompts attribute
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([[0.1] * 768])
            del mock_model.prompts  # ensure AttributeError on access
            mock_st.return_value = mock_model

            provider = LocalProvider(model_name="test-model")
            result = await provider.embed_batch(["hello"], prompt_name="query")
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_embed_batch_empty_input(self):
        """Empty input returns empty list without loading model."""
        provider = LocalProvider(model_name="test-model")
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_model_load_failure_propagates(self):
        """Model loading failure propagates with useful error."""
        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_st.side_effect = OSError("Model not found: bad-model")

            provider = LocalProvider(model_name="bad-model")
            with pytest.raises(OSError, match="Model not found"):
                await provider.embed_batch(["hello"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedding_protocol.py::TestLocalProvider -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_memory_service.embedding.local'`

- [ ] **Step 3: Implement LocalProvider**

Extract from `qdrant_storage.py` lines 209-226 and 685-753. The key code to lift:

**IMPORTANT mock compatibility note:** The lazy import `from sentence_transformers import SentenceTransformer` happens inside `_ensure_loaded`. To make `patch("mcp_memory_service.embedding.local.SentenceTransformer")` work, we do an eager `import sentence_transformers as _st` at module level (guarded with try/except for the api target where it's absent), then reference `_st.SentenceTransformer` inside `_ensure_loaded`. The module-level patch target replaces it cleanly.

```python
# src/mcp_memory_service/embedding/local.py
"""LocalProvider — in-process SentenceTransformer embedding generation."""
import asyncio
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Module-level import for mockability. Absent in api Docker target.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

# Prompt name mappings for instruction-tuned models
_PROMPT_PREFIXES: dict[str, dict[str, str]] = {
    "query": {"nomic": "search_query: ", "e5": "query: "},
    "passage": {"nomic": "search_document: ", "e5": "passage: "},
}


class LocalProvider:
    """EmbeddingProvider backed by in-process SentenceTransformer.

    Thread-safe lazy model loading. Delegates encode() to asyncio executor
    to avoid blocking the event loop.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: object | None = None
        self._lock = threading.Lock()
        self._dimensions: int | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model with double-checked locking."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Use the 'full' Docker target or install it: pip install sentence-transformers"
                )

            device = self._device
            if device is None:
                from ..utils.system_detection import get_torch_device
                device = str(get_torch_device())

            logger.info("Loading embedding model %s on %s", self._model_name, device)
            self._model = SentenceTransformer(self._model_name, device=device)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info("Model loaded: %s (%d dimensions)", self._model_name, self._dimensions)

    @property
    def dimensions(self) -> int:
        self._ensure_loaded()
        return self._dimensions  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode_sync(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        """Synchronous encode — called in executor."""
        self._ensure_loaded()
        model = self._model

        # Try native prompt_name support (sentence-transformers >= 2.2.2)
        try:
            if hasattr(model, "prompts") and model.prompts:
                result = model.encode(texts, prompt_name=prompt_name, convert_to_tensor=False)
            else:
                raise AttributeError("no prompts")
        except (TypeError, AttributeError):
            # Fallback: manual prefix for older sentence-transformers
            prefix = self._get_manual_prefix(prompt_name)
            if prefix:
                prefixed = [f"{prefix}{t}" for t in texts]
                result = model.encode(prefixed, convert_to_tensor=False)
            else:
                result = model.encode(texts, convert_to_tensor=False)

        # Convert numpy to Python lists
        if isinstance(result, np.ndarray):
            return result.tolist()
        return [r.tolist() if isinstance(r, np.ndarray) else list(r) for r in result]

    def _get_manual_prefix(self, prompt_name: str) -> str | None:
        """Get manual prefix for models without native prompt support."""
        model_lower = self._model_name.lower()
        prefixes = _PROMPT_PREFIXES.get(prompt_name, {})
        for family, prefix in prefixes.items():
            if family in model_lower:
                return prefix
        return None

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, prompt_name)
```

Update `__init__.py`:
```python
# src/mcp_memory_service/embedding/__init__.py
"""Embedding provider protocol and adapters."""
from .local import LocalProvider
from .protocol import EmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalProvider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedding_protocol.py -v`
Expected: 13 PASSED (6 protocol + 7 LocalProvider)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
uv run ruff format src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
git add src/mcp_memory_service/embedding/ tests/unit/test_embedding_protocol.py
git commit -m "feat: implement LocalProvider wrapping SentenceTransformer"
```

---

### Task 1.3: Add EmbeddingSettings to Config

**Files:**
- Modify: `src/mcp_memory_service/config.py`
- Test: `tests/unit/test_embedding_factory.py` (config portion)

- [ ] **Step 1: Write config test**

```python
# tests/unit/test_embedding_factory.py
"""Tests for embedding provider factory and configuration."""
import pytest


class TestEmbeddingSettings:
    """Test EmbeddingSettings pydantic model."""

    def test_default_provider_is_local(self):
        from mcp_memory_service.config import settings
        assert settings.embedding.provider == "local"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_EMBEDDING_PROVIDER", "openai_compat")
        # Force re-read — pydantic-settings reads env at instantiation
        from mcp_memory_service.config import EmbeddingSettings
        s = EmbeddingSettings()
        assert s.provider == "openai_compat"

    def test_url_optional_for_local(self):
        from mcp_memory_service.config import EmbeddingSettings
        s = EmbeddingSettings(provider="local")
        assert s.url is None

    def test_timeout_default(self):
        from mcp_memory_service.config import EmbeddingSettings
        s = EmbeddingSettings()
        assert s.timeout == 30

    def test_max_batch_default(self):
        from mcp_memory_service.config import EmbeddingSettings
        s = EmbeddingSettings()
        assert s.max_batch == 64

    def test_http_url_logs_warning(self, monkeypatch, caplog):
        """http:// URLs should emit a warning about plaintext."""
        from mcp_memory_service.config import EmbeddingSettings
        import logging
        with caplog.at_level(logging.WARNING):
            s = EmbeddingSettings(provider="openai_compat", url="http://localhost:8080")
        assert s.url == "http://localhost:8080"
        # Warning is emitted by the validator or factory — check at factory level
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedding_factory.py::TestEmbeddingSettings -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'embedding'` or `ImportError`

- [ ] **Step 3: Add EmbeddingSettings to config.py**

Add to `src/mcp_memory_service/config.py` (after the existing settings classes):

```python
class EmbeddingSettings(BaseSettings):
    """Embedding provider configuration."""

    model_config = SettingsConfigDict(env_prefix="MCP_EMBEDDING_")

    provider: str = "local"  # local | openai_compat
    url: str | None = None  # Base URL for HTTP provider
    timeout: int = 30  # Request timeout seconds
    max_batch: int = 64  # Max texts per request
    tls_verify: bool = True  # TLS certificate verification
    api_key: SecretStr | None = None  # For managed providers
    prompt_name_map: dict[str, dict[str, str]] | None = None  # Custom prompt mappings
```

Add `embedding: EmbeddingSettings` field to the main `Settings` class. Wire into `_SettingsProxy`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedding_factory.py::TestEmbeddingSettings -v`
Expected: 6 PASSED

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/mcp_memory_service/config.py tests/unit/test_embedding_factory.py
uv run ruff format src/mcp_memory_service/config.py tests/unit/test_embedding_factory.py
git add src/mcp_memory_service/config.py tests/unit/test_embedding_factory.py
git commit -m "feat: add EmbeddingSettings to config"
```

---

### Task 1.4: Inject EmbeddingProvider into QdrantStorage

This is the hardest task. QdrantStorage has 5+ methods that generate embeddings internally. All must delegate to the injected provider.

**Files:**
- Modify: `src/mcp_memory_service/storage/qdrant_storage.py`
- Modify: `tests/unit/test_qdrant_storage.py`
- Modify: `tests/unit/conftest.py`

**Current call sites in qdrant_storage.py:**
- `__init__` (line ~180): `self._embedding_model_instance = None`
- `_ensure_model_loaded()` (lines 209-226): loads SentenceTransformer
- `_generate_embedding()` (lines 685-753): sync single-text embedding
- `generate_embeddings_batch()` (lines 1048-1082): async batch embedding
- `initialize()` (line ~257): calls `_ensure_model_loaded()`
- `store()` (line ~834): calls `_generate_embedding(text, "passage")`
- `retrieve()` (line ~944): calls `_generate_embedding(query, "query")`
- `_upsert_tag_embeddings()` (line ~488): calls `generate_embeddings_batch()`
- `recall_memory()` (line ~1924): **transitive** — delegates to `retrieve()`, not a separate call site
- `count_semantic_search()`: **transitive** — delegates to `retrieve()`, not a separate call site

- [ ] **Step 1: Update test fixtures**

The existing tests mock `storage._generate_embedding` and `storage.generate_embeddings_batch` directly. After this change, those methods delegate to the provider. Tests need to mock the provider instead.

Add to `tests/unit/conftest.py`:

```python
@pytest.fixture
def mock_embedding_provider():
    """Mock EmbeddingProvider for unit tests."""
    from unittest.mock import AsyncMock, PropertyMock
    from mcp_memory_service.embedding.protocol import EmbeddingProvider

    provider = AsyncMock(spec=EmbeddingProvider)
    provider.embed_batch = AsyncMock(side_effect=_deterministic_embed_batch)
    type(provider).dimensions = PropertyMock(return_value=768)
    type(provider).model_name = PropertyMock(return_value="test-model")
    return provider


def _deterministic_embed_batch(texts: list[str], prompt_name: str = "query") -> list[list[float]]:
    """Deterministic embeddings based on text hash. Consistent with existing test patterns."""
    import hashlib
    import random

    result = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        result.append([rng.random() * 2 - 1 for _ in range(768)])
    return result
```

- [ ] **Step 2: Modify QdrantStorage constructor to accept optional provider**

In `qdrant_storage.py`, add `embedding_provider` parameter to `__init__`:

```python
def __init__(
    self,
    embedding_model: str,
    collection_name: str = "memories",
    # ... existing params ...
    embedding_provider: "EmbeddingProvider | None" = None,
) -> None:
    # ... existing init ...
    self._embedding_provider = embedding_provider
```

When `embedding_provider` is set:
- `_ensure_model_loaded()` becomes a no-op
- `_generate_embedding(text, prompt_name)` delegates to `asyncio.get_running_loop().run_in_executor(None, lambda: asyncio.run(provider.embed_batch([text], prompt_name)))[0]` — OR simpler: make it async and call `await provider.embed_batch([text], prompt_name)[0]` at all call sites. Review whether `_generate_embedding` callers are already async (they are — `store`, `retrieve` are all `async def`).
- `generate_embeddings_batch()` delegates to `await provider.embed_batch(texts, prompt_name)`

When `embedding_provider` is None (backward compat):
- Existing behavior unchanged

- [ ] **Step 3: Run existing embedding tests**

Run: `pytest tests/unit/test_qdrant_storage.py -v -k "embedding or prompt"`
Expected: All PASS (backward compat — no provider injected means old code path)

- [ ] **Step 4: Write tests for provider-injected path**

```python
# Append to tests/unit/test_qdrant_storage.py
from unittest.mock import AsyncMock, PropertyMock

class TestEmbeddingProviderInjection:
    """Tests for QdrantStorage with injected EmbeddingProvider."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that returns deterministic embeddings."""
        from mcp_memory_service.embedding.protocol import EmbeddingProvider

        provider = AsyncMock(spec=EmbeddingProvider)
        provider.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        type(provider).dimensions = PropertyMock(return_value=768)
        type(provider).model_name = PropertyMock(return_value="test-model")
        return provider

    @pytest.fixture
    async def storage_with_provider(self, mock_provider):
        """QdrantStorage wired with a mock provider, Qdrant in :memory: mode."""
        from mcp_memory_service.storage.qdrant_storage import QdrantStorage

        storage = QdrantStorage(
            embedding_model="test",
            storage_path=":memory:",
            embedding_provider=mock_provider,
        )
        await storage.initialize()
        return storage

    @pytest.mark.asyncio
    async def test_store_uses_provider_with_passage(self, storage_with_provider, mock_provider):
        """store() delegates to injected provider with prompt_name='passage'."""
        await storage_with_provider.store({
            "content": "test memory",
            "content_hash": "abc123",
            "metadata": {},
            "tags": [],
            "memory_type": "note",
            "created_at": "2026-01-01T00:00:00Z",
        })
        # Provider should have been called with prompt_name="passage"
        mock_provider.embed_batch.assert_any_await(
            ["test memory"], prompt_name="passage"
        )

    @pytest.mark.asyncio
    async def test_retrieve_uses_provider_with_query(self, storage_with_provider, mock_provider):
        """retrieve() delegates to injected provider with prompt_name='query'."""
        await storage_with_provider.retrieve("search text")
        mock_provider.embed_batch.assert_any_await(
            ["search text"], prompt_name="query"
        )

    @pytest.mark.asyncio
    async def test_provider_bypasses_model_loading(self, mock_provider):
        """When provider is injected, SentenceTransformer is never loaded."""
        from unittest.mock import patch
        from mcp_memory_service.storage.qdrant_storage import QdrantStorage

        with patch("mcp_memory_service.storage.qdrant_storage.SentenceTransformer") as mock_st:
            storage = QdrantStorage(
                embedding_model="test",
                storage_path=":memory:",
                embedding_provider=mock_provider,
            )
            await storage.initialize()
            await storage.retrieve("test")
            mock_st.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_embeddings_use_provider(self, storage_with_provider, mock_provider):
        """generate_embeddings_batch delegates to provider.embed_batch."""
        mock_provider.embed_batch = AsyncMock(return_value=[[0.1]*768, [0.2]*768])
        result = await storage_with_provider.generate_embeddings_batch(
            ["hello", "world"], prompt_name="passage"
        )
        assert len(result) == 2
        mock_provider.embed_batch.assert_awaited_once_with(
            ["hello", "world"], prompt_name="passage"
        )
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
git add src/mcp_memory_service/storage/qdrant_storage.py tests/
git commit -m "feat: inject EmbeddingProvider into QdrantStorage"
```

---

### Task 1.5: Inject EmbeddingProvider into MemoryService

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py`
- Modify: `tests/unit/test_memory_service.py`

- [ ] **Step 1: Modify MemoryService constructor**

Add `embedding_provider` parameter. When set, `_get_embeddings()` (line ~1076) delegates to `self._embedding_provider.embed_batch()` instead of `self.storage.generate_embeddings_batch()`.

The existing `_cached_embed` CacheKit wiring stays for now (Phase 3 moves it to `CachedEmbeddingProvider`). In this phase, if a provider is injected, bypass the per-text cache loop and call the provider directly.

- [ ] **Step 2: Update MemoryService tests**

Tests that mock `storage.generate_embeddings_batch` should also work with provider injection. Add parallel tests that inject a mock provider.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
git add src/mcp_memory_service/services/memory_service.py tests/unit/test_memory_service.py
git commit -m "feat: inject EmbeddingProvider into MemoryService"
```

---

### Task 1.6: Wire Provider Through shared_storage.py and Server Lifecycle

**Files:**
- Modify: `src/mcp_memory_service/shared_storage.py`
- Modify: `src/mcp_memory_service/mcp_server.py`
- Modify: `src/mcp_memory_service/web/app.py`
- Create: `src/mcp_memory_service/embedding/factory.py`
- Test: `tests/unit/test_embedding_factory.py` (extend)

**Note:** `unified_server.py` calls `get_shared_storage()` so it gets the provider transitively — no changes needed there.

- [ ] **Step 1: Write factory test**

```python
# Append to tests/unit/test_embedding_factory.py
class TestEmbeddingFactory:
    """Test create_embedding_provider factory."""

    def test_default_creates_local_provider(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider
        provider = create_embedding_provider()
        assert isinstance(provider, LocalProvider)

    def test_unknown_provider_raises(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(provider="nonexistent")

    def test_http_url_plaintext_warning(self, caplog):
        """Factory logs warning for http:// URLs."""
        import logging
        from mcp_memory_service.embedding.factory import create_embedding_provider
        with caplog.at_level(logging.WARNING):
            provider = create_embedding_provider(
                provider="openai_compat",
                model_name="test",
                base_url="http://localhost:8080",
                dimensions=768,
            )
        assert "plaintext HTTP" in caplog.text
```

- [ ] **Step 2: Implement factory**

```python
# src/mcp_memory_service/embedding/factory.py
"""Factory for creating EmbeddingProvider instances."""
import logging

from ..config import settings
from .protocol import EmbeddingProvider

logger = logging.getLogger(__name__)


def create_embedding_provider(
    provider: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """Create an EmbeddingProvider based on configuration.

    Args:
        provider: Override provider type. Defaults to settings.embedding.provider.
    """
    provider_type = provider or settings.embedding.provider

    if provider_type == "local":
        from .local import LocalProvider
        model_name = kwargs.get("model_name", settings.storage.embedding_model)
        return LocalProvider(model_name=model_name)

    if provider_type == "openai_compat":
        from .http import OpenAICompatAdapter

        base_url = kwargs.get("base_url", settings.embedding.url)
        if base_url and base_url.startswith("http://"):
            logger.warning(
                "Embedding service URL uses plaintext HTTP: %s. "
                "Consider using HTTPS for production deployments.",
                base_url,
            )

        return OpenAICompatAdapter(
            base_url=base_url,
            model_name=kwargs.get("model_name", settings.storage.embedding_model),
            dimensions=kwargs.get("dimensions", settings.storage.embedding_dimension),
            timeout=settings.embedding.timeout,
            max_batch=settings.embedding.max_batch,
            api_key=settings.embedding.api_key,
            tls_verify=settings.embedding.tls_verify,
        )

    msg = f"Unknown embedding provider: {provider_type!r}. Valid: local, openai_compat"
    raise ValueError(msg)
```

- [ ] **Step 3: Wire into shared_storage.py**

In `StorageManager.get_storage()`, create the provider via factory before creating storage:

```python
provider = create_embedding_provider()
storage = QdrantStorage(..., embedding_provider=provider)
service = MemoryService(storage=storage, ..., embedding_provider=provider)
```

Both share the same provider instance.

- [ ] **Step 4: Wire into mcp_server.py and web/app.py lifecycles**

Update the lifespan functions to pass the provider through. These already call `get_shared_storage()` — the provider comes along for free via `StorageManager`.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS

- [ ] **Step 6: Run lint**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/mcp_memory_service/embedding/factory.py src/mcp_memory_service/shared_storage.py \
        src/mcp_memory_service/mcp_server.py src/mcp_memory_service/web/app.py \
        tests/unit/test_embedding_factory.py
git commit -m "feat: wire EmbeddingProvider through server lifecycle"
```

---

### Task 1.7: Deprecate BaseStorage.generate_embeddings_batch

**Files:**
- Modify: `src/mcp_memory_service/storage/base.py`

- [ ] **Step 1: Add deprecation warning**

```python
# In base.py, modify generate_embeddings_batch:
import warnings

async def generate_embeddings_batch(self, texts: list[str], prompt_name: str = "query") -> list[list[float]]:
    """Deprecated: Use EmbeddingProvider.embed_batch() instead."""
    warnings.warn(
        "generate_embeddings_batch is deprecated. Use EmbeddingProvider.embed_batch().",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError("Use EmbeddingProvider.embed_batch() instead")
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS (nothing should call this anymore if provider injection is working)

- [ ] **Step 3: Commit**

```bash
git add src/mcp_memory_service/storage/base.py
git commit -m "deprecate: BaseStorage.generate_embeddings_batch in favor of EmbeddingProvider"
```

---

## Chunk 2: OpenAICompatAdapter (Phase 2)

### Task 2.1: Implement OpenAICompatAdapter

**Files:**
- Create: `src/mcp_memory_service/embedding/http.py`
- Test: `tests/unit/test_embedding_http.py`

**Note:** `httpx>=0.24.0` is already in `pyproject.toml` — no dependency change needed.

- [ ] **Step 1: Write adapter tests**

```python
# tests/unit/test_embedding_http.py
"""Tests for OpenAICompatAdapter — HTTP embedding provider."""
import pytest
from unittest.mock import AsyncMock, patch
import httpx

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

    def test_dimensions_required_for_http(self):
        """HTTP adapter requires explicit dimensions (no model introspection)."""
        adapter = OpenAICompatAdapter(
            base_url="http://localhost:8080",
            model_name="test",
            # dimensions defaults to 768 — acceptable default but caller should set explicitly
        )
        assert adapter.dimensions == 768

    @pytest.mark.asyncio
    async def test_embed_batch_sends_correct_request(self):
        """Verifies POST /v1/embeddings with OpenAI schema."""
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
    async def test_retries_on_503(self):
        """Retryable errors get 3 attempts with backoff."""
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

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self):
        """Fatal errors (400, 401) raise immediately."""
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
            base_url="http://x", model_name="x", dimensions=768
        ).embed_batch([])
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedding_http.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement OpenAICompatAdapter**

```python
# src/mcp_memory_service/embedding/http.py
"""OpenAI-compatible HTTP embedding adapter."""
import asyncio
import logging

import httpx
from pydantic import SecretStr

from .protocol import EmbeddingProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {503, 429, 502, 504}
_MAX_RETRIES = 3


class OpenAICompatAdapter:
    """EmbeddingProvider that speaks POST /v1/embeddings (OpenAI schema).

    Works with: HuggingFace TEI, vLLM, Ollama, OpenAI, and any
    server implementing the OpenAI embeddings API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dimensions: int = 768,
        timeout: int = 30,
        max_batch: int = 64,
        api_key: SecretStr | None = None,
        tls_verify: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimensions = dimensions
        self._max_batch = max_batch

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
            verify=tls_verify,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        if not texts:
            return []

        # Chunk into max_batch-sized requests
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch):
            chunk = texts[i : i + self._max_batch]
            embeddings = await self._embed_chunk(chunk, prompt_name)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _embed_chunk(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self._model_name,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post("/v1/embeddings", json=payload)

                if response.status_code in _RETRYABLE_STATUS:
                    if attempt < _MAX_RETRIES - 1:
                        wait = 2**attempt
                        logger.warning(
                            "Embedding service returned %d, retrying in %ds (attempt %d/%d)",
                            response.status_code, wait, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()

                # OpenAI schema: {"data": [{"embedding": [...], "index": N}]}
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Embedding request timed out, retrying in %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                raise

        return []  # unreachable but satisfies type checker

    async def health_check(self) -> bool:
        """Check if the embedding service is reachable."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedding_http.py -v`
Expected: All PASS

- [ ] **Step 5: Verify factory handles openai_compat**

Already handled in Task 1.6 factory code. Verify:

Run: `pytest tests/unit/test_embedding_factory.py -v`
Expected: All PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
git add src/mcp_memory_service/embedding/http.py tests/unit/test_embedding_http.py
git commit -m "feat: OpenAICompatAdapter for TEI/vLLM/Ollama/OpenAI"
```

---

## Chunk 3: CachedEmbeddingProvider + Cache Migration (Phase 3)

### Task 3.1: Implement CachedEmbeddingProvider

**Design decision (from review):** CacheKit's decorator pattern is transparent — you cannot distinguish cache hits from cache misses. The "partial batch optimization" (only send misses to inner) is incompatible with this. Instead, we accept per-text CacheKit calls. Each text goes through `_cache_fn(text, prompt_name)`, which either returns from L1/L2 cache or transparently calls the inner provider for a single text. This is simpler, correct, and the per-text overhead is negligible (L1 is a dict lookup, L2 is a Redis GET, embedding inference dominates).

**Files:**
- Create: `src/mcp_memory_service/embedding/cached.py`
- Test: `tests/unit/test_embedding_cached.py`

- [ ] **Step 1: Write cached provider tests**

```python
# tests/unit/test_embedding_cached.py
"""Tests for CachedEmbeddingProvider wrapper."""
import pytest
from unittest.mock import AsyncMock, PropertyMock

from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
from mcp_memory_service.embedding.protocol import EmbeddingProvider


@pytest.fixture
def inner_provider():
    provider = AsyncMock(spec=EmbeddingProvider)
    provider.embed_batch = AsyncMock(return_value=[[0.1] * 768])
    type(provider).dimensions = PropertyMock(return_value=768)
    type(provider).model_name = PropertyMock(return_value="test-model")
    return provider


class TestCachedEmbeddingProvider:
    def test_satisfies_protocol(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert isinstance(cached, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_first_call_delegates_to_inner(self, inner_provider):
        """Cache miss calls inner provider once per text."""
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch(["hello"])
        # Inner called once for the single text
        assert inner_provider.embed_batch.await_count == 1
        assert result == [[0.1] * 768]

    @pytest.mark.asyncio
    async def test_cache_hit_skips_inner(self, inner_provider):
        """Second call for same text+prompt_name doesn't call inner."""
        cached = CachedEmbeddingProvider(inner_provider)
        await cached.embed_batch(["hello"])
        await cached.embed_batch(["hello"])
        # CacheKit caches per-text — second call should be a hit.
        # With CacheKit L1, inner is called once (first miss), not twice.
        assert inner_provider.embed_batch.await_count == 1

    @pytest.mark.asyncio
    async def test_different_prompt_name_separate_cache(self, inner_provider):
        """'query' and 'passage' for same text produce separate cache entries."""
        inner_provider.embed_batch = AsyncMock(
            side_effect=[[[0.1] * 768], [[0.9] * 768]]
        )
        cached = CachedEmbeddingProvider(inner_provider)
        r1 = await cached.embed_batch(["hello"], prompt_name="query")
        r2 = await cached.embed_batch(["hello"], prompt_name="passage")
        assert r1 != r2
        assert inner_provider.embed_batch.await_count == 2

    @pytest.mark.asyncio
    async def test_batch_calls_inner_per_text(self, inner_provider):
        """Batch of N texts results in N per-text cache lookups."""
        inner_provider.embed_batch = AsyncMock(
            side_effect=[[[0.1] * 768], [[0.2] * 768]]
        )
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch(["hello", "world"])
        assert len(result) == 2
        # Each text is a separate cache call → separate inner call on miss
        assert inner_provider.embed_batch.await_count == 2

    @pytest.mark.asyncio
    async def test_mixed_hits_and_misses(self, inner_provider):
        """Pre-cached text is not re-embedded."""
        call_count = 0
        async def track_calls(texts, prompt_name="query"):
            nonlocal call_count
            call_count += 1
            return [[float(call_count)] * 768]

        inner_provider.embed_batch = AsyncMock(side_effect=track_calls)
        cached = CachedEmbeddingProvider(inner_provider)

        # Cache "hello"
        await cached.embed_batch(["hello"])
        assert call_count == 1

        # Request ["hello", "world"] — "hello" should be cached
        result = await cached.embed_batch(["hello", "world"])
        assert len(result) == 2
        # Only "world" should trigger a new inner call
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_empty_batch(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        result = await cached.embed_batch([])
        assert result == []
        inner_provider.embed_batch.assert_not_awaited()

    def test_dimensions_delegates(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert cached.dimensions == 768

    def test_model_name_delegates(self, inner_provider):
        cached = CachedEmbeddingProvider(inner_provider)
        assert cached.model_name == "test-model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedding_cached.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CachedEmbeddingProvider**

```python
# src/mcp_memory_service/embedding/cached.py
"""CachedEmbeddingProvider — CacheKit L1/L2 wrapper around any EmbeddingProvider."""
import asyncio
import logging
from typing import Any

from .protocol import EmbeddingProvider

logger = logging.getLogger(__name__)

try:
    import os
    from cachekit import cache as _cachekit_cache

    _ck_kwargs: dict[str, Any] = {}
    if not os.environ.get("REDIS_URL") and not os.environ.get("CACHEKIT_REDIS_URL"):
        _ck_kwargs["backend"] = None  # L1-only

    _CACHEKIT_AVAILABLE = True
except ImportError:
    _CACHEKIT_AVAILABLE = False


class CachedEmbeddingProvider:
    """Wraps any EmbeddingProvider with CacheKit L1 (in-process) + L2 (Redis) caching.

    Each text is cached individually via CacheKit's decorator. CacheKit handles
    miss-through transparently: on cache miss, it calls the inner provider for
    that single text, stores the result, and returns it.

    Cache key auto-generated by CacheKit includes (text, prompt_name) as function
    parameters. Namespace includes model_name and dimensions for auto-invalidation
    on model change.

    No batch optimization — per-text CacheKit calls are simpler and correct.
    L1 hits are dict lookups (nanoseconds), L2 hits are Redis GETs (~1ms).
    Embedding inference (~10-50ms/text) dominates, so per-text overhead is negligible.
    """

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self._cache_fn = None

        if _CACHEKIT_AVAILABLE:
            model_slug = inner.model_name.replace("/", "_")
            namespace = f"mcp_memory_embed_{model_slug}_{inner.dimensions}"

            @_cachekit_cache(ttl=86400, namespace=namespace, **_ck_kwargs)
            async def _cached_single_embed(text: str, prompt_name: str = "query") -> list[float]:
                """CacheKit-decorated single-text embed. Miss-through to inner provider."""
                result = await inner.embed_batch([text], prompt_name=prompt_name)
                return result[0]

            self._cache_fn = _cached_single_embed

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        if not texts:
            return []

        if not self._cache_fn:
            # CacheKit not available — pass through directly
            return await self._inner.embed_batch(texts, prompt_name=prompt_name)

        # Per-text cache lookup. CacheKit handles miss-through transparently.
        results = await asyncio.gather(
            *(self._cache_fn(text, prompt_name) for text in texts)
        )
        return list(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedding_cached.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/mcp_memory_service/embedding/cached.py tests/unit/test_embedding_cached.py
uv run ruff format src/mcp_memory_service/embedding/cached.py tests/unit/test_embedding_cached.py
git add src/mcp_memory_service/embedding/cached.py tests/unit/test_embedding_cached.py
git commit -m "feat: CachedEmbeddingProvider with CacheKit L1/L2"
```

---

### Task 3.2: Migrate Cache Logic from memory_service.py

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py`
- Modify: `src/mcp_memory_service/embedding/factory.py`

- [ ] **Step 1: Update factory to wrap provider in CachedEmbeddingProvider**

In `factory.py`, after creating the inner provider, wrap it:

```python
from .cached import CachedEmbeddingProvider

def create_embedding_provider(...) -> EmbeddingProvider:
    inner = _create_inner_provider(provider_type, **kwargs)
    return CachedEmbeddingProvider(inner)
```

- [ ] **Step 2: Remove `_cached_embed` and `_get_embeddings` cache logic from memory_service.py**

In `memory_service.py`:
- Remove `_cached_embed()` function (lines ~88-92)
- Simplify `_get_embeddings()` to just `return await self._embedding_provider.embed_batch(texts, prompt_name)`
- Keep `_cached_fetch_all_tags`, `_cached_corpus_count`, `_cached_extract_keywords` (these stay)
- Remove `_embed_fn` module-level var

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
git add src/mcp_memory_service/services/memory_service.py src/mcp_memory_service/embedding/factory.py
git commit -m "feat: migrate embedding cache from memory_service to CachedEmbeddingProvider"
```

---

### Task 3.3: Startup Dimension Check

**Files:**
- Modify: `src/mcp_memory_service/shared_storage.py`
- Test: `tests/unit/test_embedding_factory.py` (extend)

- [ ] **Step 1: Write dimension check test**

```python
class TestDimensionCheck:
    @pytest.mark.asyncio
    async def test_dimension_mismatch_fails_fast(self):
        """Provider with wrong dimensions vs Qdrant collection raises at startup."""
        # ... mock provider with 384 dims, mock Qdrant collection with 768 dims
        # ... assert ValueError raised with clear message
```

- [ ] **Step 2: Implement in shared_storage.py**

After creating storage and provider, before returning:

```python
collection_dims = storage._vector_size or settings.storage.embedding_dimension
provider_dims = provider.dimensions
if collection_dims and provider_dims != collection_dims:
    raise ValueError(
        f"Embedding provider dimensions ({provider_dims}) don't match "
        f"Qdrant collection dimensions ({collection_dims}). "
        f"Provider: {provider.model_name}, Collection: {storage.collection_name}"
    )
```

- [ ] **Step 3: Run tests, lint, commit**

```bash
pytest tests/ -x -m "not slow" --timeout=60
uv run ruff check src/ tests/
git add src/mcp_memory_service/shared_storage.py tests/unit/test_embedding_factory.py
git commit -m "feat: fail-fast on embedding dimension mismatch at startup"
```

---

## Chunk 4: Dockerfile + Cleanup (Phases 4-5)

### Task 4.1: Multi-Stage Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Refactor Dockerfile**

Split into three stages:

```dockerfile
# ── Base (shared) ──────────────────────────
FROM python:3.13-slim AS base
# uv, source copy, spaCy model download
# Everything EXCEPT torch and sentence-transformers

# ── Full (backward compat, default) ────────
FROM base AS full
# Install torch (CPU) + sentence-transformers
# Download HF embedding model at build time

# ── API (thin, scale-out) ─────────────────
FROM base AS api
# Install deps WITHOUT torch/sentence-transformers
# uv sync --prune torch or equivalent
```

Key: `full` is the last stage → default when `--target` is omitted.

- [ ] **Step 2: Build and test both targets**

```bash
docker build --target api -t mcp-memory:api .
docker build --target full -t mcp-memory:full .
# Verify api target doesn't contain torch
docker run --rm mcp-memory:api python -c "import torch" 2>&1 | grep -q "ModuleNotFoundError"
# Verify full target works
docker run --rm mcp-memory:full python -c "import sentence_transformers; print('ok')"
```

- [ ] **Step 3: Verify api target boots with MCP_EMBEDDING_PROVIDER=openai_compat**

```bash
docker run --rm -e MCP_EMBEDDING_PROVIDER=openai_compat -e MCP_EMBEDDING_URL=http://fake:8080 \
  mcp-memory:api python -c "from mcp_memory_service.embedding.factory import create_embedding_provider; print('ok')"
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat: multi-stage Dockerfile with api and full targets"
```

---

### Task 4.2: Docker Compose Dev Example

**Files:**
- Create: `docker-compose.dev.yml`

- [ ] **Step 1: Create compose file**

Use the security-hardened example from the spec (no host ports for backend services, Redis auth).

- [ ] **Step 2: Commit**

```bash
git add docker-compose.dev.yml
git commit -m "docs: add docker-compose.dev.yml for two-service local deployment"
```

---

### Task 4.3: Remove SSE (Phase 5)

**Files:**
- Delete: `src/mcp_memory_service/web/sse.py`
- Modify: `src/mcp_memory_service/web/app.py` (remove SSE imports, routes, manager start/stop)

- [ ] **Step 1: Check for SSE consumers**

```bash
grep -r "sse\|events\|EventSource\|event-stream" src/ tests/ --include="*.py" -l
```

- [ ] **Step 2: Remove SSE references from app.py**

Remove: `sse_manager.start()`, `sse_manager.stop()`, SSE event broadcasts in store/delete/search handlers, events router.

- [ ] **Step 3: Delete sse.py**

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -x -m "not slow" --timeout=60`
Expected: All PASS (some SSE-specific tests may need removal)

- [ ] **Step 5: Commit**

```bash
git add -u  # stages deletions
git commit -m "refactor: remove SSE (stateful, unused by MCP consumers)"
```

---

### Task 4.4: Replace WriteQueue with Semaphore (Phase 5)

**Files:**
- Delete: `src/mcp_memory_service/web/write_queue.py`
- Modify: `src/mcp_memory_service/web/api/memories.py` (5 call sites)
- Modify: `src/mcp_memory_service/web/api/health.py` (stats reporting)

- [ ] **Step 1: Replace WriteQueue usage in memories.py**

Replace `await write_queue.put(...)` with `async with write_semaphore:` around the actual write calls.

```python
# Module-level in memories.py or app.py
import asyncio
_write_semaphore = asyncio.Semaphore(20)

# In each write endpoint:
async with _write_semaphore:
    result = await storage.store(memory)
```

- [ ] **Step 2: Update health stats**

Replace `write_queue.get_stats()` with semaphore info:

```python
"write_concurrency": {
    "max": 20,
    "available": _write_semaphore._value,  # current available slots
}
```

- [ ] **Step 3: Delete write_queue.py**

- [ ] **Step 4: Run full test suite, lint, commit**

```bash
pytest tests/ -x -m "not slow" --timeout=60
uv run ruff check src/ tests/
git add -u
git commit -m "refactor: replace WriteQueue with asyncio.Semaphore(20)"
```

---

### Task 4.5: Gate Diagnostic Deques Behind Debug Flag

**Files:**
- Modify: `src/mcp_memory_service/services/memory_service.py`

- [ ] **Step 1: Gate deque creation**

```python
# In MemoryService.__init__:
if settings.debug_tools_enabled:
    self._search_logs: deque = deque(maxlen=1000)
    self._audit_logs: deque = deque(maxlen=1000)
else:
    self._search_logs = None
    self._audit_logs = None
```

Guard all deque appends with `if self._search_logs is not None:`.

- [ ] **Step 2: Add per-instance warning to analytics endpoints**

When debug tools are enabled and analytics are queried, include:
```json
{"warning": "Analytics data is per-instance. In multi-instance deployments, each instance tracks its own data."}
```

- [ ] **Step 3: Run full test suite, lint, commit**

```bash
pytest tests/ -x -m "not slow" --timeout=60
uv run ruff check src/ tests/
git add src/mcp_memory_service/services/memory_service.py
git commit -m "refactor: gate diagnostic deques behind MCP_MEMORY_EXPOSE_DEBUG_TOOLS"
```

---

## Chunk 5: CI + Final Validation

### Task 5.1: Update CI to Build Both Docker Targets

**Files:**
- Modify: `.github/workflows/qa.yml`

- [ ] **Step 1: Add api target build to QA workflow**

```yaml
- name: Build Docker (full)
  run: docker build --target full .
- name: Build Docker (api)
  run: docker build --target api .
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/qa.yml
git commit -m "ci: build both docker targets in QA workflow"
```

---

### Task 5.2: Wire Readiness Probe

**Files:**
- Modify: `src/mcp_memory_service/web/api/health.py`

The spec defines `/health/ready` (checks embedding provider, cached 10s). The `OpenAICompatAdapter.health_check()` method exists from Task 2.1.

- [ ] **Step 1: Add readiness endpoint**

Wire `provider.health_check()` into `/health/ready`. For `LocalProvider`, readiness = model loaded. For HTTP adapter, readiness = `/health` returns 200. Cache the result for 10 seconds to avoid hammering the embedding service.

- [ ] **Step 2: Run tests, lint, commit**

```bash
pytest tests/ -x -m "not slow" --timeout=60
uv run ruff check src/ tests/
git add src/mcp_memory_service/web/api/health.py
git commit -m "feat: readiness probe checks embedding provider health"
```

---

### Task 5.3: Final Integration Test

- [ ] **Step 1: Run full test suite one final time**

```bash
pytest tests/ -x --timeout=120
```

- [ ] **Step 2: Run lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

- [ ] **Step 3: Build both Docker images**

```bash
docker build --target api -t mcp-memory:api .
docker build --target full -t mcp-memory:full .
```

- [ ] **Step 4: Verify backward compat (full image, local provider)**

```bash
docker run --rm mcp-memory:full python -c "
from mcp_memory_service.embedding.factory import create_embedding_provider
p = create_embedding_provider()
print(f'Provider: {p.model_name}, dims: {p.dimensions}')
"
```

- [ ] **Step 5: Final commit if any fixups needed**

---

## Deferred Items

These are spec requirements that are out of scope for this plan but should be tracked:

1. **Cloud metadata IP blocking (SSRF)** — URL validation to block `169.254.x.x` and link-local ranges in `EmbeddingSettings`. Tracked for security hardening follow-up.
2. **OpenAICompatAdapter dimension discovery** — Currently requires explicit `dimensions` parameter. Future: send a test embed at factory creation time to auto-discover dimensions. Low priority since dimension mismatch is caught at startup by Task 3.3.

---

## Summary

| Chunk | Tasks | Phase | Key Risk |
|-------|-------|-------|----------|
| 1 | 7 tasks | Phase 1 | QdrantStorage embedding call site surgery |
| 2 | 1 task | Phase 2 | HTTP adapter correctness |
| 3 | 3 tasks | Phase 3 | Cache migration without breaking existing behavior |
| 4 | 5 tasks | Phases 4-5 | Dockerfile refactor + stateful component removal |
| 5 | 3 tasks | Validation | CI + readiness probe + final integration |

Total: 19 tasks, ~55 discrete steps. Each step is testable independently. Every commit leaves tests passing.
