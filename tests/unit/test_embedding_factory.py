"""Tests for embedding provider factory and configuration."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


class TestEmbeddingSettings:
    """Test EmbeddingSettings pydantic model."""

    def test_default_provider_is_local(self):
        from mcp_memory_service.config import EmbeddingSettings

        s = EmbeddingSettings()
        assert s.provider == "local"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_EMBEDDING_PROVIDER", "openai_compat")
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

    def test_api_key_is_secret(self):
        from pydantic import SecretStr

        from mcp_memory_service.config import EmbeddingSettings

        s = EmbeddingSettings(api_key="my-secret")
        assert isinstance(s.api_key, SecretStr)
        assert s.api_key.get_secret_value() == "my-secret"


class TestEmbeddingFactory:
    """Test create_embedding_provider factory."""

    def test_default_creates_cached_local_provider(self):
        from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = create_embedding_provider()
        assert isinstance(provider, CachedEmbeddingProvider)
        assert isinstance(provider._inner, LocalProvider)

    def test_unknown_provider_raises(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(provider="nonexistent")

    def test_openai_compat_requires_url(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider

        with pytest.raises(ValueError, match="requires.*URL"):
            create_embedding_provider(provider="openai_compat")

    def test_openai_compat_creates_cached_adapter(self):
        from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.http import OpenAICompatAdapter

        provider = create_embedding_provider(
            provider="openai_compat",
            base_url="http://localhost:8080",
            model_name="test",
            dimensions=768,
        )
        assert isinstance(provider, CachedEmbeddingProvider)
        assert isinstance(provider._inner, OpenAICompatAdapter)

    def test_http_url_plaintext_warning(self, caplog):
        import logging

        from mcp_memory_service.embedding.factory import create_embedding_provider

        with caplog.at_level(logging.WARNING):
            create_embedding_provider(
                provider="openai_compat",
                base_url="http://localhost:8080",
                model_name="test",
                dimensions=768,
            )
        assert "plaintext HTTP" in caplog.text

    def test_explicit_model_name_override(self):
        from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = create_embedding_provider(model_name="intfloat/e5-small-v2")
        assert isinstance(provider, CachedEmbeddingProvider)
        assert isinstance(provider._inner, LocalProvider)
        assert provider.model_name == "intfloat/e5-small-v2"

    def test_factory_reads_default_model_from_config(self):
        from mcp_memory_service.config import settings
        from mcp_memory_service.embedding.factory import create_embedding_provider

        provider = create_embedding_provider()
        assert provider.model_name == settings.storage.embedding_model

    def test_inner_provider_skips_cache(self):
        """_create_inner_provider returns unwrapped provider."""
        from mcp_memory_service.embedding.cached import CachedEmbeddingProvider
        from mcp_memory_service.embedding.factory import _create_inner_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = _create_inner_provider()
        assert isinstance(provider, LocalProvider)
        assert not isinstance(provider, CachedEmbeddingProvider)


class TestDimensionCheck:
    """Test embedding dimension mismatch detection at startup."""

    @pytest.mark.asyncio
    async def test_dimension_mismatch_fails_fast(self):
        """Provider with wrong dimensions vs storage raises at startup."""
        from mcp_memory_service.shared_storage import StorageManager

        manager = StorageManager.__new__(StorageManager)
        manager._storage = None
        manager._graph_client = None
        manager._write_queue = None
        manager._embedding_provider = None
        manager._initialization_lock = asyncio.Lock()
        manager._initialized = False

        # Mock provider with 384 dims
        mock_provider = AsyncMock()
        mock_provider.dimensions = 384
        mock_provider.model_name = "test-model"

        # Mock storage with 768 dims (mismatch!)
        mock_storage = AsyncMock()
        mock_storage._vector_size = 768
        mock_storage.collection_name = "memories"
        mock_storage.initialize = AsyncMock()

        with (
            patch(
                "mcp_memory_service.embedding.factory.create_embedding_provider",
                return_value=mock_provider,
            ),
            patch(
                "mcp_memory_service.shared_storage.create_storage_instance",
                return_value=mock_storage,
            ),
            patch("mcp_memory_service.shared_storage.create_graph_layer", return_value=None),
        ):
            with pytest.raises(ValueError, match="dimensions.*384.*don't match.*768"):
                await manager.get_storage()

    @pytest.mark.asyncio
    async def test_dimension_match_succeeds(self):
        """Matching dimensions proceed without error."""
        from mcp_memory_service.shared_storage import StorageManager

        manager = StorageManager.__new__(StorageManager)
        manager._storage = None
        manager._graph_client = None
        manager._write_queue = None
        manager._embedding_provider = None
        manager._initialization_lock = asyncio.Lock()
        manager._initialized = False

        mock_provider = AsyncMock()
        mock_provider.dimensions = 768
        mock_provider.model_name = "test-model"

        mock_storage = AsyncMock()
        mock_storage._vector_size = 768
        mock_storage.collection_name = "memories"
        mock_storage.initialize = AsyncMock()

        with (
            patch(
                "mcp_memory_service.embedding.factory.create_embedding_provider",
                return_value=mock_provider,
            ),
            patch(
                "mcp_memory_service.shared_storage.create_storage_instance",
                return_value=mock_storage,
            ),
            patch("mcp_memory_service.shared_storage.create_graph_layer", return_value=None),
        ):
            result = await manager.get_storage()
            assert result is mock_storage
