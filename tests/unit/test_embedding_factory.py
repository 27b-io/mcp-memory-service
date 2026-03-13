"""Tests for embedding provider factory and configuration."""

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

    def test_default_creates_local_provider(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = create_embedding_provider()
        assert isinstance(provider, LocalProvider)

    def test_unknown_provider_raises(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(provider="nonexistent")

    def test_openai_compat_not_yet_implemented(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider

        with pytest.raises(NotImplementedError):
            create_embedding_provider(provider="openai_compat")

    def test_explicit_model_name_override(self):
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = create_embedding_provider(model_name="intfloat/e5-small-v2")
        assert isinstance(provider, LocalProvider)
        assert provider.model_name == "intfloat/e5-small-v2"

    def test_factory_reads_default_model_from_config(self):
        from mcp_memory_service.config import settings
        from mcp_memory_service.embedding.factory import create_embedding_provider
        from mcp_memory_service.embedding.local import LocalProvider

        provider = create_embedding_provider()
        assert isinstance(provider, LocalProvider)
        assert provider.model_name == settings.storage.embedding_model
