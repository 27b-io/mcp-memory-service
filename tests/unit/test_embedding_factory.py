"""Tests for embedding provider factory and configuration."""


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
