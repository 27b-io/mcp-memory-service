"""Tests for EmbeddingProvider protocol."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_memory_service.embedding.local import LocalProvider
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
            _ = provider.dimensions
            _ = provider.dimensions
            mock_st.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_name_fallback_no_prompts(self):
        """Models without .prompts attribute fall back to manual prefix."""
        import numpy as np

        with patch("mcp_memory_service.embedding.local.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([[0.1] * 768])
            # Remove prompts attribute to simulate older sentence-transformers
            del mock_model.prompts
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
