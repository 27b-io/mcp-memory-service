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
