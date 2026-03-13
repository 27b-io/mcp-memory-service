"""Embedding provider protocol and adapters."""

from .factory import create_embedding_provider
from .http import OpenAICompatAdapter
from .local import LocalProvider
from .protocol import EmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalProvider", "OpenAICompatAdapter", "create_embedding_provider"]
