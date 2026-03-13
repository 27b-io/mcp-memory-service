"""Embedding provider protocol and adapters."""

from .factory import create_embedding_provider
from .local import LocalProvider
from .protocol import EmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalProvider", "create_embedding_provider"]
