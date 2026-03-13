"""Embedding provider protocol and adapters."""

from .local import LocalProvider
from .protocol import EmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalProvider"]
