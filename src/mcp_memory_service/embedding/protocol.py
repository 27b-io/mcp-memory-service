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
