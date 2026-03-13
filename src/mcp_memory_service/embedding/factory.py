"""Factory for creating EmbeddingProvider instances."""

import logging

from .protocol import EmbeddingProvider

logger = logging.getLogger(__name__)


def create_embedding_provider(
    provider: str | None = None,
    **kwargs: object,
) -> EmbeddingProvider:
    """Create an EmbeddingProvider based on configuration.

    Args:
        provider: Override provider type. Defaults to settings.embedding.provider.
        **kwargs: Override specific settings (model_name, base_url, dimensions, etc.)
    """
    from ..config import settings

    provider_type = provider or settings.embedding.provider

    if provider_type == "local":
        from .local import LocalProvider

        model_name = str(kwargs.get("model_name", settings.storage.embedding_model))
        return LocalProvider(model_name=model_name)

    if provider_type == "openai_compat":
        msg = "openai_compat provider not yet implemented (Task 2.1)"
        raise NotImplementedError(msg)

    msg = f"Unknown embedding provider: {provider_type!r}. Valid: local, openai_compat"
    raise ValueError(msg)
