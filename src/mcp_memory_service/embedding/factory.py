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
        from .http import OpenAICompatAdapter

        base_url = str(kwargs.get("base_url", settings.embedding.url) or "")
        if not base_url:
            msg = "openai_compat provider requires MCP_EMBEDDING_URL or base_url parameter"
            raise ValueError(msg)
        if base_url.startswith("http://"):
            logger.warning(
                "Embedding service URL uses plaintext HTTP: %s. Consider using HTTPS for production deployments.",
                base_url,
            )

        return OpenAICompatAdapter(
            base_url=base_url,
            model_name=str(kwargs.get("model_name", settings.storage.embedding_model)),
            dimensions=int(kwargs.get("dimensions", 768)),  # type: ignore[arg-type]
            timeout=settings.embedding.timeout,
            max_batch=settings.embedding.max_batch,
            api_key=settings.embedding.api_key,
            tls_verify=settings.embedding.tls_verify,
        )

    msg = f"Unknown embedding provider: {provider_type!r}. Valid: local, openai_compat"
    raise ValueError(msg)
