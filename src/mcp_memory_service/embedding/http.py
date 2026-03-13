"""OpenAI-compatible HTTP embedding adapter."""

import asyncio
import logging

import httpx
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 502, 503, 504}
_MAX_RETRIES = 3


class OpenAICompatAdapter:
    """EmbeddingProvider that speaks POST /v1/embeddings (OpenAI schema).

    Works with: HuggingFace TEI, vLLM, Ollama, OpenAI, and any
    server implementing the OpenAI embeddings API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dimensions: int = 768,
        timeout: int = 30,
        max_batch: int = 64,
        api_key: SecretStr | None = None,
        tls_verify: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimensions = dimensions
        self._max_batch = max_batch

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
            verify=tls_verify,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch):
            chunk = texts[i : i + self._max_batch]
            embeddings = await self._embed_chunk(chunk, prompt_name)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _embed_chunk(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        payload: dict[str, object] = {
            "input": texts,
            "model": self._model_name,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post("/v1/embeddings", json=payload)

                if response.status_code in _RETRYABLE_STATUS:
                    if attempt < _MAX_RETRIES - 1:
                        wait = 2**attempt
                        logger.warning(
                            "Embedding service returned %d, retrying in %ds (attempt %d/%d)",
                            response.status_code,
                            wait,
                            attempt + 1,
                            _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()

                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Embedding request timed out, retrying in %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                raise

        return []  # unreachable but satisfies type checker

    async def health_check(self) -> bool:
        """Check if the embedding service is reachable."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
