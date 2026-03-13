# src/mcp_memory_service/embedding/local.py
"""LocalProvider — in-process SentenceTransformer embedding generation."""

import asyncio
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Module-level import for mockability. Absent in api Docker target.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

# Prompt name mappings for instruction-tuned models
_PROMPT_PREFIXES: dict[str, dict[str, str]] = {
    "query": {"nomic": "search_query: ", "e5": "query: "},
    "passage": {"nomic": "search_document: ", "e5": "passage: "},
}


class LocalProvider:
    """EmbeddingProvider backed by in-process SentenceTransformer.

    Thread-safe lazy model loading. Delegates encode() to asyncio executor
    to avoid blocking the event loop.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: object | None = None
        self._lock = threading.Lock()
        self._dimensions: int | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model with double-checked locking."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Use the 'full' Docker target or install it: pip install sentence-transformers"
                )

            device = self._device
            if device is None:
                from ..utils.system_detection import get_torch_device

                device = str(get_torch_device())

            logger.info("Loading embedding model %s on %s", self._model_name, device)
            self._model = SentenceTransformer(self._model_name, device=device)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info("Model loaded: %s (%d dimensions)", self._model_name, self._dimensions)

    @property
    def dimensions(self) -> int:
        self._ensure_loaded()
        return self._dimensions  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode_sync(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        """Synchronous encode — called in executor."""
        self._ensure_loaded()
        model = self._model

        # Try native prompt_name support (sentence-transformers >= 2.2.2)
        if hasattr(model, "prompts") and model.prompts:
            try:
                result = model.encode(texts, prompt_name=prompt_name, convert_to_tensor=False)
            except TypeError:
                # prompt_name kwarg not supported by this version
                prefix = self._get_manual_prefix(prompt_name)
                if prefix:
                    result = model.encode([f"{prefix}{t}" for t in texts], convert_to_tensor=False)
                else:
                    result = model.encode(texts, convert_to_tensor=False)
        else:
            prefix = self._get_manual_prefix(prompt_name)
            if prefix:
                result = model.encode([f"{prefix}{t}" for t in texts], convert_to_tensor=False)
            else:
                result = model.encode(texts, convert_to_tensor=False)

        # Convert numpy to Python lists
        if isinstance(result, np.ndarray):
            return result.tolist()
        return [r.tolist() if isinstance(r, np.ndarray) else list(r) for r in result]

    def _get_manual_prefix(self, prompt_name: str) -> str | None:
        """Get manual prefix for models without native prompt support."""
        model_lower = self._model_name.lower()
        prefixes = _PROMPT_PREFIXES.get(prompt_name, {})
        for family, prefix in prefixes.items():
            if family in model_lower:
                return prefix
        return None

    async def embed_batch(
        self,
        texts: list[str],
        prompt_name: str = "query",
    ) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, prompt_name)
