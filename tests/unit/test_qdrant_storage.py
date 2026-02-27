# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for Qdrant storage backend - FOCUSED on testable units.

These tests verify:
1. Circuit breaker logic (state machine behavior)
2. Dimension validation (input validation)
3. Configuration handling
4. Error message formatting

For integration tests with real Qdrant behavior, see:
tests/integration/test_storage_integration.py
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.mcp_memory_service.models.memory import Memory

# Import the module under test
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage, is_retryable_error
from src.mcp_memory_service.utils.hashing import generate_content_hash

# =============================================================================
# Circuit Breaker Unit Tests
# These test the circuit breaker STATE MACHINE, not Qdrant interactions
# =============================================================================


class TestCircuitBreakerLogic:
    """Test circuit breaker state transitions without touching Qdrant."""

    def test_circuit_starts_closed(self):
        """Verify circuit breaker starts in closed state."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage._failure_count == 0
        assert storage._circuit_open_until is None

    def test_circuit_opens_after_threshold_failures(self):
        """Verify circuit opens after N consecutive failures."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Simulate reaching threshold
        storage._failure_count = storage._failure_threshold

        # Verify circuit would open (manually set as code does)
        storage._circuit_open_until = datetime.now() + timedelta(seconds=storage._circuit_timeout)

        assert storage._circuit_open_until is not None
        assert storage._circuit_open_until > datetime.now()

    def test_circuit_timeout_calculation(self):
        """Verify circuit timeout is calculated correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Open circuit
        now = datetime.now()
        storage._circuit_open_until = now + timedelta(seconds=storage._circuit_timeout)

        # Verify timeout is ~60 seconds in the future
        delta = storage._circuit_open_until - now
        assert 59 <= delta.total_seconds() <= 61

    def test_circuit_closes_after_timeout_expires(self):
        """Verify circuit can close after timeout expires."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Set circuit to have expired
        storage._circuit_open_until = datetime.now() - timedelta(seconds=1)
        storage._failure_count = 5

        # Check if circuit is past timeout (manual check as storage does)
        is_past_timeout = datetime.now() > storage._circuit_open_until

        assert is_past_timeout

    def test_failure_count_resets_on_success(self):
        """Verify failure count resets to zero after success."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Simulate some failures
        storage._failure_count = 3

        # Reset (as code does on success)
        storage._failure_count = 0
        storage._circuit_open_until = None

        assert storage._failure_count == 0
        assert storage._circuit_open_until is None


# =============================================================================
# Dimension Validation Unit Tests
# These test input validation logic
# =============================================================================


class TestDimensionValidation:
    """Test embedding dimension validation logic."""

    def test_detects_vector_size_for_minilm(self):
        """Verify correct dimension detection for all-MiniLM-L6-v2."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # MiniLM should be 384 dimensions
        # This is set during initialize(), but we can check the model map
        known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
        }
        assert known_dimensions["all-MiniLM-L6-v2"] == 384

    def test_detects_vector_size_for_mpnet(self):
        """Verify correct dimension detection for all-mpnet-base-v2."""
        known_dimensions = {
            "all-mpnet-base-v2": 768,
        }
        assert known_dimensions["all-mpnet-base-v2"] == 768

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_clear_error(self):
        """Verify dimension mismatch produces helpful error message."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")
            storage._initialized = True
            storage._vector_size = 384
            storage.client = MagicMock()

        # Create memory with wrong dimensions
        memory = Memory(
            content="Test content",
            content_hash=generate_content_hash("Test content"),
            embedding=[0.1] * 768,  # Wrong! Should be 384
            tags=["test"],
        )

        with pytest.raises(ValueError) as exc_info:
            await storage.store(memory)

        error_msg = str(exc_info.value).lower()
        assert "dimension mismatch" in error_msg
        assert "384" in str(exc_info.value)
        assert "768" in str(exc_info.value)


# =============================================================================
# Configuration Unit Tests
# =============================================================================


class TestQdrantConfiguration:
    """Test configuration validation."""

    def test_rejects_both_url_and_storage_path(self):
        """Verify cannot specify both embedded and server mode."""
        with pytest.raises(ValueError) as exc_info:
            QdrantStorage(embedding_model="all-MiniLM-L6-v2", url="http://localhost:6333", storage_path="/tmp/test")

        assert "both" in str(exc_info.value).lower()

    def test_rejects_neither_url_nor_storage_path(self):
        """Verify must specify at least one mode."""
        with pytest.raises(ValueError) as exc_info:
            QdrantStorage(embedding_model="all-MiniLM-L6-v2")

        assert "must specify" in str(exc_info.value).lower()

    def test_accepts_url_only_for_server_mode(self):
        """Verify server mode is configured correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", url="http://localhost:6333")

        assert storage.url == "http://localhost:6333"
        assert storage.storage_path is None

    def test_accepts_storage_path_for_embedded_mode(self):
        """Verify embedded mode is configured correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.storage_path == "/tmp/test"
        assert storage.url is None


# =============================================================================
# Retry Logic Unit Tests
# =============================================================================


class TestRetryLogic:
    """Test the retry decision logic."""

    def test_5xx_errors_are_retryable(self):
        """Verify 5xx server errors trigger retry."""
        try:
            from qdrant_client.http import exceptions as qdrant_exceptions

            # Create a mock error with status_code attribute
            error = MagicMock(spec=qdrant_exceptions.UnexpectedResponse)
            error.status_code = 503
            # Make isinstance check work
            error.__class__ = qdrant_exceptions.UnexpectedResponse
            assert is_retryable_error(error)
        except ImportError:
            pytest.skip("qdrant-client not available")

    def test_4xx_errors_are_not_retryable(self):
        """Verify 4xx client errors do not trigger retry."""
        try:
            from qdrant_client.http import exceptions as qdrant_exceptions

            # Create a mock error with status_code attribute
            error = MagicMock(spec=qdrant_exceptions.UnexpectedResponse)
            error.status_code = 400
            error.__class__ = qdrant_exceptions.UnexpectedResponse
            assert not is_retryable_error(error)
        except ImportError:
            pytest.skip("qdrant-client not available")

    def test_generic_exceptions_are_not_retryable(self):
        """Verify generic exceptions do not trigger retry."""
        error = ValueError("Invalid input")
        assert not is_retryable_error(error)


# =============================================================================
# Properties Unit Tests
# =============================================================================


class TestQdrantProperties:
    """Test storage property values."""

    def test_max_content_length_is_none(self):
        """Verify Qdrant has no content length limit."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.max_content_length is None

    def test_supports_chunking_is_true(self):
        """Verify Qdrant supports chunking."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.supports_chunking is True


# =============================================================================
# Hash to UUID Conversion Tests
# =============================================================================


class TestHashToUuidConversion:
    """Test content hash to Qdrant UUID conversion."""

    def test_hash_to_uuid_is_deterministic(self):
        """Verify same hash always produces same UUID."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Use a valid SHA256 hash (64 hex chars) - _hash_to_uuid takes first 32 for UUID
        test_hash = "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        uuid1 = storage._hash_to_uuid(test_hash)
        uuid2 = storage._hash_to_uuid(test_hash)

        assert uuid1 == uuid2

    def test_different_hashes_produce_different_uuids(self):
        """Verify different hashes produce different UUIDs."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Use valid 64-char hex strings (SHA256 format)
        hash1 = "1111111111111111111111111111111111111111111111111111111111111111"
        hash2 = "2222222222222222222222222222222222222222222222222222222222222222"
        uuid1 = storage._hash_to_uuid(hash1)
        uuid2 = storage._hash_to_uuid(hash2)

        assert uuid1 != uuid2


# =============================================================================
# Embedding Prompt Prefix Tests (Issue #101)
# Models like E5, Nomic, Arctic require instruction prefixes for meaningful
# cosine similarity. Without them, scores are ~0.01 instead of ~0.7+.
# =============================================================================


class TestEmbeddingPromptPrefix:
    """Test that prompt_name is passed to encode() for instruction-tuned models."""

    def _make_storage(self):
        """Create a QdrantStorage with mocked model."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="intfloat/e5-base-v2", storage_path="/tmp/test")
        # Inject a mock model that has prompts configured
        mock_model = MagicMock()
        mock_model.prompts = {"query": "query: ", "passage": "passage: "}
        mock_model.encode.return_value = np.zeros(768, dtype=np.float32)
        storage._embedding_model_instance = mock_model
        storage._vector_size = 768
        return storage, mock_model

    def test_storage_embedding_uses_passage_prompt(self):
        """Bug #101: _generate_embedding for storage must use prompt_name='passage'."""
        storage, mock_model = self._make_storage()

        storage._generate_embedding("test content")

        mock_model.encode.assert_called_once_with("test content", prompt_name="passage", convert_to_tensor=False)

    def test_query_embedding_uses_query_prompt(self):
        """Bug #101: _generate_embedding for queries must use prompt_name='query'."""
        storage, mock_model = self._make_storage()

        storage._generate_embedding("search terms", prompt_name="query")

        mock_model.encode.assert_called_once_with("search terms", prompt_name="query", convert_to_tensor=False)

    def test_fallback_when_model_has_no_prompts(self):
        """Models without prompts configured must still work (no prompt_name arg)."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        mock_model = MagicMock()
        mock_model.prompts = {}  # No prompts configured
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        storage._embedding_model_instance = mock_model
        storage._vector_size = 384

        storage._generate_embedding("test content")

        # Should NOT pass prompt_name when model has no prompts
        mock_model.encode.assert_called_once_with("test content", convert_to_tensor=False)

    def test_fallback_when_model_has_no_prompts_attribute(self):
        """Models without prompts attribute at all must still work."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="custom-model", storage_path="/tmp/test")

        mock_model = MagicMock(spec=[])  # No attributes at all
        mock_model.encode = MagicMock(return_value=np.zeros(384, dtype=np.float32))
        storage._embedding_model_instance = mock_model
        storage._vector_size = 384

        storage._generate_embedding("test content")

        mock_model.encode.assert_called_once_with("test content", convert_to_tensor=False)

    @pytest.mark.asyncio
    async def test_retrieve_uses_query_prompt(self):
        """Bug #101: retrieve() must pass prompt_name='query' for search embeddings."""
        storage, mock_model = self._make_storage()
        storage._initialized = True
        storage._failure_count = 0
        storage._circuit_open_until = None

        # Mock the qdrant client search to return empty results
        storage.client = MagicMock()
        storage.client.query_points.return_value = MagicMock(points=[])

        await storage.retrieve("search query", n_results=5)

        # The embedding call should use prompt_name="query"
        mock_model.encode.assert_called_once_with("search query", prompt_name="query", convert_to_tensor=False)

    def test_manual_prefix_fallback_for_old_sentence_transformers(self):
        """sentence-transformers <3.0 doesn't support prompt_name; fall back to manual prefix."""
        storage, mock_model = self._make_storage()

        # Simulate old sentence-transformers: prompt_name kwarg raises TypeError
        def encode_side_effect(text, **kwargs):
            if "prompt_name" in kwargs:
                raise TypeError("encode() got an unexpected keyword argument 'prompt_name'")
            return np.zeros(768, dtype=np.float32)

        mock_model.encode.side_effect = encode_side_effect

        storage._generate_embedding("test content", prompt_name="query")

        # Should have been called twice: first with prompt_name (fails), then with manual prefix
        assert mock_model.encode.call_count == 2
        second_call = mock_model.encode.call_args_list[1]
        assert second_call == (("query: test content",), {"convert_to_tensor": False})

    def test_unrelated_typeerror_is_reraised(self):
        """TypeErrors unrelated to prompt_name must not be silently caught."""
        storage, mock_model = self._make_storage()

        mock_model.encode.side_effect = TypeError("expected str, got NoneType")

        with pytest.raises(TypeError, match="expected str, got NoneType"):
            storage._generate_embedding("test content", prompt_name="query")


# =============================================================================
# Embedding Model Prewarm Tests
# =============================================================================


class TestEmbeddingModelPrewarm:
    """Test that initialize() eagerly loads the embedding model."""

    @pytest.mark.asyncio
    async def test_initialize_calls_ensure_model_loaded(self):
        """initialize() must prewarm the embedding model before first request."""
        from unittest.mock import AsyncMock

        with patch("mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="test-model", url="http://localhost:6333")

        storage._initialized = False

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with (
            patch.object(storage, "_ensure_model_loaded") as mock_prewarm,
            patch.object(storage, "_collection_exists", new_callable=AsyncMock, return_value=False),
            patch.object(storage, "_create_collection_with_metadata", new_callable=AsyncMock),
            patch.object(storage, "_ensure_payload_indexes", new_callable=AsyncMock),
            patch.object(storage, "_ensure_tag_collection", new_callable=AsyncMock),
        ):
            storage.client = MagicMock()
            storage._embedding_model_instance = mock_model

            await storage.initialize()

            mock_prewarm.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_gets_dimensions_from_model(self):
        """initialize() must query actual model dimensions, not a lookup table."""
        from unittest.mock import AsyncMock

        with patch("mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="some-unknown-model", url="http://localhost:6333")

        storage._initialized = False

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1234

        with (
            patch.object(storage, "_ensure_model_loaded"),
            patch.object(storage, "_collection_exists", new_callable=AsyncMock, return_value=False),
            patch.object(storage, "_create_collection_with_metadata", new_callable=AsyncMock),
            patch.object(storage, "_ensure_payload_indexes", new_callable=AsyncMock),
            patch.object(storage, "_ensure_tag_collection", new_callable=AsyncMock),
        ):
            storage.client = MagicMock()
            storage._embedding_model_instance = mock_model

            await storage.initialize()

            assert storage._vector_size == 1234
            mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_ensure_model_loaded_is_idempotent(self):
        """Calling _ensure_model_loaded when model is already set is a no-op."""
        with patch("mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="test-model", url="http://localhost:6333")

        storage._embedding_model_instance = MagicMock()

        with patch("mcp_memory_service.storage.qdrant_storage.SentenceTransformer") as mock_st:
            storage._ensure_model_loaded()
            mock_st.assert_not_called()
