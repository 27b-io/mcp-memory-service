"""
Integration tests for all 9 MCP tools exercised through the FastMCP Client interface.

Tests the full pipeline: MCP tool → MemoryService → QdrantStorage (embedded mode).
No external services required — Qdrant runs in embedded/local mode, graph layer is None.
"""

import hashlib
import json
import random
import shutil

import numpy as np
import pytest
from fastmcp import Client
from src.mcp_memory_service.mcp_server import mcp
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_embedding(text: str, vector_size: int = 384) -> list[float]:
    """Create a deterministic 384-dim embedding from text hash."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.random() * 2 - 1 for _ in range(vector_size)]


def parse_tool_result(result) -> dict | str:
    """Parse a FastMCP CallToolResult into a dict (JSON) or a raw string (plain text)."""
    text = result.content[0].text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


async def _invalidate_tag_cache() -> None:
    """Clear the CacheKit tag cache so hybrid search sees freshly stored tags.

    The memory service caches get_all_tags() with a 60s TTL via CacheKit.
    In tests where we store a memory with a new tag and immediately do a
    hybrid search, the stale cache would miss the new tag.
    """
    try:
        from src.mcp_memory_service.services.memory_service import _cached_fetch_all_tags

        await _cached_fetch_all_tags.ainvalidate_cache()
    except (ImportError, AttributeError):
        pass  # CacheKit not available or function doesn't exist — no-op


# ---------------------------------------------------------------------------
# Module-scoped storage (survives across all tests in this file)
# ---------------------------------------------------------------------------

# Shared state across the module — avoids re-init per test
_shared_storage: QdrantStorage | None = None
_shared_storage_path = None


async def _get_or_create_storage(tmp_path_factory) -> QdrantStorage:
    """Lazily create and cache a single QdrantStorage for the module."""
    global _shared_storage, _shared_storage_path
    if _shared_storage is not None:
        return _shared_storage

    storage_path = tmp_path_factory.mktemp("qdrant_mcp_tools")
    _shared_storage_path = storage_path

    storage = QdrantStorage(
        storage_path=str(storage_path),
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_mcp_tools",
    )

    # Mock embeddings — no model download required
    # _generate_embedding accepts prompt_name kwarg for instruction-tuned models
    storage._generate_embedding = lambda text, prompt_name="passage": deterministic_embedding(text)

    async def mock_query_embedding(query: str) -> list[float]:
        return deterministic_embedding(query)

    storage._generate_query_embedding = mock_query_embedding

    # Mock generate_embeddings_batch (used by find_duplicates)
    async def mock_batch(texts, prompt_name="query"):
        return [deterministic_embedding(t) for t in texts]

    storage.generate_embeddings_batch = mock_batch

    # Mock embedding_service.encode (fallback path)
    class MockEmbeddingService:
        def encode(self, texts, convert_to_numpy=False, **kwargs):
            vecs = [deterministic_embedding(t) for t in texts]
            if convert_to_numpy:
                return np.array(vecs)
            return vecs

    storage.embedding_service = MockEmbeddingService()

    await storage.initialize()
    _shared_storage = storage
    return storage


def _patch_shared_storage(monkeypatch, storage):
    """Monkeypatch the shared_storage module so mcp lifespan uses our test storage."""
    import src.mcp_memory_service.shared_storage as shared_mod

    monkeypatch.setattr(shared_mod, "is_storage_initialized", lambda: True)

    async def _get_storage():
        return storage

    monkeypatch.setattr(shared_mod, "get_shared_storage", _get_storage)
    monkeypatch.setattr(shared_mod, "get_graph_client", lambda: None)
    monkeypatch.setattr(shared_mod, "get_write_queue", lambda: None)


# ---------------------------------------------------------------------------
# Fixtures (function-scoped to work with pytest-asyncio event loop)
# ---------------------------------------------------------------------------


@pytest.fixture
async def qdrant_storage(tmp_path_factory, monkeypatch):
    """Real Qdrant embedded storage with deterministic mock embeddings."""
    storage = await _get_or_create_storage(tmp_path_factory)
    _patch_shared_storage(monkeypatch, storage)
    return storage


@pytest.fixture
async def mcp_client(qdrant_storage):
    """FastMCP in-process client wired to the test storage."""
    async with Client(mcp) as client:
        yield client


# ---------------------------------------------------------------------------
# store_memory
# ---------------------------------------------------------------------------


class TestStoreMemory:
    """Tests for the store_memory MCP tool."""

    async def test_store_basic(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("store_memory", {"content": "The sky is blue"}))
        assert result["success"] is True
        assert isinstance(result["content_hash"], str)
        assert len(result["content_hash"]) > 8

    async def test_store_with_tags_and_metadata(self, mcp_client):
        result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {
                    "content": "Python uses indentation for blocks",
                    "tags": ["python", "syntax"],
                    "metadata": {"importance": 0.8},
                },
            )
        )
        assert result["success"] is True

    async def test_store_readback_preserves_tags(self, mcp_client):
        """Store with tags, then retrieve by tag to verify tags and content are preserved."""
        unique_tag = "readback-verify-tag-abc"
        content = "Readback verification: tags and metadata must survive roundtrip"
        store_result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {
                    "content": content,
                    "tags": [unique_tag, "readback-extra"],
                    "metadata": {"importance": 0.9},
                },
            )
        )
        assert store_result["success"] is True
        content_hash = store_result["content_hash"]

        # Retrieve by tag — tag mode doesn't use embeddings, always works
        search_result = parse_tool_result(
            await mcp_client.call_tool("search", {"query": "", "mode": "tag", "tags": [unique_tag]})
        )
        assert isinstance(search_result, str)
        assert content_hash[:12] in search_result
        # Verify at least one of the stored tags appears in the plain text output
        assert unique_tag in search_result or "readback-extra" in search_result

    async def test_store_comma_separated_tags(self, mcp_client):
        result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {"content": "Comma tags test memory", "tags": "alpha,beta,gamma"},
            )
        )
        assert result["success"] is True

    async def test_store_with_summary(self, mcp_client):
        result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {
                    "content": "A detailed explanation of how TCP handshakes work in networking",
                    "summary": "TCP handshake overview",
                },
            )
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def _seed_search(mcp_client):
    """Ensure at least one searchable memory exists."""
    await mcp_client.call_tool(
        "store_memory",
        {"content": "Searchable memory for integration tests", "tags": ["search-test"]},
    )


class TestSearch:
    """Tests for the search MCP tool (5 modes)."""

    async def test_search_hybrid(self, mcp_client):
        """Hybrid search uses tag-boosted RRF. Store with a unique tag, query with that
        tag word so extract_query_keywords matches it and search_by_tags finds it."""
        unique_tag = "hybridunique"
        store_result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {"content": "Hybrid search verification with hybridunique marker", "tags": [unique_tag]},
            )
        )
        content_hash = store_result["content_hash"]

        # Invalidate the CacheKit tag cache so the freshly stored tag is visible
        await _invalidate_tag_cache()

        # Query contains the tag word — extract_query_keywords will match it,
        # search_by_tags will find the memory, RRF includes it in results.
        # Use min_similarity=0.0 because tag-only hits get a base score of 0.1
        # which is below the default threshold of 0.3.
        result = parse_tool_result(
            await mcp_client.call_tool("search", {"query": "hybridunique marker", "mode": "hybrid", "min_similarity": 0.0})
        )
        assert isinstance(result, str)
        # The plain text output must contain our stored memory's hash prefix
        assert (
            content_hash[:12] in result
        ), f"Hybrid search should find memory {content_hash[:12]} via tag-boosted RRF, got: {result[:200]}"

    async def test_search_tag_mode(self, mcp_client):
        await _seed_search(mcp_client)
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": "", "mode": "tag", "tags": ["search-test"]}))
        assert isinstance(result, str)

    async def test_search_recent_mode(self, mcp_client):
        await _seed_search(mcp_client)
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": "", "mode": "recent"}))
        assert isinstance(result, str)

    async def test_search_scan_mode(self, mcp_client):
        """Scan mode uses vector search. Same text → same embedding → cosine sim 1.0."""
        exact_content = "Scan mode exact match verification content zxy987"
        store_result = parse_tool_result(await mcp_client.call_tool("store_memory", {"content": exact_content}))
        stored_hash = store_result["content_hash"]

        # Search for the EXACT same content — deterministic embedding → similarity 1.0
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": exact_content, "mode": "scan"}))
        assert isinstance(result, dict)
        assert result.get("count", 0) > 0, f"Scan should find exact-match memory, got: {result}"
        found_hashes = [r["content_hash"] for r in result.get("results", [])]
        assert (
            stored_hash in found_hashes
        ), f"Scan results should contain {stored_hash[:12]}, got hashes: {[h[:12] for h in found_hashes]}"

    async def test_search_similar_mode(self, mcp_client):
        """Similar mode uses pure k-NN vector search. Same text → same embedding → similarity 1.0."""
        exact_content = "Similar mode exact match verification content qrs456"
        store_result = parse_tool_result(await mcp_client.call_tool("store_memory", {"content": exact_content}))
        stored_hash = store_result["content_hash"]

        result = parse_tool_result(await mcp_client.call_tool("search", {"query": exact_content, "mode": "similar"}))
        assert isinstance(result, dict)
        assert result.get("count", 0) > 0, f"Similar should find exact-match memory, got: {result}"
        found_hashes = [m["content_hash"] for m in result.get("memories", [])]
        assert (
            stored_hash in found_hashes
        ), f"Similar results should contain {stored_hash[:12]}, got hashes: {[h[:12] for h in found_hashes]}"

    async def test_search_invalid_mode(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": "anything", "mode": "bogus"}))
        assert isinstance(result, dict)
        assert "error" in result

    async def test_search_hybrid_no_query(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": "", "mode": "hybrid"}))
        assert isinstance(result, dict)
        assert "error" in result
        assert "query" in result["error"].lower() or "required" in result["error"].lower()

    async def test_search_tag_no_tags(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("search", {"query": "", "mode": "tag"}))
        assert isinstance(result, dict)
        assert "error" in result
        assert "tags" in result["error"].lower()


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------


class TestDeleteMemory:
    """Tests for the delete_memory MCP tool."""

    async def test_delete_success(self, mcp_client):
        store_result = parse_tool_result(await mcp_client.call_tool("store_memory", {"content": "Memory to delete in test"}))
        content_hash = store_result["content_hash"]
        delete_result = parse_tool_result(await mcp_client.call_tool("delete_memory", {"content_hash": content_hash}))
        assert delete_result["success"] is True

    async def test_delete_nonexistent(self, mcp_client):
        # Qdrant delete is idempotent — nonexistent hashes return success
        result = parse_tool_result(await mcp_client.call_tool("delete_memory", {"content_hash": "nonexistent_hash_abc123"}))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# check_database_health
# ---------------------------------------------------------------------------


class TestCheckDatabaseHealth:
    """Tests for the check_database_health MCP tool."""

    async def test_health_check(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("check_database_health", {}))
        assert isinstance(result, dict)
        assert result["status"] == "operational", f"Expected operational status, got: {result.get('status')}"
        assert isinstance(result.get("total_memories"), int)


# ---------------------------------------------------------------------------
# relation (graph layer = None → graceful degradation)
# ---------------------------------------------------------------------------


class TestRelation:
    """Tests for the relation MCP tool with no graph layer."""

    async def test_relation_create_no_graph(self, mcp_client):
        result = parse_tool_result(
            await mcp_client.call_tool(
                "relation",
                {
                    "action": "create",
                    "content_hash": "abc",
                    "target_hash": "def",
                    "relation_type": "RELATES_TO",
                },
            )
        )
        assert result["success"] is False
        assert "graph" in result.get("error", "").lower() or "not enabled" in result.get("error", "").lower()

    async def test_relation_get_no_graph(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("relation", {"action": "get", "content_hash": "abc"}))
        assert result["relations"] == []
        assert result["content_hash"] == "abc"

    async def test_relation_delete_no_graph(self, mcp_client):
        result = parse_tool_result(
            await mcp_client.call_tool(
                "relation",
                {
                    "action": "delete",
                    "content_hash": "abc",
                    "target_hash": "def",
                    "relation_type": "RELATES_TO",
                },
            )
        )
        assert result["success"] is False
        assert "graph" in result.get("error", "").lower() or "not enabled" in result.get("error", "").lower()

    async def test_relation_invalid_action(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("relation", {"action": "bogus", "content_hash": "abc"}))
        assert result["success"] is False
        assert "create" in result.get("error", "").lower() or "literal_error" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# memory_supersede
# ---------------------------------------------------------------------------


async def _store_via_tool(mcp_client, content: str) -> str:
    """Store a memory via the MCP tool and return its content_hash."""
    result = parse_tool_result(await mcp_client.call_tool("store_memory", {"content": content}))
    return result["content_hash"]


class TestMemorySupersede:
    """Tests for the memory_supersede MCP tool."""

    async def test_supersede_success(self, mcp_client):
        hash_a = await _store_via_tool(mcp_client, "Old fact: the capital of Australia is Sydney")
        hash_b = await _store_via_tool(mcp_client, "New fact: the capital of Australia is Canberra")
        result = parse_tool_result(
            await mcp_client.call_tool(
                "memory_supersede",
                {"old_id": hash_a, "new_id": hash_b, "reason": "corrected"},
            )
        )
        assert result["success"] is True
        assert result["superseded"] == hash_a
        assert result["superseded_by"] == hash_b

    async def test_supersede_same_id(self, mcp_client):
        hash_a = await _store_via_tool(mcp_client, "Self-referencing supersede test")
        result = parse_tool_result(await mcp_client.call_tool("memory_supersede", {"old_id": hash_a, "new_id": hash_a}))
        assert result["success"] is False
        assert "different" in result.get("error", "").lower()

    async def test_supersede_nonexistent(self, mcp_client):
        hash_a = await _store_via_tool(mcp_client, "Real memory for supersede test")
        result = parse_tool_result(
            await mcp_client.call_tool(
                "memory_supersede",
                {"old_id": "nonexistent_hash_xyz", "new_id": hash_a},
            )
        )
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# memory_contradictions
# ---------------------------------------------------------------------------


class TestMemoryContradictions:
    """Tests for the memory_contradictions MCP tool."""

    async def test_contradictions_no_graph(self, mcp_client):
        result = parse_tool_result(await mcp_client.call_tool("memory_contradictions", {}))
        assert result["success"] is False
        assert result["pairs"] == []
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    """Tests for the find_duplicates MCP tool."""

    async def test_find_duplicates_no_dupes(self, mcp_client):
        await mcp_client.call_tool("store_memory", {"content": "Unique memory about quantum mechanics"})
        await mcp_client.call_tool("store_memory", {"content": "Unique memory about medieval history"})
        result = parse_tool_result(await mcp_client.call_tool("find_duplicates", {"similarity_threshold": 0.99}))
        assert result["success"] is True
        assert isinstance(result["groups"], list)
        assert isinstance(result["total_memories_scanned"], int)

    async def test_find_duplicates_with_dupes(self, mcp_client, qdrant_storage):
        """Two memories with identical embeddings should be detected as duplicates.

        The deterministic mock produces different vectors for different texts,
        so we temporarily override generate_embeddings_batch to return the same
        vector for all inputs — simulating near-identical semantic content.
        """
        hash_a = await _store_via_tool(mcp_client, "Duplicate detection test memory alpha")
        hash_b = await _store_via_tool(mcp_client, "Duplicate detection test memory beta")

        # Temporarily make batch embeddings return identical vectors so
        # find_duplicates sees similarity=1.0 between all memories
        original_batch = qdrant_storage.generate_embeddings_batch
        fixed_vector = deterministic_embedding("identical-for-dedup-test")

        async def identical_batch(texts, prompt_name="query"):
            return [fixed_vector for _ in texts]

        qdrant_storage.generate_embeddings_batch = identical_batch
        try:
            result = parse_tool_result(await mcp_client.call_tool("find_duplicates", {"similarity_threshold": 0.85}))
        finally:
            qdrant_storage.generate_embeddings_batch = original_batch

        assert result["success"] is True
        assert result["total_duplicates_found"] > 0, f"Should detect duplicates when all embeddings are identical, got: {result}"
        assert len(result["groups"]) > 0, "Should have at least one duplicate group"
        # Verify both memories appear in the same duplicate group
        shared_group = None
        for group in result["groups"]:
            if hash_a in group["hashes"] and hash_b in group["hashes"]:
                shared_group = group
                break
        assert (
            shared_group is not None
        ), f"hash_a and hash_b should be in the same duplicate group, but groups were: {result['groups']}"


# ---------------------------------------------------------------------------
# merge_duplicates
# ---------------------------------------------------------------------------


class TestMergeDuplicates:
    """Tests for the merge_duplicates MCP tool."""

    async def test_merge_duplicates_dry_run(self, mcp_client, qdrant_storage):
        hash_a = await _store_via_tool(mcp_client, "Merge dry-run canonical memory")
        hash_b = await _store_via_tool(mcp_client, "Merge dry-run duplicate memory")
        result = parse_tool_result(
            await mcp_client.call_tool(
                "merge_duplicates",
                {"canonical_hash": hash_a, "duplicate_hashes": [hash_b], "dry_run": True},
            )
        )
        assert result["success"] is True
        assert result["dry_run"] is True
        # Verify the duplicate was NOT actually superseded
        mem = await qdrant_storage.get_memory_by_hash(hash_b)
        assert mem is not None
        assert not (mem.metadata or {}).get("superseded_by")

    async def test_merge_duplicates_real(self, mcp_client):
        hash_a = await _store_via_tool(mcp_client, "Merge real canonical memory")
        hash_b = await _store_via_tool(mcp_client, "Merge real duplicate memory")
        result = parse_tool_result(
            await mcp_client.call_tool(
                "merge_duplicates",
                {"canonical_hash": hash_a, "duplicate_hashes": [hash_b]},
            )
        )
        assert result["success"] is True
        assert hash_b in result.get("superseded", [])


# ---------------------------------------------------------------------------
# Cross-tool workflows
# ---------------------------------------------------------------------------


class TestCrossToolWorkflows:
    """End-to-end workflows exercising multiple tools in sequence."""

    async def test_workflow_store_search_delete_verify(self, mcp_client):
        """Store → search (found) → delete → search (gone).

        Uses tag mode because deterministic hash-based mock embeddings
        produce random vectors — cosine similarity between different texts
        is ~0, making hybrid search unreliable in tests.
        """
        unique_tag = "workflow-ephemeral-xyz789"
        store_result = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {"content": "Workflow test: ephemeral memory xyz789", "tags": [unique_tag]},
            )
        )
        content_hash = store_result["content_hash"]

        # Search by tag — should find it
        search_result = parse_tool_result(
            await mcp_client.call_tool("search", {"query": "", "mode": "tag", "tags": [unique_tag]})
        )
        assert isinstance(search_result, str)
        assert content_hash[:12] in search_result

        # Delete
        delete_result = parse_tool_result(await mcp_client.call_tool("delete_memory", {"content_hash": content_hash}))
        assert delete_result["success"] is True

        # Search again — should be gone
        search_after = parse_tool_result(
            await mcp_client.call_tool("search", {"query": "", "mode": "tag", "tags": [unique_tag]})
        )
        assert isinstance(search_after, str)
        assert content_hash[:12] not in search_after

    async def test_workflow_store_increments_health_count(self, mcp_client):
        """health(N) → store → health(N+1)."""
        before = parse_tool_result(await mcp_client.call_tool("check_database_health", {}))
        n_before = before["total_memories"]

        await mcp_client.call_tool("store_memory", {"content": "Health count increment test memory"})

        after = parse_tool_result(await mcp_client.call_tool("check_database_health", {}))
        assert after["total_memories"] == n_before + 1

    async def test_workflow_supersede_excludes_from_search(self, mcp_client):
        """Store A+B with unique tag, supersede A→B, verify:
        - hybrid search (include_superseded=False): A excluded, B present
        - hybrid search (include_superseded=True): A present (regression test)

        Uses tag-activated hybrid search so extract_query_keywords matches and
        search_by_tags finds results via RRF (not reliant on vector similarity).
        """
        unique_tag = "supersede-excl-test"
        hash_a = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {"content": "Supersede exclude STALE supersede-excl-test aaa", "tags": [unique_tag]},
            )
        )["content_hash"]
        hash_b = parse_tool_result(
            await mcp_client.call_tool(
                "store_memory",
                {"content": "Supersede exclude CURRENT supersede-excl-test bbb", "tags": [unique_tag]},
            )
        )["content_hash"]

        # Verify both are findable before superseding (via tag search)
        pre_search = parse_tool_result(await mcp_client.call_tool("search", {"query": "", "mode": "tag", "tags": [unique_tag]}))
        assert hash_a[:12] in pre_search, "A should be findable before supersede"
        assert hash_b[:12] in pre_search, "B should be findable before supersede"

        # Supersede A → B
        sup_result = parse_tool_result(
            await mcp_client.call_tool(
                "memory_supersede",
                {"old_id": hash_a, "new_id": hash_b, "reason": "updated"},
            )
        )
        assert sup_result["success"] is True

        # Invalidate tag cache so hybrid search sees the freshly stored tags
        await _invalidate_tag_cache()

        # Hybrid search with include_superseded=False (default) — A should be excluded.
        # min_similarity=0.0 because tag-only hits get base score 0.1 < default 0.3.
        # page_size=50 to capture all results on one page (test DB has ~20 memories).
        search_default = parse_tool_result(
            await mcp_client.call_tool(
                "search",
                {
                    "query": "supersede-excl-test",
                    "mode": "hybrid",
                    "include_superseded": False,
                    "min_similarity": 0.0,
                    "page_size": 50,
                },
            )
        )
        assert isinstance(search_default, str)
        assert hash_b[:12] in search_default, f"B should appear in default hybrid search, got: {search_default[:300]}"
        assert (
            hash_a[:12] not in search_default
        ), f"Superseded A should be excluded from default search, got: {search_default[:300]}"

        # Hybrid search with include_superseded=True — A should be present
        search_with_superseded = parse_tool_result(
            await mcp_client.call_tool(
                "search",
                {
                    "query": "supersede-excl-test",
                    "mode": "hybrid",
                    "include_superseded": True,
                    "min_similarity": 0.0,
                    "page_size": 50,
                },
            )
        )
        assert isinstance(search_with_superseded, str)
        assert (
            hash_a[:12] in search_with_superseded
        ), f"Superseded A should appear with include_superseded=True, got: {search_with_superseded[:300]}"


# ---------------------------------------------------------------------------
# Module teardown
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
async def _teardown_storage():
    """Close the shared storage after all tests in this module."""
    yield
    global _shared_storage, _shared_storage_path
    if _shared_storage is not None:
        await _shared_storage.close()
        _shared_storage = None
    if _shared_storage_path is not None and _shared_storage_path.exists():
        shutil.rmtree(_shared_storage_path, ignore_errors=True)
        _shared_storage_path = None
