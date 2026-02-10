"""
Evaluation harness fixtures and utilities.

Provides ground truth loading and storage setup for retrieval quality evaluation.
"""

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.qdrant_storage import QdrantStorage

# =============================================================================
# Ground Truth Utilities
# =============================================================================


def load_ground_truth() -> dict:
    """Load ground truth test cases from JSON."""
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def get_test_cases(category: str = None) -> list[dict]:
    """Get test cases, optionally filtered by category."""
    gt = load_ground_truth()
    cases = gt.get("test_cases", [])
    if category:
        cases = [c for c in cases if c.get("category") == category]
    return cases


def build_content_map() -> dict[str, str]:
    """Build a mapping from content_hash to content string from ground truth.

    Used by RAGAS tests to convert hash-based expected results into
    the string-based reference_contexts format RAGAS expects.
    """
    gt = load_ground_truth()
    return {m["content_hash"]: m["content"] for m in gt.get("memories", [])}


def memories_to_contexts(memories: list[dict]) -> list[str]:
    """Extract content strings from retrieve_memories result list.

    Converts the dict-based retrieval output to the list[str] format
    that RAGAS SingleTurnSample expects for retrieved_contexts.
    """
    return [m["content"] for m in memories if "content" in m]


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
async def eval_storage() -> AsyncGenerator[QdrantStorage, None]:
    """Create Qdrant storage seeded with evaluation test data."""
    temp_dir = tempfile.mkdtemp()
    storage_path = os.path.join(temp_dir, "eval_qdrant")

    storage = QdrantStorage(
        storage_path=storage_path,
        embedding_model="all-MiniLM-L6-v2",
        collection_name=f"eval_{uuid.uuid4().hex[:8]}",
    )
    await storage.initialize()

    # Seed with ground truth memories
    gt = load_ground_truth()
    for memory_data in gt.get("memories", []):
        memory = Memory(
            content=memory_data["content"],
            content_hash=memory_data["content_hash"],
            tags=memory_data.get("tags", []),
            memory_type=memory_data.get("memory_type", "note"),
        )
        await storage.store(memory)

    yield storage

    await storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def eval_service(eval_storage) -> MemoryService:
    """Create MemoryService for evaluation."""
    return MemoryService(eval_storage)


# =============================================================================
# Metrics Utilities
# =============================================================================


def calculate_hit_rate(results: list[dict], expected_hashes: list[str], k: int = 10) -> float:
    """
    Calculate Hit Rate@K.

    Hit Rate = 1 if any expected hash appears in top K results, else 0.
    """
    if not expected_hashes:
        return 0.0

    result_hashes = {r["content_hash"] for r in results[:k]}
    return 1.0 if any(h in result_hashes for h in expected_hashes) else 0.0


def calculate_mrr(results: list[dict], expected_hashes: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1 / position of first relevant result.
    """
    if not expected_hashes:
        return 0.0

    for i, result in enumerate(results):
        if result["content_hash"] in expected_hashes:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(results: list[dict], expected_hashes: list[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    import math

    if not expected_hashes:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        rel = 1.0 if result["content_hash"] in expected_hashes else 0.0
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed

    # Calculate ideal DCG (all relevant items at top)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_hashes), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
