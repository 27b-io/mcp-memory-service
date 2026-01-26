"""
Normalized Discounted Cumulative Gain (NDCG@K) evaluation for hybrid search.

NDCG measures ranking quality, accounting for the position of relevant results.
Higher positions are weighted more heavily.
"""

import os
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .conftest import (
    SQLITE_VEC_AVAILABLE,
    get_test_cases,
    calculate_ndcg,
)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestNDCG:
    """NDCG@10 evaluation tests."""

    @pytest.mark.asyncio
    async def test_ndcg_tag_sensitive(self, eval_service):
        """Evaluate NDCG@10 for tag-sensitive queries."""
        test_cases = get_test_cases(category="tag_sensitive")
        ndcg_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(
                query=tc["query"],
                page=1,
                page_size=10
            )
            ndcg = calculate_ndcg(result["memories"], tc["expected_hashes"], k=10)
            ndcg_scores.append(ndcg)

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        print(f"\nNDCG@10 (tag_sensitive): {avg_ndcg:.3f} ({len(ndcg_scores)} queries)")

        assert avg_ndcg > 0.0, "NDCG should be non-zero"

    @pytest.mark.asyncio
    async def test_ndcg_semantic(self, eval_service):
        """Evaluate NDCG@10 for semantic queries."""
        test_cases = get_test_cases(category="semantic")
        ndcg_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(
                query=tc["query"],
                page=1,
                page_size=10
            )
            ndcg = calculate_ndcg(result["memories"], tc["expected_hashes"], k=10)
            ndcg_scores.append(ndcg)

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        print(f"\nNDCG@10 (semantic): {avg_ndcg:.3f} ({len(ndcg_scores)} queries)")

    @pytest.mark.asyncio
    async def test_ndcg_overall(self, eval_service):
        """Evaluate overall NDCG@10."""
        test_cases = get_test_cases()
        ndcg_scores = []

        for tc in test_cases:
            result = await eval_service.retrieve_memories(
                query=tc["query"],
                page=1,
                page_size=10
            )
            ndcg = calculate_ndcg(result["memories"], tc["expected_hashes"], k=10)
            ndcg_scores.append(ndcg)

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        print(f"\nOverall NDCG@10: {avg_ndcg:.3f} ({len(ndcg_scores)} queries)")

        assert avg_ndcg > 0.0, "Overall NDCG should be non-zero"

    @pytest.mark.asyncio
    async def test_ndcg_k_values(self, eval_service):
        """Evaluate NDCG at different K values."""
        test_cases = get_test_cases()

        for k in [1, 3, 5, 10]:
            ndcg_scores = []
            for tc in test_cases:
                result = await eval_service.retrieve_memories(
                    query=tc["query"],
                    page=1,
                    page_size=k
                )
                ndcg = calculate_ndcg(result["memories"], tc["expected_hashes"], k=k)
                ndcg_scores.append(ndcg)

            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            print(f"NDCG@{k}: {avg_ndcg:.3f}")

    @pytest.mark.asyncio
    async def test_ndcg_per_category(self, eval_service):
        """Evaluate NDCG@10 broken down by category."""
        categories = ["tag_sensitive", "semantic", "mixed"]

        for category in categories:
            test_cases = get_test_cases(category=category)
            if not test_cases:
                continue

            ndcg_scores = []
            for tc in test_cases:
                result = await eval_service.retrieve_memories(
                    query=tc["query"],
                    page=1,
                    page_size=10
                )
                ndcg = calculate_ndcg(result["memories"], tc["expected_hashes"], k=10)
                ndcg_scores.append(ndcg)

            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            print(f"NDCG@10 ({category}): {avg_ndcg:.3f} ({len(ndcg_scores)} queries)")
