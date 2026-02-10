"""
RAGAS NonLLM metrics evaluation for retrieval quality.

Uses NonLLMContextRecall and NonLLMContextPrecisionWithReference —
Levenshtein-based string comparison, no LLM judge needed.

Known limitations:
    - Levenshtein distance != semantic similarity. Short ground truth
      strings match well, but semantic paraphrases will undercount.
    - These metrics complement (not replace) our hit_rate/mrr/ndcg metrics.
"""

import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ragas = pytest.importorskip("ragas", reason="ragas not installed (install with: uv sync --group eval)")
rapidfuzz = pytest.importorskip("rapidfuzz", reason="rapidfuzz not installed (install with: uv sync --group eval)")

from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

from .conftest import build_content_map, get_test_cases, memories_to_contexts


class TestRagasNonLLM:
    """RAGAS NonLLM context metrics — no LLM judge required."""

    @pytest.mark.asyncio
    async def test_context_recall_by_category(self, eval_service):
        """NonLLMContextRecall per category.

        Measures: what fraction of reference contexts were retrieved?
        """
        content_map = build_content_map()
        metric = NonLLMContextRecall()

        for category in ("tag_sensitive", "semantic", "mixed"):
            cases = get_test_cases(category=category)
            scores = []

            for tc in cases:
                # Skip edge cases with no expected results
                if not tc["expected_hashes"]:
                    continue

                result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
                retrieved = memories_to_contexts(result["memories"])
                reference = [content_map[h] for h in tc["expected_hashes"] if h in content_map]

                if not reference:
                    continue

                sample = SingleTurnSample(
                    retrieved_contexts=retrieved,
                    reference_contexts=reference,
                )
                score = await metric.single_turn_ascore(sample)
                scores.append(score)

            if scores:
                avg = sum(scores) / len(scores)
                print(f"\nNonLLMContextRecall ({category}): {avg:.3f} ({len(scores)} cases)")
                # Sanity: at least some recall for non-edge categories
                assert avg > 0.0, f"Context recall for {category} should be non-zero"

    @pytest.mark.asyncio
    async def test_context_precision_by_category(self, eval_service):
        """NonLLMContextPrecisionWithReference per category.

        Measures: what fraction of retrieved contexts are relevant?
        """
        content_map = build_content_map()
        metric = NonLLMContextPrecisionWithReference()

        for category in ("tag_sensitive", "semantic", "mixed"):
            cases = get_test_cases(category=category)
            scores = []

            for tc in cases:
                if not tc["expected_hashes"]:
                    continue

                result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
                retrieved = memories_to_contexts(result["memories"])
                reference = [content_map[h] for h in tc["expected_hashes"] if h in content_map]

                if not reference or not retrieved:
                    continue

                sample = SingleTurnSample(
                    retrieved_contexts=retrieved,
                    reference_contexts=reference,
                )
                score = await metric.single_turn_ascore(sample)
                scores.append(score)

            if scores:
                avg = sum(scores) / len(scores)
                print(f"\nNonLLMContextPrecision ({category}): {avg:.3f} ({len(scores)} cases)")
                assert avg >= 0.0, f"Context precision for {category} should be non-negative"

    @pytest.mark.asyncio
    async def test_overall_ragas_scores(self, eval_service):
        """Aggregate RAGAS scores across all non-edge test cases."""
        content_map = build_content_map()
        recall_metric = NonLLMContextRecall()
        precision_metric = NonLLMContextPrecisionWithReference()

        recall_scores = []
        precision_scores = []

        for tc in get_test_cases():
            if not tc["expected_hashes"]:
                continue

            result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)
            retrieved = memories_to_contexts(result["memories"])
            reference = [content_map[h] for h in tc["expected_hashes"] if h in content_map]

            if not reference or not retrieved:
                continue

            sample = SingleTurnSample(
                retrieved_contexts=retrieved,
                reference_contexts=reference,
            )
            recall_scores.append(await recall_metric.single_turn_ascore(sample))
            precision_scores.append(await precision_metric.single_turn_ascore(sample))

        n = len(recall_scores)
        avg_recall = sum(recall_scores) / n if n else 0.0
        avg_precision = sum(precision_scores) / n if n else 0.0

        print(f"\n{'=' * 60}")
        print(f"RAGAS Overall ({n} cases):")
        print(f"  NonLLMContextRecall:    {avg_recall:.3f}")
        print(f"  NonLLMContextPrecision: {avg_precision:.3f}")
        print(f"{'=' * 60}")

        assert n > 0, "Should have evaluated at least one test case"
        assert avg_recall > 0.0, "Overall recall should be non-zero"
