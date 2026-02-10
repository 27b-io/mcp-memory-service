"""
DeepEval custom metrics for retrieval quality and interference detection.

Custom BaseMetric subclasses wrapping our existing metric functions.
No LLM judge needed — pure computation.

Metrics:
    HitRateMetric       — wraps calculate_hit_rate()
    MRRMetric           — wraps calculate_mrr()
    NDCGMetric          — wraps calculate_ndcg()
    InterferenceDetectionMetric — calls detect_contradiction_signals() directly
"""

import json
import os
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

deepeval = pytest.importorskip("deepeval", reason="deepeval not installed (install with: uv sync --group eval)")

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from mcp_memory_service.utils.interference import detect_contradiction_signals

from .conftest import calculate_hit_rate, calculate_mrr, calculate_ndcg, get_test_cases

# =============================================================================
# Custom Metrics
# =============================================================================


class HitRateMetric(BaseMetric):
    """Hit Rate@K wrapped as a DeepEval metric."""

    def __init__(self, k: int = 10, threshold: float = 0.5):
        self.k = k
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.additional_metadata or {}
        self.score = calculate_hit_rate(meta["results"], meta["expected_hashes"], k=self.k)
        self.success = self.score >= self.threshold
        self.reason = f"Hit@{self.k}: {'hit' if self.score > 0 else 'miss'}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return f"HitRate@{self.k}"


class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank wrapped as a DeepEval metric."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.additional_metadata or {}
        self.score = calculate_mrr(meta["results"], meta["expected_hashes"])
        self.success = self.score >= self.threshold
        self.reason = f"MRR: {self.score:.3f}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "MRR"


class NDCGMetric(BaseMetric):
    """NDCG@K wrapped as a DeepEval metric."""

    def __init__(self, k: int = 10, threshold: float = 0.3):
        self.k = k
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.additional_metadata or {}
        self.score = calculate_ndcg(meta["results"], meta["expected_hashes"], k=self.k)
        self.success = self.score >= self.threshold
        self.reason = f"NDCG@{self.k}: {self.score:.3f}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return f"NDCG@{self.k}"


class InterferenceDetectionMetric(BaseMetric):
    """Interference detection quality wrapped as a DeepEval metric.

    Scores signal type recall: what fraction of expected signal types
    were actually detected by detect_contradiction_signals()?
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.additional_metadata or {}
        signals = detect_contradiction_signals(
            new_content=meta["new_content"],
            existing_content=meta["existing_content"],
            existing_hash="test_hash",
            similarity=meta["similarity"],
        )
        detected_types = {s.signal_type for s in signals}
        expected_types = set(meta["expected_signals"])

        if not expected_types:
            # No signals expected — score 1.0 if none detected, 0.0 otherwise
            self.score = 1.0 if not detected_types else 0.0
            self.reason = f"Expected no signals, got: {detected_types or 'none'}"
        else:
            # Signal type recall
            hits = expected_types & detected_types
            self.score = len(hits) / len(expected_types)
            self.reason = f"Expected {expected_types}, detected {detected_types}, recall {self.score:.2f}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "InterferenceDetection"


# =============================================================================
# Load interference test cases
# =============================================================================


def _load_interference_cases() -> list[dict]:
    cases_path = Path(__file__).parent / "interference_cases.json"
    with open(cases_path) as f:
        data = json.load(f)
    return data["cases"]


# =============================================================================
# Retrieval Quality Tests
# =============================================================================


class TestDeepEvalRetrieval:
    """Retrieval quality metrics via DeepEval framework."""

    @staticmethod
    def _make_retrieval_ids() -> list[str]:
        """Generate test case IDs for parametrize."""
        cases = get_test_cases()
        return [tc["id"] for tc in cases if tc["expected_hashes"]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case_idx", range(len(get_test_cases())))
    async def test_retrieval_metrics(self, eval_service, case_idx):
        """Run HitRate, MRR, NDCG on each test case via DeepEval."""
        cases = get_test_cases()
        tc = cases[case_idx]

        # Skip edge cases with no expected results
        if not tc["expected_hashes"]:
            pytest.skip(f"Edge case {tc['id']} — no expected results")

        result = await eval_service.retrieve_memories(query=tc["query"], page=1, page_size=10)

        test_case = LLMTestCase(
            input=tc["query"],
            actual_output=f"Retrieved {len(result['memories'])} memories",
            additional_metadata={
                "results": result["memories"],
                "expected_hashes": tc["expected_hashes"],
                "case_id": tc["id"],
                "category": tc["category"],
            },
        )

        # Use lenient thresholds — this is a spike proving integration,
        # not setting performance gates
        metrics = [
            HitRateMetric(k=10, threshold=0.0),
            MRRMetric(threshold=0.0),
            NDCGMetric(k=10, threshold=0.0),
        ]

        for metric in metrics:
            metric.measure(test_case)
            print(f"  {tc['id']} {metric.__name__}: {metric.score:.3f}")

        # At least assert the framework ran without error
        assert_test(test_case, metrics)


# =============================================================================
# Interference Detection Tests
# =============================================================================


class TestDeepEvalInterference:
    """Interference detection via DeepEval custom metric."""

    @pytest.mark.parametrize("case_idx", range(len(_load_interference_cases())))
    def test_interference_detection(self, case_idx):
        """Run InterferenceDetectionMetric on each test case."""
        cases = _load_interference_cases()
        ic = cases[case_idx]

        test_case = LLMTestCase(
            input=ic["new_content"],
            actual_output=ic["existing_content"],
            additional_metadata={
                "new_content": ic["new_content"],
                "existing_content": ic["existing_content"],
                "similarity": ic["similarity"],
                "expected_signals": ic["expected_signals"],
                "case_id": ic["id"],
                "category": ic["category"],
            },
        )

        metric = InterferenceDetectionMetric(threshold=0.0)
        metric.measure(test_case)

        category = ic["category"]
        print(f"\n  {ic['id']} ({category}): score={metric.score:.2f} — {metric.reason}")

        if category == "false_positive":
            # Document known false positives — don't fail, just report
            if metric.score < 1.0:
                print("    ^ Known false positive (issue #67): got signals when none expected")
        else:
            assert_test(test_case, [metric])
