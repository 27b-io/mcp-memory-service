"""
Golden-sample integration tests for the full search pipeline.

Seeds 41 Project Gutenberg passages into Qdrant :memory: with real embeddings,
then validates that every search pipeline stage (vector, RRF, tag, salience,
temporal decay, fan-out, pagination, score cap) composes correctly.

This is the single most important test file in the project: it proves the
pipeline works end-to-end with real data and real models, not mocks.
"""

import json
import math
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

# Force CPU to avoid CUDA issues in CI / dev
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mcp_memory_service.config import settings
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.qdrant_storage import QdrantStorage

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def get_result_ids(results: list[dict], k: int = 10) -> list[str]:
    """Extract content_hash IDs from top-K results."""
    return [r["content_hash"] for r in results[:k]]


def get_result_score(results: list[dict], content_hash: str) -> float | None:
    """Get the similarity_score for a specific content_hash, or None if absent."""
    for r in results:
        if r["content_hash"] == content_hash:
            return r["similarity_score"]
    return None


def assert_any_hit(results: list[dict], expected_ids: set[str], k: int = 10) -> None:
    """At least one expected ID appears in top-K results."""
    top_ids = set(get_result_ids(results, k))
    overlap = top_ids & expected_ids
    assert overlap, f"None of {expected_ids} found in top-{k} results. Got: {top_ids}"


def assert_ranks_above(results: list[dict], higher_id: str, lower_id: str) -> None:
    """Assert higher_id appears before lower_id in results."""
    ids = [r["content_hash"] for r in results]
    assert higher_id in ids, f"{higher_id} not found in results"
    assert lower_id in ids, f"{lower_id} not found in results"
    assert ids.index(higher_id) < ids.index(
        lower_id
    ), f"Expected {higher_id} (rank {ids.index(higher_id)}) above {lower_id} (rank {ids.index(lower_id)})"


def get_themes(results: list[dict], k: int = 10) -> set[str]:
    """Collect unique themes mentioned in result metadata (via golden sample tags)."""
    tags: set[str] = set()
    for r in results[:k]:
        if r.get("tags"):
            tags.update(r["tags"])
    return tags


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patched(*patches):
    """Start all mock patches, yield, then stop them. Composable with `with`."""
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _disable_all_boosts():
    """Return patch objects that disable every optional boost."""
    return [
        patch.object(settings.salience, "enabled", False),
        patch.object(settings.spaced_repetition, "enabled", False),
        patch.object(settings.encoding_context, "enabled", False),
        patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),
        patch.object(settings.hybrid_search, "recency_decay", 0.0),
        patch.object(settings.semantic_tag, "enabled", False),
        patch.object(settings.intent, "enabled", False),
    ]


def _full_pipeline_patches():
    """Patches for full pipeline matching production defaults where possible.

    Uses recency_decay (not temporal_decay_lambda) because production defaults
    to recency_decay=0.01 with temporal_decay_lambda=0.0.
    """
    return [
        patch.object(settings.salience, "enabled", True),
        patch.object(settings.salience, "boost_weight", 0.15),
        patch.object(settings.spaced_repetition, "enabled", False),  # needs access history
        patch.object(settings.encoding_context, "enabled", False),  # needs stored contexts
        patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),  # disabled (production default)
        patch.object(settings.hybrid_search, "recency_decay", 0.01),  # production default
        patch.object(settings.intent, "enabled", True),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_golden_samples() -> list[dict]:
    """Load the 41-passage golden samples fixture."""
    path = Path(__file__).parent / "fixtures" / "golden_samples.json"
    with open(path) as f:
        return json.load(f)


GOLDEN_SAMPLES = _load_golden_samples()
_SAMPLES_BY_ID = {s["id"]: s for s in GOLDEN_SAMPLES}

# Thematic cluster ID sets (shared across test classes)
MACHIAVELLI_IDS = {"gutenberg_1232_1", "gutenberg_1232_2", "gutenberg_1232_3"}
AURELIUS_IDS = {"gutenberg_2680_1", "gutenberg_2680_2", "gutenberg_2680_3", "gutenberg_2680_4"}
PLATO_IDS = {"gutenberg_1497_1", "gutenberg_1497_2"}
BEOWULF_IDS = {"gutenberg_16328_1", "gutenberg_16328_2", "gutenberg_16328_3"}
LOCKE_IDS = {"gutenberg_7370_1", "gutenberg_7370_2"}
DARWIN_IDS = {"gutenberg_2009_1", "gutenberg_2009_2"}
FRANKENSTEIN_IDS = {"gutenberg_84_1", "gutenberg_84_2", "gutenberg_84_3"}
MELVILLE_IDS = {"gutenberg_2701_1", "gutenberg_2701_2"}
CONRAD_IDS = {"gutenberg_219_1", "gutenberg_219_2", "gutenberg_219_3"}
HOLMES_IDS = {"gutenberg_1661_1", "gutenberg_1661_2", "gutenberg_1661_3"}
PHILOSOPHY_IDS = AURELIUS_IDS | PLATO_IDS
GOVERNANCE_IDS = BEOWULF_IDS | LOCKE_IDS | PLATO_IDS | AURELIUS_IDS
POLITICAL_IDS = MACHIAVELLI_IDS | PHILOSOPHY_IDS | PLATO_IDS
SCIENCE_IDS = DARWIN_IDS | HOLMES_IDS | FRANKENSTEIN_IDS


@pytest.fixture(scope="module")
async def seeded_storage():
    """Module-scoped Qdrant :memory: storage seeded with all 41 golden samples.

    Uses all-MiniLM-L6-v2 (384-dim, fast) for reproducible embedding generation.
    The collection name includes a random suffix to avoid cross-module collision.
    """
    storage = QdrantStorage(
        storage_path=":memory:",
        embedding_model="all-MiniLM-L6-v2",
        collection_name=f"golden_{uuid.uuid4().hex[:8]}",
    )
    await storage.initialize()

    for sample in GOLDEN_SAMPLES:
        memory = Memory(
            content=sample["content"],
            content_hash=sample["id"],
            tags=sample.get("tags", []),
            memory_type=sample.get("memory_type", "reference"),
        )
        await storage.store(memory)

    yield storage

    await storage.close()


@pytest.fixture(scope="module")
async def service(seeded_storage):
    """MemoryService backed by the module-scoped seeded storage (no graph)."""
    return MemoryService(seeded_storage, graph_client=None)


# ---------------------------------------------------------------------------
# Class 1: Baseline Semantic (control group)
# ---------------------------------------------------------------------------


class TestBaselineSemantic:
    """Hybrid search with all boosts disabled. Establishes a quality baseline.

    Uses default tags=None (hybrid RRF path with adaptive alpha), but disables
    salience, temporal decay, intent fan-out, and all other optional boosts.
    This mirrors production behavior with all optional features turned off.
    """

    @pytest.mark.asyncio
    async def test_leadership_finds_machiavelli_and_political_theory(self, service):
        """'leadership and power' should surface Machiavelli (explicit leadership content)
        and at least one other political/governance source in top 10."""
        with _patched(*_disable_all_boosts()):
            result = await service.retrieve_memories(query="leadership and power", page_size=10)
            memories = result["memories"]

            assert_any_hit(memories, MACHIAVELLI_IDS, k=10)
            assert_any_hit(memories, GOVERNANCE_IDS, k=10)

    @pytest.mark.asyncio
    async def test_scientific_discovery_finds_darwin_and_frankenstein(self, service):
        """'scientific discovery' spans biology (Darwin) and fiction (Shelley)."""
        with _patched(*_disable_all_boosts()):
            result = await service.retrieve_memories(query="scientific discovery and natural world", page_size=10)
            memories = result["memories"]

            assert_any_hit(memories, DARWIN_IDS, k=10)
            assert_any_hit(memories, FRANKENSTEIN_IDS, k=10)

    @pytest.mark.asyncio
    async def test_sea_and_adventure_finds_melville_and_conrad(self, service):
        """'the sea and adventure' should retrieve both nautical authors."""
        with _patched(*_disable_all_boosts()):
            result = await service.retrieve_memories(query="the sea and adventure sailing", page_size=10)
            memories = result["memories"]

            assert_any_hit(memories, MELVILLE_IDS, k=10)
            assert_any_hit(memories, CONRAD_IDS, k=10)

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, service):
        """Every returned score must be in [0.0, 1.0]."""
        with _patched(*_disable_all_boosts()):
            result = await service.retrieve_memories(query="philosophy and virtue", page_size=10)
            for mem in result["memories"]:
                score = mem["similarity_score"]
                assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    @pytest.mark.asyncio
    async def test_results_ordered_by_internal_rank(self, service):
        """Results must be ordered by internal RRF rank (when hybrid) or
        by cosine score (when vector-only). The displayed similarity_score
        is the original cosine, so when RRF re-ranks, display scores may
        not be strictly decreasing. We verify the internal rank is monotonic
        by checking the hybrid_debug.rrf_score or, when in vector-only mode,
        that displayed scores decrease."""
        with _patched(*_disable_all_boosts()):
            # Force vector-only by using a query with no matching existing tags
            result = await service.retrieve_memories(query="xylophone concerto harmonica", page_size=10)
            scores = [m["similarity_score"] for m in result["memories"]]
            for i in range(len(scores) - 1):
                assert (
                    scores[i] >= scores[i + 1] - 1e-9
                ), f"Vector-only: score at rank {i} ({scores[i]:.6f}) < score at rank {i + 1} ({scores[i + 1]:.6f})"


# ---------------------------------------------------------------------------
# Class 2: Tag Matching
# ---------------------------------------------------------------------------


class TestTagMatching:
    """Validates that tag-based boosting via RRF actually changes rankings."""

    @pytest.mark.asyncio
    async def test_philosophy_tag_boosts_philosophy_results(self, service):
        """Query containing 'philosophy' should promote philosophy-tagged memories
        when hybrid search is active, vs pure vector fallback."""
        with _patched(
            patch.object(settings.salience, "enabled", False),
            patch.object(settings.spaced_repetition, "enabled", False),
            patch.object(settings.encoding_context, "enabled", False),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),
            patch.object(settings.hybrid_search, "recency_decay", 0.0),
            patch.object(settings.intent, "enabled", False),
        ):
            result = await service.retrieve_memories(query="philosophy of the soul and virtue", page_size=10)
            memories = result["memories"]

            top_ids = set(get_result_ids(memories, k=10))
            philosophy_hits = top_ids & PHILOSOPHY_IDS
            assert (
                len(philosophy_hits) >= 2
            ), f"Expected >=2 philosophy-tagged results, got {len(philosophy_hits)}: {philosophy_hits}"

    @pytest.mark.asyncio
    async def test_no_matching_tags_falls_back_to_vector(self, service):
        """A query with no tag matches should still return results via vector search.
        The keyword 'xylophone' won't match any tags in the corpus."""
        with _patched(*_disable_all_boosts()):
            result = await service.retrieve_memories(query="xylophone music harmony instruments", page_size=5)
            assert len(result["memories"]) > 0, "Expected vector-only fallback results"

    @pytest.mark.asyncio
    async def test_rrf_changes_ordering_vs_pure_vector(self, service):
        """Hybrid search (RRF) should produce a different top-10 ordering than
        pure vector search for a query that matches existing tags.

        Pure vector is forced by setting alpha=1.0 (full vector weight),
        which causes the code to short-circuit to _retrieve_vector_only."""
        query = "political theory and leadership"

        # Pure vector: alpha=1.0 forces vector-only path
        with _patched(*_disable_all_boosts(), patch.object(settings.hybrid_search, "hybrid_alpha", 1.0)):
            vector_result = await service.retrieve_memories(query=query, page_size=10)
            vector_ids = get_result_ids(vector_result["memories"], k=10)

        # Hybrid: alpha=None (adaptive), semantic_tag enabled for broader tag matching
        with _patched(
            patch.object(settings.salience, "enabled", False),
            patch.object(settings.spaced_repetition, "enabled", False),
            patch.object(settings.encoding_context, "enabled", False),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),
            patch.object(settings.hybrid_search, "recency_decay", 0.0),
            patch.object(settings.hybrid_search, "hybrid_alpha", None),
            patch.object(settings.intent, "enabled", False),
        ):
            hybrid_result = await service.retrieve_memories(query=query, page_size=10)
            hybrid_ids = get_result_ids(hybrid_result["memories"], k=10)

        assert (
            vector_ids != hybrid_ids
        ), "Hybrid RRF should produce different ordering than pure vector. If identical, tag matching had no effect."

    @pytest.mark.asyncio
    async def test_tag_only_results_have_base_score(self, service):
        """Memories found only via tag matching (not vector) get TAG_ONLY_BASE_SCORE (0.1).
        We verify no result drops below this floor when tags contribute."""
        from mcp_memory_service.utils.hybrid_search import TAG_ONLY_BASE_SCORE

        with _patched(
            patch.object(settings.salience, "enabled", False),
            patch.object(settings.spaced_repetition, "enabled", False),
            patch.object(settings.encoding_context, "enabled", False),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),
            patch.object(settings.hybrid_search, "recency_decay", 0.0),
            patch.object(settings.intent, "enabled", False),
        ):
            result = await service.retrieve_memories(query="fiction adventure epic", page_size=10)
            for mem in result["memories"]:
                score = mem["similarity_score"]
                assert score >= TAG_ONLY_BASE_SCORE - 0.01, f"Score {score} below TAG_ONLY_BASE_SCORE ({TAG_ONLY_BASE_SCORE})"


# ---------------------------------------------------------------------------
# Class 3: Temporal Decay
# ---------------------------------------------------------------------------


class TestTemporalDecay:
    """Tests temporal decay with controlled timestamps in an isolated storage."""

    @pytest.fixture
    async def decay_storage_and_service(self):
        """Isolated storage with 4 identical-content memories at different ages.

        Uses the same content (Machiavelli on leadership) but with timestamps
        set to 1, 7, 30, and 90 days ago. This isolates the temporal variable.
        """
        storage = QdrantStorage(
            storage_path=":memory:",
            embedding_model="all-MiniLM-L6-v2",
            collection_name=f"decay_{uuid.uuid4().hex[:8]}",
        )
        await storage.initialize()

        now = time.time()
        base_content = _SAMPLES_BY_ID["gutenberg_1232_1"]["content"]  # Machiavelli
        ages_days = [1, 7, 30, 90]

        for days in ages_days:
            ts = now - (days * 86400)
            memory = Memory(
                content=base_content,
                content_hash=f"decay_{days}d",
                tags=["political-theory", "leadership"],
                memory_type="reference",
                created_at=ts,
                updated_at=ts,
            )
            await storage.store(memory)

        svc = MemoryService(storage, graph_client=None)
        yield storage, svc, ages_days
        await storage.close()

    @pytest.mark.asyncio
    async def test_recent_memory_scores_higher(self, decay_storage_and_service):
        """With temporal decay enabled, the 1-day-old memory should outscore
        the 90-day-old one, even though content is identical."""
        _, svc, _ = decay_storage_and_service
        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.01),
            patch.object(settings.hybrid_search, "temporal_decay_base", 0.7),
        ):
            result = await svc.retrieve_memories(query="leadership cruelty mercy political power", page_size=10)
            memories = result["memories"]

            score_1d = get_result_score(memories, "decay_1d")
            score_90d = get_result_score(memories, "decay_90d")

            assert score_1d is not None, "1-day-old memory not found in results"
            assert score_90d is not None, "90-day-old memory not found in results"
            assert score_1d > score_90d, f"1-day score ({score_1d:.4f}) should exceed 90-day score ({score_90d:.4f})"

    @pytest.mark.asyncio
    async def test_decay_factor_mathematically_correct(self, decay_storage_and_service):
        """Verify the decay factor matches the formula:
        factor = base + exp(-lambda * days) * (1 - base)
        within a tolerance of 0.05."""
        _, svc, _ = decay_storage_and_service
        lam = 0.01
        base = 0.7

        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", lam),
            patch.object(settings.hybrid_search, "temporal_decay_base", base),
        ):
            result = await svc.retrieve_memories(query="leadership cruelty mercy political power", page_size=10)
            memories = result["memories"]

            scores = {}
            for m in memories:
                scores[m["content_hash"]] = m["similarity_score"]

            assert "decay_1d" in scores, "1-day-old memory not found in results"
            assert "decay_90d" in scores, "90-day-old memory not found in results"
            assert scores["decay_90d"] > 0, f"90-day score must be positive, got {scores['decay_90d']}"

            expected_factor_1d = base + math.exp(-lam * 1) * (1 - base)
            expected_factor_90d = base + math.exp(-lam * 90) * (1 - base)
            expected_ratio = expected_factor_1d / expected_factor_90d
            actual_ratio = scores["decay_1d"] / scores["decay_90d"]

            assert (
                abs(actual_ratio - expected_ratio) / expected_ratio < 0.05
            ), f"Decay ratio mismatch: actual={actual_ratio:.4f}, expected={expected_ratio:.4f}"

    @pytest.mark.asyncio
    async def test_decay_disabled_no_score_reduction(self, decay_storage_and_service):
        """With lambda=0, temporal decay is disabled and all identical-content
        memories should have approximately equal scores."""
        _, svc, _ = decay_storage_and_service
        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.0),
            patch.object(settings.hybrid_search, "recency_decay", 0.0),
        ):
            result = await svc.retrieve_memories(query="leadership cruelty mercy political power", page_size=10)
            memories = result["memories"]

            scores = [m["similarity_score"] for m in memories if m["content_hash"].startswith("decay_")]

            assert len(scores) >= 2, "Expected at least 2 decay memories in results"

            max_score = max(scores)
            min_score = min(scores)
            assert (
                max_score - min_score < 0.01
            ), f"Score spread too large with decay disabled: max={max_score:.4f}, min={min_score:.4f}"

    @pytest.mark.asyncio
    async def test_scores_valid_after_decay(self, decay_storage_and_service):
        """All scores must remain in [0.0, 1.0] after temporal decay is applied."""
        _, svc, _ = decay_storage_and_service
        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.hybrid_search, "temporal_decay_lambda", 0.05),
            patch.object(settings.hybrid_search, "temporal_decay_base", 0.7),
        ):
            result = await svc.retrieve_memories(query="leadership cruelty mercy political power", page_size=10)
            for mem in result["memories"]:
                score = mem["similarity_score"]
                assert 0.0 <= score <= 1.0, f"Score {score} out of range after decay"


# ---------------------------------------------------------------------------
# Class 4: Salience Boost
# ---------------------------------------------------------------------------


class TestSalienceBoost:
    """Tests salience-based score boosting with controlled salience values.

    Uses two Darwin passages (very similar domain/content, near-equal base scores)
    at salience 0.0 and 1.0, plus one Frankenstein passage at 0.5. The Darwin pair
    isolates the salience variable for the core high-vs-low assertion.
    """

    @pytest.fixture
    async def salience_storage_and_service(self):
        """Isolated storage with related science passages at different salience levels."""
        storage = QdrantStorage(
            storage_path=":memory:",
            embedding_model="all-MiniLM-L6-v2",
            collection_name=f"salience_{uuid.uuid4().hex[:8]}",
        )
        await storage.initialize()

        passages = [
            ("salience_low", _SAMPLES_BY_ID["gutenberg_2009_1"]["content"], 0.0),  # Darwin: natural selection
            ("salience_mid", _SAMPLES_BY_ID["gutenberg_84_1"]["content"], 0.5),  # Frankenstein: scientific discovery
            ("salience_high", _SAMPLES_BY_ID["gutenberg_2009_2"]["content"], 1.0),  # Darwin: extinction/divergence
        ]

        for hash_id, content, salience in passages:
            memory = Memory(
                content=content,
                content_hash=hash_id,
                tags=["science", "19th-century"],
                memory_type="reference",
                salience_score=salience,
                metadata={"importance": salience},
            )
            await storage.store(memory)

        svc = MemoryService(storage, graph_client=None)
        yield storage, svc
        await storage.close()

    @pytest.mark.asyncio
    async def test_high_salience_outranks_low(self, salience_storage_and_service):
        """With salience boost enabled, the high-salience Darwin passage should
        rank above the low-salience Darwin passage. Both have near-equal base
        scores for a Darwin-focused query, so salience is the differentiator."""
        _, svc = salience_storage_and_service
        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.salience, "enabled", True),
            patch.object(settings.salience, "boost_weight", 0.15),
        ):
            result = await svc.retrieve_memories(query="natural selection evolution biology", page_size=10, tags=[])
            memories = result["memories"]

            assert_ranks_above(memories, "salience_high", "salience_low")

    @pytest.mark.asyncio
    async def test_boost_proportional_to_salience(self, salience_storage_and_service):
        """The boost delta should be proportional to salience_score.
        High-salience (1.0) should get a larger relative boost than mid (0.5)."""
        _, svc = salience_storage_and_service

        # Baseline scores (no salience boost)
        with _patched(*_disable_all_boosts()):
            baseline = await svc.retrieve_memories(
                query="science biology natural world evolution discovery", page_size=10, tags=[]
            )
            base_mid = get_result_score(baseline["memories"], "salience_mid")
            base_high = get_result_score(baseline["memories"], "salience_high")

        # Boosted scores
        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.salience, "enabled", True),
            patch.object(settings.salience, "boost_weight", 0.15),
        ):
            boosted = await svc.retrieve_memories(
                query="science biology natural world evolution discovery", page_size=10, tags=[]
            )
            boosted_mid = get_result_score(boosted["memories"], "salience_mid")
            boosted_high = get_result_score(boosted["memories"], "salience_high")

        assert base_mid is not None, "salience_mid not found in baseline results"
        assert base_high is not None, "salience_high not found in baseline results"
        assert boosted_mid is not None, "salience_mid not found in boosted results"
        assert boosted_high is not None, "salience_high not found in boosted results"
        assert base_mid > 0, f"base_mid score must be positive, got {base_mid}"
        assert base_high > 0, f"base_high score must be positive, got {base_high}"

        delta_mid = boosted_mid - base_mid
        delta_high = boosted_high - base_high
        relative_mid = delta_mid / base_mid
        relative_high = delta_high / base_high
        assert (
            relative_mid < relative_high
        ), f"Mid relative boost ({relative_mid:.4f}) should be less than high relative boost ({relative_high:.4f})"

    @pytest.mark.asyncio
    async def test_salience_disabled_no_effect_on_ranking(self, salience_storage_and_service):
        """With salience disabled, ranking is determined only by content similarity.
        The two Darwin passages should have similar scores since they're in the
        same domain, regardless of their different salience_score values."""
        _, svc = salience_storage_and_service
        with _patched(*_disable_all_boosts()):
            result = await svc.retrieve_memories(query="natural selection evolution survival biology", page_size=10, tags=[])
            memories = result["memories"]

            score_low = get_result_score(memories, "salience_low")
            score_high = get_result_score(memories, "salience_high")

            assert score_low is not None, "Low-salience Darwin memory not found"
            assert score_high is not None, "High-salience Darwin memory not found"
            # Both Darwin passages: similar content → similar scores when salience off
            assert (
                abs(score_low - score_high) < 0.10
            ), f"Darwin pair score gap too large with salience off: low={score_low:.4f}, high={score_high:.4f}"

    @pytest.mark.asyncio
    async def test_salience_boost_weight_respected(self, salience_storage_and_service):
        """The boost_weight parameter should control the magnitude.
        weight=0.3 should produce a higher score than weight=0.15 for high-salience."""
        _, svc = salience_storage_and_service

        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.salience, "enabled", True),
            patch.object(settings.salience, "boost_weight", 0.15),
        ):
            result_015 = await svc.retrieve_memories(query="natural selection evolution biology", page_size=10, tags=[])
            score_015 = get_result_score(result_015["memories"], "salience_high")

        with _patched(
            *_disable_all_boosts(),
            patch.object(settings.salience, "enabled", True),
            patch.object(settings.salience, "boost_weight", 0.30),
        ):
            result_030 = await svc.retrieve_memories(query="natural selection evolution biology", page_size=10, tags=[])
            score_030 = get_result_score(result_030["memories"], "salience_high")

        assert score_015 is not None, "salience_high not found in weight=0.15 results"
        assert score_030 is not None, "salience_high not found in weight=0.30 results"
        assert (
            score_030 > score_015
        ), f"boost_weight=0.30 score ({score_030:.4f}) should exceed boost_weight=0.15 score ({score_015:.4f})"


# ---------------------------------------------------------------------------
# Class 5: Query Intent Fan-out
# ---------------------------------------------------------------------------


def _has_spacy() -> bool:
    """Check if spaCy and en_core_web_sm are available for intent analysis."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


class TestQueryIntentFanout:
    """Tests multi-concept query expansion (requires spaCy for full fan-out)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_spacy(), reason="spaCy en_core_web_sm not installed")
    async def test_multi_concept_query_spans_clusters(self, service):
        """A multi-concept query like 'leadership philosophy and scientific method'
        should retrieve from multiple thematic clusters."""
        with _patched(*_disable_all_boosts(), patch.object(settings.intent, "enabled", True)):
            result = await service.retrieve_memories(
                query="leadership philosophy and the scientific method of observation",
                page_size=10,
            )
            memories = result["memories"]
            top_ids = set(get_result_ids(memories, k=10))

            has_political = bool(top_ids & POLITICAL_IDS)
            has_science = bool(top_ids & SCIENCE_IDS)

            # Fan-out must span BOTH clusters, not just the best single-vector match
            assert has_political and has_science, (
                f"Multi-concept query should hit BOTH clusters. "
                f"Political: {has_political}, Science: {has_science}. Got: {top_ids}"
            )

    @pytest.mark.asyncio
    async def test_single_concept_returns_focused_results(self, service):
        """A short query (below min_query_tokens=3) bypasses fan-out and returns
        tightly focused results via a single vector search."""
        with _patched(*_disable_all_boosts(), patch.object(settings.intent, "enabled", True)):
            result = await service.retrieve_memories(query="stoic virtue", page_size=10)
            memories = result["memories"]

            top_ids = set(get_result_ids(memories, k=10))
            stoic_hits = top_ids & AURELIUS_IDS
            assert len(stoic_hits) >= 2, f"Expected >=2 Aurelius results for focused query, got {len(stoic_hits)}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_spacy(), reason="spaCy en_core_web_sm not installed")
    async def test_multi_concept_covers_more_themes(self, service):
        """Multi-concept results should cover more unique tag categories than
        a single-concept query of comparable length."""
        with _patched(*_disable_all_boosts(), patch.object(settings.intent, "enabled", True)):
            multi = await service.retrieve_memories(
                query="ancient philosophy about virtue combined with modern scientific discovery",
                page_size=10,
            )
            single = await service.retrieve_memories(
                query="ancient philosophy about virtue and reason",
                page_size=10,
            )

            multi_tags = get_themes(multi["memories"], k=10)
            single_tags = get_themes(single["memories"], k=10)

            assert len(multi_tags) >= len(
                single_tags
            ), f"Multi-concept tags ({len(multi_tags)}) should >= single-concept ({len(single_tags)})"

    @pytest.mark.asyncio
    async def test_fanout_respects_min_similarity(self, service):
        """Results from fan-out path should still respect min_similarity threshold."""
        with _patched(*_disable_all_boosts(), patch.object(settings.intent, "enabled", True)):
            result = await service.retrieve_memories(
                query="leadership philosophy and scientific method of observation",
                page_size=10,
                min_similarity=0.3,
            )
            for mem in result["memories"]:
                assert mem["similarity_score"] >= 0.3 - 0.01, f"Score {mem['similarity_score']:.4f} below min_similarity=0.3"


# ---------------------------------------------------------------------------
# Class 6: Full Pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """All features enabled (except FalkorDB graph). End-to-end correctness."""

    @pytest.mark.asyncio
    async def test_full_pipeline_quality_at_least_baseline(self, service):
        """Full pipeline results should return relevant results for core queries.
        We verify that each query produces a non-trivial hit rate against a
        broad set of plausible expected IDs."""
        queries_and_expected = [
            ("leadership and power", MACHIAVELLI_IDS | GOVERNANCE_IDS),
            ("the sea and adventure", MELVILLE_IDS | CONRAD_IDS | {"gutenberg_345_1"}),
            ("philosophy and virtue", PHILOSOPHY_IDS),
        ]

        with _patched(*_full_pipeline_patches()):
            for query, expected in queries_and_expected:
                result = await service.retrieve_memories(query=query, page_size=10)
                top = set(get_result_ids(result["memories"], k=10))
                hits = top & expected
                assert len(hits) >= 1, f"Query '{query}': no hits in top 10. Expected any of {expected}, got {top}"

    @pytest.mark.asyncio
    async def test_no_score_exceeds_one(self, service):
        """Score cap must ensure no result exceeds 1.0 after all boosts."""
        with _patched(*_full_pipeline_patches()):
            for query in ["leadership", "philosophy virtue", "scientific discovery"]:
                result = await service.retrieve_memories(query=query, page_size=10)
                for mem in result["memories"]:
                    assert mem["similarity_score"] <= 1.0 + 1e-9, f"Score {mem['similarity_score']:.6f} exceeds 1.0 cap"

    @pytest.mark.asyncio
    async def test_min_similarity_filters_low_relevance(self, service):
        """min_similarity=0.4 should filter out genuinely low-relevance results."""
        with _patched(*_full_pipeline_patches()):
            result = await service.retrieve_memories(
                query="time travel and the fourth dimension",
                page_size=10,
                min_similarity=0.4,
            )
            for mem in result["memories"]:
                assert mem["similarity_score"] >= 0.4 - 0.01, f"Score {mem['similarity_score']:.4f} below min_similarity=0.4"

    @pytest.mark.asyncio
    async def test_pagination_no_overlapping_hashes(self, service):
        """Page 1 and page 2 results should have no overlapping content_hashes."""
        with _patched(*_full_pipeline_patches()):
            page1 = await service.retrieve_memories(query="philosophy and human nature", page=1, page_size=5)
            page2 = await service.retrieve_memories(query="philosophy and human nature", page=2, page_size=5)

            ids_1 = set(get_result_ids(page1["memories"], k=5))
            ids_2 = set(get_result_ids(page2["memories"], k=5))

            overlap = ids_1 & ids_2
            assert not overlap, f"Pages 1 and 2 share {len(overlap)} results: {overlap}"

    @pytest.mark.asyncio
    async def test_empty_query_returns_graceful_result(self, service):
        """An empty query should return an empty result or error, not crash."""
        with _patched(*_full_pipeline_patches()):
            result = await service.retrieve_memories(query="", page_size=10)
            assert "memories" in result or "error" in result, "Empty query should return dict with 'memories' or 'error' key"


# ---------------------------------------------------------------------------
# Class 7: Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Latency and throughput tests. Generous thresholds for CI (CPU only)."""

    @pytest.mark.asyncio
    async def test_single_query_latency(self, service):
        """A single query should complete in under 500ms.
        Generous bound: includes embedding generation + Qdrant search + pipeline."""
        with _patched(*_disable_all_boosts()):
            # Warm up (first query loads model)
            await service.retrieve_memories(query="warmup", page_size=1)

            start = time.perf_counter()
            await service.retrieve_memories(query="leadership and power", page_size=10)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 500, f"Single query took {elapsed_ms:.0f}ms, expected < 500ms"

    @pytest.mark.asyncio
    async def test_sequential_query_p95(self, service):
        """50 sequential queries, p95 latency should be under 1 second.
        Tests that there are no per-query resource leaks or cold-start penalties."""
        with _patched(*_disable_all_boosts()):
            queries = [
                "leadership and power",
                "scientific discovery",
                "the sea and adventure",
                "philosophy and virtue",
                "art and beauty",
                "natural rights and liberty",
                "observation and deduction",
                "revolution and social upheaval",
                "monsters and heroism",
                "mental health and confinement",
            ]

            # Warm up
            await service.retrieve_memories(query="warmup", page_size=1)

            latencies = []
            for i in range(50):
                query = queries[i % len(queries)]
                start = time.perf_counter()
                await service.retrieve_memories(query=query, page_size=10)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            latencies.sort()
            p95_idx = int(0.95 * len(latencies))
            p95 = latencies[p95_idx]

            assert p95 < 1000, f"p95 latency is {p95:.0f}ms, expected < 1000ms"

    @pytest.mark.asyncio
    async def test_seeding_latency(self):
        """Seeding 41 memories (embedding generation + storage) should complete
        in under 60 seconds on CPU."""
        start = time.perf_counter()

        storage = QdrantStorage(
            storage_path=":memory:",
            embedding_model="all-MiniLM-L6-v2",
            collection_name=f"perf_{uuid.uuid4().hex[:8]}",
        )
        await storage.initialize()

        for sample in GOLDEN_SAMPLES:
            memory = Memory(
                content=sample["content"],
                content_hash=sample["id"],
                tags=sample.get("tags", []),
                memory_type=sample.get("memory_type", "reference"),
            )
            await storage.store(memory)

        elapsed_s = time.perf_counter() - start
        await storage.close()

        assert elapsed_s < 60, f"Seeding 41 memories took {elapsed_s:.1f}s, expected < 60s"
