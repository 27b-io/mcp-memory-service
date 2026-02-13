"""
Tests for temporal decay in memory relevance scoring (issue #73).

Formula: final_score = similarity * (base + exp(-lambda * days) * (1 - base))
- base=0.7 ensures old memories retain 70% minimum relevance
- lambda=0.01 gives ~69-day half-life
- Default OFF (lambda=0.0)
"""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mcp_memory_service.utils.hybrid_search import (
    apply_recency_decay,
    temporal_decay_factor,
)

# ── Pure function: temporal_decay_factor ─────────────────────────────


class TestTemporalDecayFactor:
    """Tests for the temporal_decay_factor pure function."""

    def test_fresh_memory_returns_one(self):
        """0 days old → factor = 1.0 regardless of base."""
        assert temporal_decay_factor(0, lambda_=0.01, base=0.7) == pytest.approx(1.0)

    def test_fresh_memory_base_zero(self):
        """0 days, base=0 → factor = 1.0."""
        assert temporal_decay_factor(0, lambda_=0.01, base=0.0) == pytest.approx(1.0)

    def test_70_day_halflife_no_base(self):
        """lambda=0.01, base=0, 70 days → pure exp decay ≈ 0.497."""
        factor = temporal_decay_factor(70, lambda_=0.01, base=0.0)
        expected = math.exp(-0.01 * 70)  # ≈ 0.4966
        assert factor == pytest.approx(expected, rel=0.01)

    def test_70_days_with_base_floor(self):
        """lambda=0.01, base=0.7, 70 days → 0.7 + exp(-0.7) * 0.3 ≈ 0.849."""
        factor = temporal_decay_factor(70, lambda_=0.01, base=0.7)
        decay = math.exp(-0.01 * 70)  # ≈ 0.4966
        expected = 0.7 + decay * 0.3  # ≈ 0.849
        assert factor == pytest.approx(expected, rel=0.01)

    def test_365_days_with_base_floor(self):
        """1 year old, base=0.7 → approaches but stays above base."""
        factor = temporal_decay_factor(365, lambda_=0.01, base=0.7)
        decay = math.exp(-0.01 * 365)  # ≈ 0.026
        expected = 0.7 + decay * 0.3  # ≈ 0.708
        assert factor == pytest.approx(expected, rel=0.01)
        assert factor > 0.7  # Must stay above base

    def test_very_old_approaches_base(self):
        """1000 days old → factor ≈ base."""
        factor = temporal_decay_factor(1000, lambda_=0.01, base=0.7)
        assert factor == pytest.approx(0.7, abs=0.01)

    def test_lambda_zero_returns_one(self):
        """lambda=0 → disabled, factor always 1.0."""
        assert temporal_decay_factor(365, lambda_=0.0, base=0.7) == pytest.approx(1.0)

    def test_base_zero_reduces_to_pure_exp(self):
        """base=0 → reduces to pure exponential decay (backward compat)."""
        for days in [0, 30, 70, 180, 365]:
            factor = temporal_decay_factor(days, lambda_=0.01, base=0.0)
            expected = math.exp(-0.01 * days)
            assert factor == pytest.approx(expected, rel=0.001), f"Failed at {days} days"

    def test_base_one_always_one(self):
        """base=1.0 → factor always 1.0 (no decay at all)."""
        assert temporal_decay_factor(365, lambda_=0.01, base=1.0) == pytest.approx(1.0)

    def test_negative_days_treated_as_zero(self):
        """Negative days (future timestamp) should not boost above 1.0."""
        factor = temporal_decay_factor(-5, lambda_=0.01, base=0.7)
        assert factor <= 1.0


# ── apply_recency_decay with base parameter ──────────────────────────


class TestApplyRecencyDecayWithBase:
    """Tests for apply_recency_decay with the base floor parameter."""

    def _make_result(self, content_hash: str, score: float, days_ago: int = 0):
        """Create a (memory, score, debug_info) tuple."""
        memory = MagicMock()
        memory.content_hash = content_hash
        updated_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
        memory.updated_at_iso = updated_at.isoformat().replace("+00:00", "Z")
        return (memory, score, {"vector_score": score})

    def test_backward_compat_no_base(self):
        """Without base parameter, behavior unchanged (base=0)."""
        results = [self._make_result("h1", 1.0, days_ago=70)]
        decayed = apply_recency_decay(results, decay_rate=0.01)
        expected = math.exp(-0.01 * 70)
        assert decayed[0][1] == pytest.approx(expected, rel=0.01)

    def test_base_floor_prevents_total_decay(self):
        """base=0.7 prevents score from dropping below 70%."""
        results = [self._make_result("old", 1.0, days_ago=365)]
        decayed = apply_recency_decay(results, decay_rate=0.01, base=0.7)
        assert decayed[0][1] >= 0.7

    def test_base_floor_formula(self):
        """Verify exact formula: score * (base + exp(-lambda*days) * (1-base))."""
        results = [self._make_result("h1", 0.85, days_ago=70)]
        decayed = apply_recency_decay(results, decay_rate=0.01, base=0.7)

        decay = math.exp(-0.01 * 70)
        expected_factor = 0.7 + decay * 0.3
        expected_score = 0.85 * expected_factor
        assert decayed[0][1] == pytest.approx(expected_score, rel=0.01)

    def test_reranking_with_base(self):
        """Recency reranks results even with base floor."""
        results = [
            self._make_result("old", 0.90, days_ago=200),
            self._make_result("new", 0.85, days_ago=1),
        ]
        decayed = apply_recency_decay(results, decay_rate=0.01, base=0.7)

        # New memory (0.85 * ~1.0 = 0.85) should beat old memory (0.90 * ~0.718 = 0.646)
        assert decayed[0][0].content_hash == "new"

    def test_debug_info_includes_base(self):
        """Debug info should show temporal_decay_base when base > 0."""
        results = [self._make_result("h1", 1.0, days_ago=30)]
        decayed = apply_recency_decay(results, decay_rate=0.01, base=0.7)

        debug = decayed[0][2]
        assert "recency_factor" in debug
        assert "temporal_decay_base" in debug
        assert debug["temporal_decay_base"] == 0.7


# ── Config tests ─────────────────────────────────────────────────────


class TestTemporalDecayConfig:
    """Tests for temporal decay configuration fields."""

    def test_lambda_default_off(self):
        """temporal_decay_lambda defaults to 0.0 (OFF)."""
        from mcp_memory_service.config import HybridSearchSettings

        s = HybridSearchSettings()
        assert s.temporal_decay_lambda == 0.0

    def test_base_default(self):
        """temporal_decay_base defaults to 0.7."""
        from mcp_memory_service.config import HybridSearchSettings

        s = HybridSearchSettings()
        assert s.temporal_decay_base == pytest.approx(0.7)

    def test_lambda_configurable_via_env(self, monkeypatch):
        """MCP_MEMORY_TEMPORAL_DECAY_LAMBDA env var works."""
        from mcp_memory_service.config import HybridSearchSettings

        monkeypatch.setenv("MCP_MEMORY_TEMPORAL_DECAY_LAMBDA", "0.02")
        s = HybridSearchSettings()
        assert s.temporal_decay_lambda == pytest.approx(0.02)

    def test_base_configurable_via_env(self, monkeypatch):
        """MCP_MEMORY_TEMPORAL_DECAY_BASE env var works."""
        from mcp_memory_service.config import HybridSearchSettings

        monkeypatch.setenv("MCP_MEMORY_TEMPORAL_DECAY_BASE", "0.5")
        s = HybridSearchSettings()
        assert s.temporal_decay_base == pytest.approx(0.5)

    def test_lambda_rejects_negative(self):
        """Negative lambda should be rejected."""
        from mcp_memory_service.config import HybridSearchSettings

        with pytest.raises(ValidationError):
            HybridSearchSettings(temporal_decay_lambda=-0.01)

    def test_base_clamped_0_to_1(self):
        """Base must be between 0.0 and 1.0."""
        from mcp_memory_service.config import HybridSearchSettings

        with pytest.raises(ValidationError):
            HybridSearchSettings(temporal_decay_base=1.5)
        with pytest.raises(ValidationError):
            HybridSearchSettings(temporal_decay_base=-0.1)
