"""Tests for SemanticTagSettings configuration."""

from mcp_memory_service.config import SemanticTagSettings


class TestSemanticTagSettings:
    def test_defaults(self):
        """All defaults should be sane out of the box."""
        s = SemanticTagSettings()
        assert s.enabled is True
        assert s.similarity_threshold == 0.5
        assert s.max_tags == 10
        assert s.rrf_weight == 0.5
        assert s.cache_ttl == 3600

    def test_env_override(self, monkeypatch):
        """Env vars with MCP_SEMANTIC_TAG_ prefix should override defaults."""
        monkeypatch.setenv("MCP_SEMANTIC_TAG_ENABLED", "false")
        monkeypatch.setenv("MCP_SEMANTIC_TAG_SIMILARITY_THRESHOLD", "0.7")
        monkeypatch.setenv("MCP_SEMANTIC_TAG_MAX_TAGS", "5")
        s = SemanticTagSettings()
        assert s.enabled is False
        assert s.similarity_threshold == 0.7
        assert s.max_tags == 5

    def test_threshold_bounds(self):
        """Threshold should be clamped to 0.0-1.0."""
        s = SemanticTagSettings(similarity_threshold=0.0)
        assert s.similarity_threshold == 0.0
        s = SemanticTagSettings(similarity_threshold=1.0)
        assert s.similarity_threshold == 1.0
