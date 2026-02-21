# tests/unit/test_config_intent.py
"""Tests for QueryIntentSettings configuration."""


def test_default_settings():
    """QueryIntentSettings loads with sensible defaults."""
    from mcp_memory_service.config import QueryIntentSettings

    s = QueryIntentSettings()
    assert s.enabled is True
    assert s.spacy_model == "en_core_web_sm"
    assert s.max_sub_queries == 4
    assert s.min_query_tokens == 3
    assert s.graph_inject is True
    assert s.graph_inject_limit == 10
    assert s.graph_inject_min_activation == 0.05
    assert s.llm_rerank is False
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-haiku-4-5-20251001"
    assert s.llm_timeout_ms == 2000


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("MCP_INTENT_ENABLED", "false")
    monkeypatch.setenv("MCP_INTENT_MAX_SUB_QUERIES", "6")
    monkeypatch.setenv("MCP_INTENT_LLM_RERANK", "true")
    from mcp_memory_service.config import QueryIntentSettings

    s = QueryIntentSettings()
    assert s.enabled is False
    assert s.max_sub_queries == 6
    assert s.llm_rerank is True


def test_settings_on_main_settings():
    """QueryIntentSettings accessible as settings.intent."""
    from mcp_memory_service.config import Settings

    s = Settings()
    assert hasattr(s, "intent")
    assert s.intent.enabled is True
