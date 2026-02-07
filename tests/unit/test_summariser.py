"""Unit tests for extractive and LLM summarisers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.utils.summariser import extract_summary, llm_summarise, summarise


class TestExtractSummary:
    """Test extract_summary with various content patterns."""

    def test_client_provided_summary_takes_precedence(self):
        content = "This is a very long piece of content that goes on and on."
        summary = extract_summary(content, client_summary="Custom summary")
        assert summary == "Custom summary"

    def test_client_summary_stripped(self):
        summary = extract_summary("content", client_summary="  spaces  ")
        assert summary == "spaces"

    def test_client_summary_empty_returns_none(self):
        summary = extract_summary("content", client_summary="   ")
        assert summary is None

    def test_empty_content_returns_none(self):
        assert extract_summary("") is None
        assert extract_summary("   ") is None

    def test_whitespace_only_returns_none(self):
        assert extract_summary("\t\n") is None
        assert extract_summary("  \n  ") is None

    def test_decision_line_extracted(self):
        content = "Some preamble.\nDecision: Use Qdrant as the sole backend.\nMore details."
        summary = extract_summary(content)
        assert summary == "Use Qdrant as the sole backend."

    def test_conclusion_line_extracted(self):
        content = "Analysis follows.\nConclusion: The approach is viable.\nEnd."
        summary = extract_summary(content)
        assert summary == "The approach is viable."

    def test_key_insight_line_extracted(self):
        content = "Observations.\nKey insight: Caching reduces latency by 50%."
        summary = extract_summary(content)
        assert summary == "Caching reduces latency by 50%."

    def test_summary_line_extracted(self):
        content = "Long content.\nSummary: Brief overview of the topic."
        summary = extract_summary(content)
        assert summary == "Brief overview of the topic."

    def test_tldr_line_extracted(self):
        content = "Detailed analysis.\nTL;DR: It works.\nMore stuff."
        summary = extract_summary(content)
        assert summary == "It works."

    def test_result_line_extracted(self):
        content = "Experiment setup.\nResult: 95% accuracy achieved."
        summary = extract_summary(content)
        assert summary == "95% accuracy achieved."

    def test_signal_line_case_insensitive(self):
        content = "Stuff.\ndecision: lowercase works too."
        summary = extract_summary(content)
        assert summary == "lowercase works too."

    def test_first_sentence_fallback(self):
        content = "This is the first sentence. This is the second."
        summary = extract_summary(content)
        assert summary == "This is the first sentence."

    def test_first_sentence_question_mark(self):
        content = "Is this a question? Yes it is."
        summary = extract_summary(content)
        assert summary == "Is this a question?"

    def test_first_sentence_exclamation(self):
        content = "Watch out! Something happened."
        summary = extract_summary(content)
        assert summary == "Watch out!"

    def test_no_sentence_boundary_returns_full_text(self):
        content = "No punctuation at all"
        summary = extract_summary(content)
        assert summary == "No punctuation at all"

    def test_long_content_truncated(self):
        content = "A " * 200 + "end of content."
        summary = extract_summary(content)
        assert len(summary) <= 203  # 200 + "..."
        assert summary.endswith("...")

    def test_truncation_on_word_boundary(self):
        # Build content with words that won't split mid-word
        content = "word " * 60  # ~300 chars
        summary = extract_summary(content)
        assert not summary.endswith(" ...")  # No trailing space before ...
        assert "..." in summary

    def test_multiline_content_collapses_whitespace(self):
        content = "First line\n\nSecond line. Third line."
        summary = extract_summary(content)
        assert summary == "First line Second line."

    def test_signal_line_priority_over_first_sentence(self):
        """Signal lines should be preferred over first sentence."""
        content = "Short intro. More text.\nDecision: Go with plan B."
        summary = extract_summary(content)
        assert summary == "Go with plan B."


class TestMemorySummaryField:
    """Test that summary field works in Memory dataclass."""

    def test_memory_with_summary(self):
        from mcp_memory_service.models.memory import Memory

        mem = Memory(
            content="Full content here",
            content_hash="abc123",
            summary="Brief summary",
        )
        assert mem.summary == "Brief summary"

    def test_memory_without_summary(self):
        from mcp_memory_service.models.memory import Memory

        mem = Memory(content="Full content", content_hash="abc123")
        assert mem.summary is None

    def test_memory_to_dict_includes_summary(self):
        from mcp_memory_service.models.memory import Memory

        mem = Memory(
            content="Content",
            content_hash="abc123",
            summary="A summary",
        )
        d = mem.to_dict()
        assert d["summary"] == "A summary"

    def test_memory_to_dict_none_summary(self):
        from mcp_memory_service.models.memory import Memory

        mem = Memory(content="Content", content_hash="abc123")
        d = mem.to_dict()
        assert d["summary"] is None

    def test_memory_from_dict_with_summary(self):
        from mcp_memory_service.models.memory import Memory

        data = {
            "content": "Content",
            "content_hash": "abc123",
            "summary": "Restored summary",
        }
        mem = Memory.from_dict(data)
        assert mem.summary == "Restored summary"

    def test_memory_from_dict_without_summary(self):
        from mcp_memory_service.models.memory import Memory

        data = {"content": "Content", "content_hash": "abc123"}
        mem = Memory.from_dict(data)
        assert mem.summary is None


class TestNumberedListRegexFix:
    """Test that numbered lists don't trigger sentence boundary detection (mm-yy6k)."""

    def test_numbered_list_not_truncated(self):
        """Numbered lists like '1. Item' should not be treated as sentence boundaries."""
        content = "Steps: 1. First step 2. Second step 3. Third step. Done."
        summary = extract_summary(content)
        # Should extract the first sentence including all numbered items
        assert "1. First step 2. Second step 3. Third step." in summary
        assert "Done." not in summary  # Real sentence boundary stops here

    def test_numbered_list_in_middle(self):
        """Number before period mid-content should not split."""
        content = "We have 3 options here. Option A is best."
        summary = extract_summary(content)
        assert summary == "We have 3 options here."

    def test_decimal_number_not_sentence_boundary(self):
        """Decimals like '2.5' should not split sentences."""
        content = "The value is 2.5 units. More text follows."
        summary = extract_summary(content)
        assert summary == "The value is 2.5 units."

    def test_regular_sentence_still_works(self):
        """Normal sentence boundaries still work after regex fix."""
        content = "First sentence. Second sentence."
        summary = extract_summary(content)
        assert summary == "First sentence."


class TestLLMSummarise:
    """Test LLM-powered summarisation with mocked API responses."""

    @pytest.mark.asyncio
    async def test_llm_summarise_success(self):
        """Test successful LLM summarisation with mocked Gemini API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "This is an LLM-generated summary."}]}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            summary = await llm_summarise(
                content="Long content to summarise",
                api_key="fake-api-key",
                model="gemini-2.0-flash-exp",
                max_tokens=50,
                timeout=5.0,
            )

            assert summary == "This is an LLM-generated summary."

    @pytest.mark.asyncio
    async def test_llm_summarise_empty_content(self):
        """Empty content returns None without API call."""
        summary = await llm_summarise(
            content="",
            api_key="fake-api-key",
        )
        assert summary is None

    @pytest.mark.asyncio
    async def test_llm_summarise_no_api_key(self):
        """No API key returns None without API call."""
        summary = await llm_summarise(
            content="Content",
            api_key="",
        )
        assert summary is None

    @pytest.mark.asyncio
    async def test_llm_summarise_timeout(self):
        """Timeout returns None for fallback to extractive."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=__import__("httpx").TimeoutException("Timeout")
            )

            summary = await llm_summarise(
                content="Content",
                api_key="fake-api-key",
                timeout=1.0,
            )

            assert summary is None

    @pytest.mark.asyncio
    async def test_llm_summarise_http_error(self):
        """HTTP errors return None for fallback to extractive."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=__import__("httpx").HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
            )

            summary = await llm_summarise(
                content="Content",
                api_key="fake-api-key",
            )

            assert summary is None

    @pytest.mark.asyncio
    async def test_llm_summarise_malformed_response(self):
        """Malformed API response returns None for fallback."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"candidates": []}  # Empty candidates
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            summary = await llm_summarise(
                content="Content",
                api_key="fake-api-key",
            )

            assert summary is None


class TestSummariseWrapper:
    """Test the main summarise() wrapper function with config-based mode detection."""

    def test_summarise_client_summary_precedence(self):
        """Client-provided summary takes precedence over all modes."""
        summary = summarise(
            content="Long content",
            client_summary="Custom summary",
            config=None,
        )
        assert summary == "Custom summary"

    def test_summarise_extractive_mode_explicit(self):
        """Explicit extractive mode uses extract_summary."""
        mock_config = MagicMock()
        mock_config.summary.get_effective_mode.return_value = "extractive"

        content = "First sentence. Second sentence."
        summary = summarise(content, config=mock_config)
        assert summary == "First sentence."

    def test_summarise_no_config_defaults_to_extractive(self):
        """No config defaults to extractive mode."""
        content = "First sentence. Second sentence."
        summary = summarise(content, config=None)
        assert summary == "First sentence."

    def test_summarise_empty_content(self):
        """Empty content returns None regardless of mode."""
        assert summarise("", config=None) is None
        assert summarise("   ", config=None) is None

    @patch("mcp_memory_service.utils.summariser.asyncio.run")
    def test_summarise_llm_mode_success(self, mock_asyncio_run):
        """LLM mode uses llm_summarise when API key is present."""
        mock_config = MagicMock()
        mock_config.summary.get_effective_mode.return_value = "llm"
        mock_config.summary.api_key.get_secret_value.return_value = "fake-key"
        mock_config.summary.model = "gemini-2.0-flash-exp"
        mock_config.summary.max_tokens = 50
        mock_config.summary.timeout_seconds = 5.0

        # Mock successful LLM response
        mock_asyncio_run.return_value = "LLM summary"

        summary = summarise("Content to summarise", config=mock_config)
        assert summary == "LLM summary"
        mock_asyncio_run.assert_called_once()

    @patch("mcp_memory_service.utils.summariser.asyncio.run")
    def test_summarise_llm_fallback_to_extractive(self, mock_asyncio_run):
        """LLM mode falls back to extractive on error."""
        mock_config = MagicMock()
        mock_config.summary.get_effective_mode.return_value = "llm"
        mock_config.summary.api_key.get_secret_value.return_value = "fake-key"

        # Mock LLM failure (returns None)
        mock_asyncio_run.return_value = None

        content = "First sentence. Second sentence."
        summary = summarise(content, config=mock_config)
        # Should fall back to extractive
        assert summary == "First sentence."


class TestConfigAutoDetection:
    """Test SummarySettings auto-detection of mode based on API key presence."""

    def test_auto_detect_llm_when_api_key_present(self):
        """Auto-detect should choose LLM mode when API key is set."""
        from pydantic import SecretStr

        from mcp_memory_service.config import SummarySettings

        settings = SummarySettings(mode=None, api_key=SecretStr("fake-key"))
        assert settings.get_effective_mode() == "llm"

    def test_auto_detect_extractive_when_no_api_key(self):
        """Auto-detect should choose extractive mode when no API key."""
        from mcp_memory_service.config import SummarySettings

        settings = SummarySettings(mode=None, api_key=None)
        assert settings.get_effective_mode() == "extractive"

    def test_explicit_extractive_overrides_api_key(self):
        """Explicit mode='extractive' is honored even with API key."""
        from pydantic import SecretStr

        from mcp_memory_service.config import SummarySettings

        settings = SummarySettings(mode="extractive", api_key=SecretStr("fake-key"))
        assert settings.get_effective_mode() == "extractive"

    def test_explicit_llm_requires_api_key(self):
        """Explicit mode='llm' without API key falls back to extractive."""
        from mcp_memory_service.config import SummarySettings

        settings = SummarySettings(mode="llm", api_key=None)
        assert settings.get_effective_mode() == "extractive"

    def test_invalid_mode_falls_back_to_extractive(self):
        """Invalid mode falls back to extractive."""
        from mcp_memory_service.config import SummarySettings

        settings = SummarySettings(mode="invalid", api_key=None)
        assert settings.get_effective_mode() == "extractive"
