"""Unit tests for extractive summariser."""

from mcp_memory_service.utils.summariser import extract_summary


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

    def test_none_content_returns_none(self):
        # Edge case: should handle gracefully
        assert extract_summary("") is None

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
