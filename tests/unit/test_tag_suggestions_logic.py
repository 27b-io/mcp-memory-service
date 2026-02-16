"""
Unit tests for tag suggestions endpoint logic.

Tests the tag filtering and matching logic without requiring HTTP server.
"""

import pytest


class TestTagFilteringLogic:
    """Test the tag filtering logic used in the tag suggestions endpoint."""

    def test_case_insensitive_prefix_matching(self):
        """Test case-insensitive prefix matching."""
        all_tags = ["python", "Python-3", "javascript", "java", "typescript", "PHP"]
        query = "py"

        # Simulate the filtering logic from the endpoint
        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "python" in matching_tags
        assert "Python-3" in matching_tags
        assert "javascript" not in matching_tags
        assert "PHP" not in matching_tags

    def test_case_insensitive_prefix_matching_uppercase_query(self):
        """Test that uppercase query works the same as lowercase."""
        all_tags = ["python", "Python-3", "javascript", "java", "typescript", "PHP"]
        query = "PY"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "python" in matching_tags
        assert "Python-3" in matching_tags

    def test_empty_query_returns_all_tags(self):
        """Test that empty query returns all tags."""
        all_tags = ["python", "javascript", "java", "typescript"]
        query = ""

        if query:
            query_lower = query.lower()
            matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]
        else:
            matching_tags = all_tags

        assert len(matching_tags) == len(all_tags)

    def test_no_matches_returns_empty_list(self):
        """Test that no matches returns empty list."""
        all_tags = ["python", "javascript", "java", "typescript"]
        query = "zzz"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert len(matching_tags) == 0

    def test_limit_restricts_results(self):
        """Test that limit parameter restricts results."""
        all_tags = ["python", "python-2", "python-3", "python-django", "python-flask"]
        query = "py"
        limit = 3

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]
        suggestions = matching_tags[:limit]

        assert len(suggestions) <= limit
        assert len(suggestions) == 3

    def test_special_characters_in_tags(self):
        """Test that tags with special characters are handled correctly."""
        all_tags = ["project-management", "data-science", "machine_learning", "web.development"]
        query = "data"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "data-science" in matching_tags
        assert len(matching_tags) == 1

    def test_exact_match_included(self):
        """Test that exact matches are included in results."""
        all_tags = ["python", "python-2", "python-3"]
        query = "python"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "python" in matching_tags
        assert "python-2" in matching_tags
        assert "python-3" in matching_tags

    def test_single_character_query(self):
        """Test that single character queries work correctly."""
        all_tags = ["python", "javascript", "java", "perl", "php"]
        query = "p"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "python" in matching_tags
        assert "perl" in matching_tags
        assert "php" in matching_tags
        assert "javascript" not in matching_tags
        assert "java" not in matching_tags

    def test_numeric_tags(self):
        """Test that numeric tags are handled correctly."""
        all_tags = ["v1.0", "v2.0", "v2.1", "version-3"]
        query = "v2"

        query_lower = query.lower()
        matching_tags = [tag for tag in all_tags if tag.lower().startswith(query_lower)]

        assert "v2.0" in matching_tags
        assert "v2.1" in matching_tags
        assert "v1.0" not in matching_tags
        assert "version-3" not in matching_tags


class TestTagSuggestionsResponse:
    """Test the TagSuggestionsResponse model structure."""

    def test_response_structure(self):
        """Test that response has correct structure."""
        from mcp_memory_service.web.api.search import TagSuggestionsResponse

        response = TagSuggestionsResponse(suggestions=["python", "javascript"], query="py", total=2)

        assert response.suggestions == ["python", "javascript"]
        assert response.query == "py"
        assert response.total == 2

    def test_empty_suggestions(self):
        """Test response with empty suggestions."""
        from mcp_memory_service.web.api.search import TagSuggestionsResponse

        response = TagSuggestionsResponse(suggestions=[], query="zzz", total=0)

        assert response.suggestions == []
        assert response.query == "zzz"
        assert response.total == 0

    def test_response_serialization(self):
        """Test that response can be serialized to JSON."""
        from mcp_memory_service.web.api.search import TagSuggestionsResponse

        response = TagSuggestionsResponse(suggestions=["python", "javascript"], query="py", total=2)

        # Test dict conversion (Pydantic model)
        data = response.model_dump()

        assert "suggestions" in data
        assert "query" in data
        assert "total" in data
        assert isinstance(data["suggestions"], list)
        assert isinstance(data["query"], str)
        assert isinstance(data["total"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
