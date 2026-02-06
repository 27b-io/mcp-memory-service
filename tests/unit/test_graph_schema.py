"""
Unit tests for graph schema definitions.

Validates the schema statements are well-formed Cypher.
"""


from mcp_memory_service.graph.schema import SCHEMA_STATEMENTS


class TestGraphSchema:
    """Test graph schema definitions."""

    def test_schema_statements_not_empty(self):
        assert len(SCHEMA_STATEMENTS) > 0

    def test_all_statements_are_index_creation(self):
        """All schema statements should be idempotent index creation."""
        for stmt in SCHEMA_STATEMENTS:
            assert "CREATE INDEX IF NOT EXISTS" in stmt

    def test_memory_content_hash_index_exists(self):
        """content_hash index is required for O(1) node lookup."""
        found = any("content_hash" in stmt for stmt in SCHEMA_STATEMENTS)
        assert found, "Missing content_hash index"

    def test_memory_created_at_index_exists(self):
        """created_at index is required for temporal queries."""
        found = any("created_at" in stmt for stmt in SCHEMA_STATEMENTS)
        assert found, "Missing created_at index"
