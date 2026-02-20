"""
Unit tests for graph schema definitions.

Validates the schema statements are well-formed Cypher and relation type whitelist.
"""

from mcp_memory_service.graph.schema import RELATION_TYPES, SCHEMA_STATEMENTS


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


class TestRelationTypes:
    """Test the typed relationship whitelist."""

    def test_relation_types_is_frozenset(self):
        """Must be immutable to prevent runtime modification."""
        assert isinstance(RELATION_TYPES, frozenset)

    def test_relation_types_contains_required_types(self):
        assert "RELATES_TO" in RELATION_TYPES
        assert "PRECEDES" in RELATION_TYPES
        assert "CONTRADICTS" in RELATION_TYPES

    def test_relation_types_count(self):
        """Exactly 4 types — SUPERSEDES added for contradiction resolution."""
        assert len(RELATION_TYPES) == 4

    def test_relation_types_are_uppercase(self):
        """All types must be uppercase for Cypher compatibility."""
        for t in RELATION_TYPES:
            assert t == t.upper(), f"{t} should be uppercase"
