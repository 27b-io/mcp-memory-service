"""Integration tests for faceted_search MCP tool."""

import pytest
from mcp_memory_service.mcp_server import faceted_search
from mcp_memory_service.server import Context


@pytest.mark.asyncio
async def test_faceted_search_returns_toon_format(temp_db_populated):
    """Test that faceted_search returns TOON format output."""
    ctx = Context(db=temp_db_populated)

    result = await faceted_search(
        tags=["python"],
        tag_match_all=False,
        memory_type=None,
        date_from=None,
        date_to=None,
        page=1,
        page_size=10,
        ctx=ctx,
    )

    # Should be string (TOON format)
    assert isinstance(result, str)

    # Should have pagination header
    assert result.startswith("# page=")

    # Should contain memory fields if results exist
    if "total=0" not in result.split("\n")[0]:
        lines = result.split("\n")
        assert len(lines) > 1  # Header + at least one memory


@pytest.mark.asyncio
async def test_faceted_search_parameter_passing(temp_db_populated):
    """Test that faceted_search correctly passes parameters to service layer."""
    ctx = Context(db=temp_db_populated)

    # Test with multiple filters
    result = await faceted_search(
        tags=["python", "test"],
        tag_match_all=True,
        memory_type="note",
        date_from="7d",
        date_to=None,
        page=1,
        page_size=5,
        ctx=ctx,
    )

    # Should return TOON format
    assert isinstance(result, str)
    assert result.startswith("# page=")

    # Check pagination metadata
    header = result.split("\n")[0]
    assert "page=1" in header
    assert "page_size=5" in header
