import time

import pytest

from mcp_memory_service.utils.date_parsing import parse_date_filter


def test_parse_relative_days():
    """Test parsing relative days format."""
    now = time.time()
    result = parse_date_filter("7d")

    # Should be approximately 7 days ago
    expected = now - (7 * 24 * 60 * 60)
    assert abs(result - expected) < 2  # Within 2 seconds


def test_parse_relative_months():
    """Test parsing relative months format."""
    now = time.time()
    result = parse_date_filter("1m")

    # Should be approximately 30 days ago
    expected = now - (30 * 24 * 60 * 60)
    assert abs(result - expected) < 2


def test_parse_relative_years():
    """Test parsing relative years format."""
    now = time.time()
    result = parse_date_filter("1y")

    # Should be approximately 365 days ago
    expected = now - (365 * 24 * 60 * 60)
    assert abs(result - expected) < 2


def test_parse_iso8601_format():
    """Test parsing ISO8601 absolute dates."""
    result = parse_date_filter("2026-01-15T10:30:00Z")

    # Should parse to specific timestamp
    from datetime import datetime, timezone

    expected = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
    assert result == expected


def test_parse_invalid_format_raises_error():
    """Test that invalid formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_date_filter("invalid")
