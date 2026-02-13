"""Date parsing utilities for faceted search."""

import re
import time
from datetime import datetime


def parse_date_filter(date_str: str) -> float:
    """
    Parse relative or absolute date to Unix timestamp.

    Supported formats:
    - Relative: "7d" (days), "1m" (months ~30d), "1y" (years ~365d)
    - Absolute: ISO8601 "2026-01-15T10:30:00Z"

    Args:
        date_str: Date string to parse

    Returns:
        Unix timestamp (float)

    Raises:
        ValueError: If format is invalid
    """
    # Try relative format first
    relative_match = re.match(r"^(\d+)([dmy])$", date_str.lower())
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)

        now = time.time()
        if unit == "d":
            return now - (amount * 24 * 60 * 60)
        elif unit == "m":
            return now - (amount * 30 * 24 * 60 * 60)  # Approximate month
        elif unit == "y":
            return now - (amount * 365 * 24 * 60 * 60)  # Approximate year

    # Try ISO8601 format
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        pass

    raise ValueError(
        f"Invalid date format: {date_str}. Use relative (e.g., '7d', '1m', '1y') or ISO8601 (e.g., '2026-01-15T10:30:00Z')"
    )
