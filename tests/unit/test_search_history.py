# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for search history tracking functionality."""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest

from mcp_memory_service.models.search_history import SearchHistoryEntry
from mcp_memory_service.storage.search_history_db import SearchHistoryDB


@pytest.fixture
async def temp_db():
    """Create a temporary search history database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_search_history.db"
        db = SearchHistoryDB(str(db_path))
        await db.initialize()
        yield db
        await db.close()


@pytest.mark.asyncio
async def test_search_history_entry_creation():
    """Test SearchHistoryEntry model creation and serialization."""
    entry = SearchHistoryEntry(
        query="test query",
        search_type="retrieve_memory",
        result_count=5,
        response_time_ms=123.4,
        parameters={"page": 1, "page_size": 10},
        client_id="test-client",
    )

    assert entry.query == "test query"
    assert entry.search_type == "retrieve_memory"
    assert entry.result_count == 5
    assert entry.response_time_ms == 123.4
    assert entry.parameters == {"page": 1, "page_size": 10}
    assert entry.client_id == "test-client"
    assert entry.timestamp > 0

    # Test to_dict
    data = entry.to_dict()
    assert data["query"] == "test query"
    assert data["search_type"] == "retrieve_memory"
    assert data["result_count"] == 5

    # Test from_dict
    entry2 = SearchHistoryEntry.from_dict(data)
    assert entry2.query == entry.query
    assert entry2.search_type == entry.search_type
    assert entry2.result_count == entry.result_count


@pytest.mark.asyncio
async def test_database_initialization(temp_db):
    """Test database schema initialization."""
    # Check that database file was created
    assert os.path.exists(temp_db.db_path)

    # Verify tables exist by running a simple query
    import aiosqlite

    async with aiosqlite.connect(temp_db.db_path) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_history'")
        result = await cursor.fetchone()
        assert result is not None
        assert result[0] == "search_history"


@pytest.mark.asyncio
async def test_log_search(temp_db):
    """Test logging search queries."""
    entry = SearchHistoryEntry(
        query="python testing",
        search_type="retrieve_memory",
        result_count=10,
        response_time_ms=50.5,
        parameters={"min_similarity": 0.7},
    )

    record_id = await temp_db.log_search(entry)
    assert record_id > 0


@pytest.mark.asyncio
async def test_get_search_stats(temp_db):
    """Test retrieving search statistics."""
    # Log some test searches
    searches = [
        SearchHistoryEntry("query 1", "retrieve_memory", result_count=5, response_time_ms=100.0),
        SearchHistoryEntry("query 2", "memory_scan", result_count=3, response_time_ms=50.0),
        SearchHistoryEntry("query 1", "retrieve_memory", result_count=7, response_time_ms=120.0),  # Duplicate query
        SearchHistoryEntry("query 3", "find_similar", result_count=8, response_time_ms=80.0),
    ]

    for search in searches:
        await temp_db.log_search(search)

    # Get stats
    stats = await temp_db.get_search_stats(days=30)

    assert stats["total_searches"] == 4
    assert stats["avg_response_time"] > 0
    assert "retrieve_memory" in stats["search_types"]
    assert stats["search_types"]["retrieve_memory"] == 2
    assert stats["search_types"]["memory_scan"] == 1
    assert stats["search_types"]["find_similar"] == 1

    # Check popular queries
    popular_queries = stats["popular_queries"]
    assert len(popular_queries) > 0
    assert popular_queries[0]["query"] == "query 1"
    assert popular_queries[0]["count"] == 2


@pytest.mark.asyncio
async def test_get_recent_searches(temp_db):
    """Test retrieving recent search history."""
    # Log some searches
    for i in range(15):
        entry = SearchHistoryEntry(
            query=f"test query {i}",
            search_type="retrieve_memory",
            result_count=i,
            response_time_ms=float(i * 10),
        )
        await temp_db.log_search(entry)
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

    # Get recent searches
    recent = await temp_db.get_recent_searches(limit=10)

    assert len(recent) == 10
    # Should be in reverse chronological order
    assert recent[0].query == "test query 14"
    assert recent[9].query == "test query 5"


@pytest.mark.asyncio
async def test_get_query_recommendations(temp_db):
    """Test getting query recommendations based on popular searches."""
    # Log searches with varying frequencies
    queries = [
        ("popular query", 5),
        ("another query", 3),
        ("rare query", 1),
    ]

    for query, count in queries:
        for _ in range(count):
            entry = SearchHistoryEntry(query=query, search_type="retrieve_memory", result_count=10, response_time_ms=100.0)
            await temp_db.log_search(entry)
            await asyncio.sleep(0.01)

    # Get recommendations
    recommendations = await temp_db.get_query_recommendations(limit=5)

    assert len(recommendations) > 0
    # Most popular query should be first
    assert recommendations[0]["query"] == "popular query"
    assert recommendations[0]["frequency"] == 5


@pytest.mark.asyncio
async def test_stats_with_time_filter(temp_db):
    """Test that stats only include searches within the specified time range."""
    # Log an old search (simulate by manually setting timestamp)
    import aiosqlite

    old_timestamp = time.time() - (60 * 86400)  # 60 days ago
    async with aiosqlite.connect(temp_db.db_path) as db:
        await db.execute(
            "INSERT INTO search_history (query, search_type, timestamp, result_count, response_time_ms) VALUES (?, ?, ?, ?, ?)",
            ("old query", "retrieve_memory", old_timestamp, 5, 100.0),
        )
        await db.commit()

    # Log a recent search
    recent_entry = SearchHistoryEntry("recent query", "memory_scan", result_count=3, response_time_ms=50.0)
    await temp_db.log_search(recent_entry)

    # Get stats for last 30 days
    stats = await temp_db.get_search_stats(days=30)

    # Should only include recent search
    assert stats["total_searches"] == 1
    assert len(stats["popular_queries"]) == 1
    assert stats["popular_queries"][0]["query"] == "recent query"


@pytest.mark.asyncio
async def test_duplicate_search_handling(temp_db):
    """Test that duplicate searches are tracked separately."""
    entry1 = SearchHistoryEntry("test query", "retrieve_memory", result_count=5, response_time_ms=100.0)

    # Log same query multiple times
    await temp_db.log_search(entry1)
    await asyncio.sleep(0.01)
    await temp_db.log_search(entry1)
    await asyncio.sleep(0.01)
    await temp_db.log_search(entry1)

    stats = await temp_db.get_search_stats(days=1)
    assert stats["total_searches"] == 3

    # Check that popular queries shows correct frequency
    popular = stats["popular_queries"]
    assert len(popular) == 1
    assert popular[0]["query"] == "test query"
    assert popular[0]["count"] == 3


@pytest.mark.asyncio
async def test_empty_database_stats(temp_db):
    """Test getting stats from empty database."""
    stats = await temp_db.get_search_stats(days=30)

    assert stats["total_searches"] == 0
    assert stats["avg_response_time"] is None
    assert len(stats["search_types"]) == 0
    assert len(stats["popular_queries"]) == 0


@pytest.mark.asyncio
async def test_parameters_serialization(temp_db):
    """Test that search parameters are correctly serialized to JSON."""
    complex_params = {
        "page": 1,
        "page_size": 10,
        "min_similarity": 0.75,
        "tags": ["python", "testing"],
    }

    entry = SearchHistoryEntry(
        query="test",
        search_type="retrieve_memory",
        parameters=complex_params,
    )

    record_id = await temp_db.log_search(entry)

    # Retrieve and verify
    recent = await temp_db.get_recent_searches(limit=1)
    assert len(recent) == 1
    assert recent[0].parameters == complex_params
