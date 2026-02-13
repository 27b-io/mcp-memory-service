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

"""
Search history database manager.

Manages SQLite database for tracking search queries, results, and analytics.
Thread-safe async operations using aiosqlite.
"""

import json
import logging
import os
from typing import Any

import aiosqlite

from ..models.search_history import SearchHistoryEntry

logger = logging.getLogger(__name__)


class SearchHistoryDB:
    """Async SQLite database manager for search history tracking."""

    def __init__(self, db_path: str):
        """
        Initialize search history database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialized = False

    async def initialize(self):
        """Initialize database schema if not exists."""
        if self._initialized:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    search_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    result_count INTEGER DEFAULT 0,
                    response_time_ms REAL DEFAULT 0.0,
                    parameters TEXT,
                    client_id TEXT
                )
            """
            )

            # Create indices for common queries
            await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON search_history(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_search_type ON search_history(search_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_history(query)")

            await db.commit()

        self._initialized = True
        logger.info(f"Search history database initialized at {self.db_path}")

    async def log_search(self, entry: SearchHistoryEntry) -> int:
        """
        Log a search query execution.

        Args:
            entry: Search history entry to log

        Returns:
            ID of the inserted record
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO search_history
                (query, search_type, timestamp, result_count, response_time_ms, parameters, client_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.query,
                    entry.search_type,
                    entry.timestamp,
                    entry.result_count,
                    entry.response_time_ms,
                    json.dumps(entry.parameters) if entry.parameters else None,
                    entry.client_id,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_search_stats(self, days: int = 30) -> dict[str, Any]:
        """
        Get aggregate search statistics for the last N days.

        Args:
            days: Number of days to include in stats

        Returns:
            Dictionary with search statistics
        """
        if not self._initialized:
            await self.initialize()

        import time

        cutoff_time = time.time() - (days * 86400)

        async with aiosqlite.connect(self.db_path) as db:
            # Total searches
            cursor = await db.execute("SELECT COUNT(*) FROM search_history WHERE timestamp >= ?", (cutoff_time,))
            row = await cursor.fetchone()
            total_searches = row[0] if row else 0

            # Average response time
            cursor = await db.execute(
                "SELECT AVG(response_time_ms) FROM search_history WHERE timestamp >= ? AND response_time_ms > 0",
                (cutoff_time,),
            )
            row = await cursor.fetchone()
            avg_response_time = row[0] if row and row[0] else None

            # Search types distribution
            cursor = await db.execute(
                "SELECT search_type, COUNT(*) FROM search_history WHERE timestamp >= ? GROUP BY search_type",
                (cutoff_time,),
            )
            search_types = {row[0]: row[1] for row in await cursor.fetchall()}

            # Popular queries (top 10)
            cursor = await db.execute(
                """
                SELECT query, COUNT(*) as count
                FROM search_history
                WHERE timestamp >= ?
                GROUP BY query
                ORDER BY count DESC
                LIMIT 10
            """,
                (cutoff_time,),
            )
            popular_queries = [{"query": row[0], "count": row[1]} for row in await cursor.fetchall()]

            return {
                "total_searches": total_searches,
                "avg_response_time": avg_response_time,
                "search_types": search_types,
                "popular_queries": popular_queries,
                "period_days": days,
            }

    async def get_recent_searches(self, limit: int = 100) -> list[SearchHistoryEntry]:
        """
        Get recent search history entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of search history entries
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, query, search_type, timestamp, result_count, response_time_ms, parameters, client_id
                FROM search_history
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            entries = []
            for row in await cursor.fetchall():
                parameters = json.loads(row[6]) if row[6] else {}
                entry = SearchHistoryEntry(
                    id=row[0],
                    query=row[1],
                    search_type=row[2],
                    timestamp=row[3],
                    result_count=row[4],
                    response_time_ms=row[5],
                    parameters=parameters,
                    client_id=row[7],
                )
                entries.append(entry)

            return entries

    async def get_query_recommendations(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get query recommendations based on popular searches.

        Args:
            limit: Maximum number of recommendations

        Returns:
            List of recommended queries with metadata
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Get popular queries from last 7 days
            import time

            cutoff_time = time.time() - (7 * 86400)

            cursor = await db.execute(
                """
                SELECT
                    query,
                    COUNT(*) as frequency,
                    AVG(result_count) as avg_results,
                    search_type
                FROM search_history
                WHERE timestamp >= ? AND result_count > 0
                GROUP BY query
                ORDER BY frequency DESC, avg_results DESC
                LIMIT ?
            """,
                (cutoff_time, limit),
            )

            recommendations = []
            for row in await cursor.fetchall():
                recommendations.append({"query": row[0], "frequency": row[1], "avg_results": row[2], "search_type": row[3]})

            return recommendations

    async def close(self):
        """Close database connections."""
        # aiosqlite doesn't maintain persistent connections, so nothing to close
        pass
