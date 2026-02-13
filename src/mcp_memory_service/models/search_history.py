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

"""Search history models for analytics and recommendations."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHistoryEntry:
    """Represents a search query execution for analytics."""

    # Query information
    query: str
    search_type: str  # retrieve_memory, memory_scan, find_similar, search_by_tag

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Results metadata
    result_count: int = 0
    response_time_ms: float = 0.0

    # Search parameters (stored as JSON string)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Client identification (optional)
    client_id: str | None = None

    # Unique ID (auto-generated)
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "query": self.query,
            "search_type": self.search_type,
            "timestamp": self.timestamp,
            "result_count": self.result_count,
            "response_time_ms": self.response_time_ms,
            "parameters": self.parameters,
            "client_id": self.client_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchHistoryEntry":
        """Create instance from dictionary."""
        return cls(
            id=data.get("id"),
            query=data["query"],
            search_type=data["search_type"],
            timestamp=data["timestamp"],
            result_count=data.get("result_count", 0),
            response_time_ms=data.get("response_time_ms", 0.0),
            parameters=data.get("parameters", {}),
            client_id=data.get("client_id"),
        )
