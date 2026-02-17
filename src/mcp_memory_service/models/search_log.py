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

"""Search query logging models for analytics."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchLog:
    """Represents a single search/query operation for analytics tracking."""

    query: str
    timestamp: float
    response_time_ms: float
    result_count: int
    tags: list[str] | None = None
    memory_type: str | None = None
    min_similarity: float | None = None
    hybrid_enabled: bool = True
    keywords_extracted: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "response_time_ms": self.response_time_ms,
            "result_count": self.result_count,
            "tags": self.tags,
            "memory_type": self.memory_type,
            "min_similarity": self.min_similarity,
            "hybrid_enabled": self.hybrid_enabled,
            "keywords_extracted": self.keywords_extracted,
            "error": self.error,
            "metadata": self.metadata,
        }
