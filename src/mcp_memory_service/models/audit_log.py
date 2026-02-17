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

"""Audit logging models for tracking memory operations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditLog:
    """Represents a single memory operation for audit trail tracking."""

    operation: str  # CREATE, DELETE, DELETE_RELATION
    content_hash: str
    timestamp: float
    actor: str | None = None  # client_hostname or similar
    memory_type: str | None = None
    tags: list[str] | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }
