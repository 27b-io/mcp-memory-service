"""Tests for audit logging functionality."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.audit_log import AuditLog
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.delete.return_value = (True, "Deleted")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    return storage


@pytest.fixture
def memory_service_with_audit_logs(mock_storage):
    """Create a MemoryService with some audit logs."""
    service = MemoryService(storage=mock_storage)

    # Simulate some operations
    service._log_audit(
        operation="CREATE",
        content_hash="hash1",
        actor="client1",
        memory_type="note",
        tags=["python", "testing"],
        success=True,
    )
    service._log_audit(
        operation="CREATE",
        content_hash="hash2",
        actor="client1",
        memory_type="decision",
        tags=["architecture"],
        success=True,
    )
    service._log_audit(
        operation="DELETE",
        content_hash="hash1",
        actor="client2",
        success=True,
    )
    service._log_audit(
        operation="DELETE",
        content_hash="hash3",
        actor="client1",
        success=False,
        error="Memory not found",
    )
    service._log_audit(
        operation="DELETE_RELATION",
        content_hash="hash2",
        actor="client2",
        success=True,
        metadata={"target_hash": "hash4", "relation_type": "RELATES_TO"},
    )

    return service


def test_audit_log_creation():
    """Test AuditLog model creation."""
    log = AuditLog(
        operation="CREATE",
        content_hash="test-hash",
        timestamp=1000.0,
        actor="test-client",
        memory_type="note",
        tags=["python"],
        success=True,
    )

    assert log.operation == "CREATE"
    assert log.content_hash == "test-hash"
    assert log.timestamp == 1000.0
    assert log.actor == "test-client"
    assert log.memory_type == "note"
    assert log.tags == ["python"]
    assert log.success is True
    assert log.error is None


def test_audit_log_to_dict():
    """Test AuditLog to_dict conversion."""
    log = AuditLog(
        operation="DELETE",
        content_hash="test-hash",
        timestamp=1000.0,
        actor="test-client",
        success=False,
        error="Not found",
    )

    data = log.to_dict()
    assert data["operation"] == "DELETE"
    assert data["content_hash"] == "test-hash"
    assert data["timestamp"] == 1000.0
    assert data["actor"] == "test-client"
    assert data["success"] is False
    assert data["error"] == "Not found"


def test_log_audit(mock_storage):
    """Test _log_audit method."""
    service = MemoryService(storage=mock_storage)

    # Initially no logs
    assert len(service._audit_logs) == 0

    # Log an operation
    service._log_audit(
        operation="CREATE",
        content_hash="test-hash",
        actor="test-client",
        success=True,
    )

    # Should have one log
    assert len(service._audit_logs) == 1
    log = service._audit_logs[0]
    assert log.operation == "CREATE"
    assert log.content_hash == "test-hash"
    assert log.actor == "test-client"
    assert log.success is True


def test_get_audit_trail_empty(mock_storage):
    """Test get_audit_trail with no logs."""
    service = MemoryService(storage=mock_storage)

    audit_trail = service.get_audit_trail()

    assert audit_trail["total_operations"] == 0
    assert audit_trail["operations"] == []
    assert audit_trail["operations_by_type"] == {}
    assert audit_trail["operations_by_actor"] == {}
    assert audit_trail["success_rate"] == 1.0


def test_get_audit_trail_with_logs(memory_service_with_audit_logs):
    """Test get_audit_trail with audit logs."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail()

    assert audit_trail["total_operations"] == 5
    assert len(audit_trail["operations"]) == 5
    assert audit_trail["operations_by_type"]["CREATE"] == 2
    assert audit_trail["operations_by_type"]["DELETE"] == 2
    assert audit_trail["operations_by_type"]["DELETE_RELATION"] == 1
    assert audit_trail["operations_by_actor"]["client1"] == 3
    assert audit_trail["operations_by_actor"]["client2"] == 2
    assert audit_trail["success_rate"] == 0.8  # 4 out of 5 succeeded


def test_get_audit_trail_filter_by_operation(memory_service_with_audit_logs):
    """Test filtering audit trail by operation type."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail(operation="CREATE")

    assert audit_trail["total_operations"] == 2
    assert all(entry["operation"] == "CREATE" for entry in audit_trail["operations"])


def test_get_audit_trail_filter_by_actor(memory_service_with_audit_logs):
    """Test filtering audit trail by actor."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail(actor="client1")

    assert audit_trail["total_operations"] == 3
    assert all(entry["actor"] == "client1" for entry in audit_trail["operations"])


def test_get_audit_trail_filter_by_content_hash(memory_service_with_audit_logs):
    """Test filtering audit trail by content hash."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail(content_hash="hash1")

    assert audit_trail["total_operations"] == 2
    assert all(entry["content_hash"] == "hash1" for entry in audit_trail["operations"])


def test_get_audit_trail_limit(memory_service_with_audit_logs):
    """Test limiting audit trail results."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail(limit=2)

    assert audit_trail["total_operations"] == 5  # Total unfiltered
    assert len(audit_trail["operations"]) == 2  # But only 2 returned


def test_get_audit_trail_sorted_by_timestamp(memory_service_with_audit_logs):
    """Test that audit trail is sorted by timestamp descending."""
    audit_trail = memory_service_with_audit_logs.get_audit_trail()

    timestamps = [entry["timestamp"] for entry in audit_trail["operations"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_audit_circular_buffer(mock_storage):
    """Test that audit logs use circular buffer (max 10K)."""
    service = MemoryService(storage=mock_storage)

    # Add more than max capacity
    for i in range(service._MAX_AUDIT_LOGS + 100):
        service._log_audit(
            operation="CREATE",
            content_hash=f"hash{i}",
            actor="test-client",
            success=True,
        )

    # Should be capped at max
    assert len(service._audit_logs) == service._MAX_AUDIT_LOGS


@pytest.mark.asyncio
async def test_store_memory_logs_audit(mock_storage):
    """Test that store_memory logs audit trail."""
    service = MemoryService(storage=mock_storage)

    result = await service.store_memory(
        content="test content",
        tags=["test"],
        memory_type="note",
        client_hostname="test-client",
    )

    assert result["success"] is True
    assert len(service._audit_logs) == 1
    log = service._audit_logs[0]
    assert log.operation == "CREATE"
    assert log.actor == "test-client"
    assert log.success is True


@pytest.mark.asyncio
async def test_delete_memory_logs_audit(mock_storage):
    """Test that delete_memory logs audit trail."""
    service = MemoryService(storage=mock_storage)

    result = await service.delete_memory("test-hash")

    assert result["success"] is True
    assert len(service._audit_logs) == 1
    log = service._audit_logs[0]
    assert log.operation == "DELETE"
    assert log.content_hash == "test-hash"
    assert log.success is True


@pytest.mark.asyncio
async def test_delete_memory_failed_logs_audit(mock_storage):
    """Test that failed delete_memory logs audit trail."""
    mock_storage.delete.return_value = (False, "Not found")
    service = MemoryService(storage=mock_storage)

    result = await service.delete_memory("test-hash")

    assert result["success"] is False
    assert len(service._audit_logs) == 1
    log = service._audit_logs[0]
    assert log.operation == "DELETE"
    assert log.content_hash == "test-hash"
    assert log.success is False
    assert log.error == "Not found"
