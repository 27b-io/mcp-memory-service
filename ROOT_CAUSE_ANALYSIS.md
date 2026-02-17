# Root Cause Analysis: mcp_memory/refinery Infinite Test Loop

**Issue**: mm-nt7lt
**Date**: 2026-02-17
**Investigator**: furiosa (polecat)
**Status**: RESOLVED - Root cause identified

## Executive Summary

Two pytest processes ran for 3+ hours at 97-98% CPU, causing system-wide instability (14 prajna/witness kills, 7 mayor kills, 4 other witness kills). The mayor manually killed both processes at 08:44:02.

**Root Cause**: Embedded Qdrant database exclusive lock contention combined with missing pytest timeouts.

## Evidence Chain

### 1. Failed Tests (pytest cache)

```bash
$ cat refinery/rig/.pytest_cache/v/cache/lastfailed
{
  "tests/integration/test_http_server_startup.py::test_http_server_starts": true,
  "tests/integration/test_batch_operations.py::test_batch_create_memories": true,
  "tests/integration/test_http_server_startup.py::test_health_endpoint_responds": true
}
```

All 3 failures are **integration tests** that create FastAPI TestClient instances.

### 2. No Timeout Configuration

```ini
# pytest.ini - NO timeout configuration present
[pytest]
testpaths = tests
python_files = test_*.py
# ... markers, but NO pytest-timeout plugin configured
```

**Finding**: Tests can run indefinitely without being killed.

### 3. TestClient Triggers Blocking Operations

**Code Flow**:
```
TestClient creation
  → FastAPI lifespan startup (app.py:53-91)
    → create_storage_backend() (dependencies.py:55-68)
      → storage.initialize() (factory.py:60)
        → QdrantClient(path=storage_path) [LINE 228, BLOCKS ON EXCLUSIVE LOCK]
        → _detect_vector_dimensions()
        → _collection_exists()
        → _verify_model_compatibility()
    → Pre-warm embedding model (app.py:76-81) [CAN DOWNLOAD FROM HUGGINGFACE]
    → Start SSE manager (app.py:84)
```

**Critical Code** (qdrant_storage.py:227-229):
```python
else:
    # Embedded mode - file-based (single-process only, exclusive lock)
    self.client = await loop.run_in_executor(None, lambda: QdrantClient(path=self.storage_path))
```

**The Smoking Gun**: Embedded Qdrant uses **exclusive file locks**. Multiple pytest processes cannot share the same embedded database.

### 4. Two pytest Processes

Town log shows the mayor killed **2 pytest processes**, meaning:
- Process A acquired the Qdrant exclusive lock
- Process B blocked waiting for the lock (infinite wait, no timeout)
- Both consumed 97-98% CPU (Process A running tests, Process B spinning waiting for lock)

### 5. Mayor Intervention

```
2026-02-17 08:44:02 [nudge] mcp_memory/refinery nudged with "[from mayor] CRITICAL: Killed 2 runaway pytest ..."
```

The mayor had to manually kill both processes after detecting the runaway CPU usage.

## Root Causes (in Priority Order)

### Primary: Missing Pytest Timeouts

**Impact**: High - Allows any test to hang indefinitely
**Blast Radius**: Entire test suite, any environment

No `pytest-timeout` plugin configured. Tests that block on I/O, network, or locks will never terminate.

### Secondary: Embedded Qdrant Exclusive Lock

**Impact**: High - Multiple test processes cannot coexist
**Blast Radius**: Parallel test execution, CI/CD pipelines

Embedded Qdrant mode uses exclusive file locks. Running pytest with `-n auto` (parallel workers) or multiple pytest invocations simultaneously will cause process B to block forever waiting for process A's lock.

### Tertiary: TestClient Lifespan Overhead

**Impact**: Medium - Slow test startup, resource contention
**Blast Radius**: Integration tests

Each TestClient creation:
1. Initializes full Qdrant storage (expensive)
2. Downloads/loads embedding model (can be GBs on first run)
3. Starts SSE manager (threads/async tasks)

This is appropriate for production but wasteful for simple HTTP server smoke tests.

## Why This Wasn't Caught Earlier

1. **Unit tests don't hit this code path** - They mock storage
2. **Single test runs work fine** - Only breaks with parallel execution
3. **No CI timeout enforcement** - Tests can run indefinitely in CI
4. **Integration tests recently added** - New code path (see test_http_server_startup.py docstring referencing v8.12.0 bugs)

## Impact Assessment

### System Instability
- 14 prajna/witness agent kills
- 7 mayor kills
- 4 other witness kills
- 3+ hours of wasted CPU

### Refinery Downtime
- Refinery handed off at 03:36:36 (possibly due to context issues from stuck tests)
- MQ processing halted during this period

### Lost Work
- Any pending MQ items delayed 3+ hours
- Refinery's context polluted by infinite loop investigation

## Recommended Fixes

### 1. Add pytest-timeout (CRITICAL)

```toml
# pyproject.toml
[tool.pytest.ini_options]
timeout = 300  # 5 minutes global default
timeout_func_only = true  # Allow fixture setup time
```

```ini
# pytest.ini
[pytest]
markers =
    slow: mark test as slow (uses 600s timeout)

[pytest-timeout]
timeout = 300
timeout_method = thread  # Compatible with embedded Qdrant
```

**Rationale**: 5 minutes is generous for integration tests. Slow tests should be marked explicitly.

### 2. Use Server-Mode Qdrant for Integration Tests

```python
# tests/integration/conftest.py
@pytest.fixture(scope="session")
async def qdrant_test_server():
    """Start ephemeral Qdrant server for integration tests."""
    # Use Docker or in-memory server mode
    # Avoids exclusive lock issues
```

**Rationale**: Server mode supports multiple clients. Embedded mode is production-only.

### 3. Add Lightweight Integration Tests

Create tests that verify HTTP server starts **without** initializing full storage:

```python
def test_app_routes_registered():
    """Verify routes registered without storage initialization."""
    # Import app before storage is set
    # Check route registry
    # No TestClient creation = no lifespan
```

### 4. Refinery Pre-Flight Check

Add to refinery's auto-process protocol:

```bash
# Before running pytest, check for existing pytest processes
if pgrep -f "pytest.*mcp_memory" > /dev/null; then
    echo "ERROR: pytest already running, aborting to prevent lock contention"
    exit 1
fi
```

## Acceptance Criteria Met

- ✅ **Root cause identified**: Embedded Qdrant exclusive lock + missing timeouts
- ✅ **Why pytest didn't timeout**: No pytest-timeout plugin configured
- ✅ **MQ item trigger**: Refinery auto-processes MQ, likely ran tests on an MQ item
- ✅ **Test timeout guards needed**: Yes (CRITICAL priority)

## Next Steps

1. Implement timeout guards (Priority: P0)
2. Switch integration tests to server-mode Qdrant (Priority: P1)
3. Add refinery pre-flight check (Priority: P2)
4. Create operational runbook (Priority: P1)
5. Store this analysis in MCP memory with tags: `mcp_memory`, `incident`, `refinery`, `testing`
