# Timeout Guard Implementation Plan

**Issue**: mm-nt7lt
**Date**: 2026-02-17
**Status**: READY FOR IMPLEMENTATION

## Objective

Prevent infinite test loops by adding pytest timeout guards and fixing test isolation issues.

## Changes Required

### 1. Add pytest-timeout Plugin

**File**: `pyproject.toml`

```toml
[project.optional-dependencies]
# ... existing dev dependencies ...
dev = [
    # ... existing entries ...
    "pytest-timeout>=2.3.1",  # ADD THIS
]
```

**File**: `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
timeout = 300  # ADD: 5 minute global timeout
timeout_method = thread  # ADD: Use thread-based timeout (compatible with embedded Qdrant)
markers =
    unit: unit tests
    integration: integration tests (uses 600s timeout)  # MODIFY: Document timeout
    performance: performance tests (uses 1800s timeout)  # MODIFY: Document timeout
    slow: mark test as slow (can be skipped with -m "not slow")
    asyncio: mark test as async
    llm_judge: requires an LLM judge endpoint (not run by default)
filterwarnings =
    ignore:Importing NonLLM.*from 'ragas.metrics' is deprecated:DeprecationWarning
```

### 2. Add Timeout Markers to Integration Tests

**File**: `tests/integration/test_http_server_startup.py`

Add at the top after imports:

```python
import pytest
from fastapi.testclient import TestClient

# All integration tests get 600s timeout (10 minutes for slow CI)
pytestmark = pytest.mark.timeout(600)
```

**File**: `tests/integration/test_qdrant_integration.py` (and other integration test files)

Add the same `pytestmark` decorator.

### 3. Fix Embedded Qdrant Contention

**Option A**: Use server-mode Qdrant for tests (RECOMMENDED)

**File**: `tests/integration/conftest.py`

```python
import os
import pytest
from qdrant_client import QdrantClient

@pytest.fixture(scope="session", autouse=True)
def force_qdrant_server_mode():
    """
    Force integration tests to use Qdrant server mode instead of embedded.

    This prevents exclusive lock contention when running multiple test processes.
    """
    # Check if Qdrant server is running
    try:
        test_client = QdrantClient(url="http://localhost:6333")
        test_client.get_collections()
        # Server is available, use it
        os.environ["QDRANT_URL"] = "http://localhost:6333"
        os.environ.pop("QDRANT_STORAGE_PATH", None)
    except Exception:
        # Server not available, use in-memory mode
        os.environ["QDRANT_URL"] = ":memory:"
        os.environ.pop("QDRANT_STORAGE_PATH", None)

    yield
```

**Option B**: Disable parallel test execution (FALLBACK)

**File**: `pytest.ini`

```ini
[pytest]
# ... existing config ...
addopts = -p no:xdist  # Disable parallel execution if pytest-xdist is installed
```

**Recommendation**: Use Option A. Server mode allows parallel execution and matches production deployment.

### 4. Add Pre-Flight Check to Refinery Workflow

**File**: `refinery/CLAUDE.md`

Add to the "Auto-Process Protocol" section:

```markdown
## Auto-Process Protocol

**When you see items in the merge queue, process them IMMEDIATELY. Do not wait for instructions.**

### Pre-Flight Checks (NEW)

Before processing any MQ item, run these safety checks:

```bash
# Check for existing pytest processes to prevent lock contention
if pgrep -f "pytest.*mcp_memory" > /dev/null; then
    echo "⚠️  WARNING: pytest already running on mcp_memory"
    echo "Aborting to prevent embedded Qdrant lock contention"
    ps aux | grep -E "pytest.*mcp_memory" | grep -v grep
    exit 1
fi

# Check for runaway processes (> 30 min runtime)
for pid in $(pgrep -f "pytest.*mcp_memory"); do
    runtime=$(ps -p $pid -o etimes= 2>/dev/null || echo 0)
    if [ "$runtime" -gt 1800 ]; then
        echo "⚠️  CRITICAL: pytest process $pid has been running for ${runtime}s (>30min)"
        echo "This indicates a timeout failure. Investigate before proceeding."
        exit 1
    fi
done
```

### Quality Gates (in order)

1. **Pre-flight checks pass** (NEW)
2. Branch is up-to-date with main (no conflicts)
...
```

## Implementation Order

1. **Add pytest-timeout plugin** (5 min)
   - Update `pyproject.toml`
   - Run `uv sync` to install

2. **Update pytest.ini** (2 min)
   - Add timeout configuration
   - Update marker descriptions

3. **Add timeout markers to integration tests** (10 min)
   - Add `pytestmark = pytest.mark.timeout(600)` to each integration test file
   - Verify with `grep -r "pytestmark.*timeout" tests/integration/`

4. **Add Qdrant server-mode fixture** (15 min)
   - Update `tests/integration/conftest.py`
   - Test with `pytest tests/integration/test_http_server_startup.py -v`

5. **Update refinery workflow** (5 min)
   - Add pre-flight check to `refinery/CLAUDE.md`

6. **Verify all changes** (10 min)
   - Run full test suite: `pytest -v --timeout=60`
   - Should see timeout warnings if configured correctly
   - All tests should pass

## Testing the Fix

### Verify Timeout Works

Create a test that intentionally hangs:

```python
# tests/test_timeout_guard.py
import time
import pytest

@pytest.mark.timeout(5)
def test_timeout_enforcement():
    """Verify pytest-timeout kills hanging tests."""
    time.sleep(10)  # Should be killed at 5 seconds
```

Run: `pytest tests/test_timeout_guard.py -v`

Expected output:
```
tests/test_timeout_guard.py::test_timeout_enforcement FAILED
...
E   Failed: Timeout >5.0s
```

### Verify Qdrant Server Mode

```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Run integration tests
pytest tests/integration/test_http_server_startup.py -v

# Check logs - should see "Connected to Qdrant server at http://localhost:6333"
```

### Verify Pre-Flight Check

```bash
# In one terminal, start a long-running pytest
pytest tests/ --timeout=600 &

# In another terminal (as refinery)
# Try to process MQ item - should abort with warning
pgrep -f "pytest.*mcp_memory" && echo "Pre-flight check would block"
```

## Success Criteria

- ✅ pytest-timeout plugin installed and configured
- ✅ All tests have reasonable timeouts (300s default, 600s integration, 1800s performance)
- ✅ Integration tests use server-mode Qdrant (no exclusive lock contention)
- ✅ Refinery pre-flight check prevents concurrent pytest runs
- ✅ Test suite completes in < 10 minutes
- ✅ Hanging test kills itself after timeout
- ✅ No more infinite loops possible

## Rollback Plan

If timeout guards cause false positives (legitimate slow tests timing out):

1. Increase timeout for specific tests:
   ```python
   @pytest.mark.timeout(1800)  # 30 minutes for slow test
   def test_very_slow_operation():
       ...
   ```

2. Disable timeout for specific tests:
   ```python
   @pytest.mark.timeout(0)  # No timeout
   def test_manual_verification():
       ...
   ```

3. Adjust global timeout in `pytest.ini`:
   ```ini
   timeout = 600  # Increase to 10 minutes if 5 minutes is too aggressive
   ```

## Notes

- Timeout guards are **non-negotiable** for production testing
- 5 minutes is generous for most integration tests
- Server-mode Qdrant is production-realistic and avoids lock contention
- Pre-flight checks add < 1s overhead but prevent 3+ hour hangs
