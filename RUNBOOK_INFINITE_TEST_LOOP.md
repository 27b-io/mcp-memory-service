# Operational Runbook: Detecting and Resolving Infinite Test Loops

**Document Type**: Operational Runbook
**Audience**: Witness agents, Deacon, Mayor
**Last Updated**: 2026-02-17
**Related Incident**: mm-nt7lt

## Purpose

This runbook provides procedures for detecting, diagnosing, and resolving infinite test loops that cause system instability.

## Detection Signatures

### Symptom 1: High CPU Usage by pytest

**Indicators:**
- `pytest` or `python -m pytest` consuming 90%+ CPU for > 5 minutes
- Multiple pytest processes running simultaneously on the same project
- System-wide agent instability (witness kills, mayor kills)

**Detection Command:**
```bash
# Check for high-CPU pytest processes
ps aux | awk '$3 > 90.0 && /pytest/ {print $2, $3, $10, $11, $12, $13}'

# Check pytest runtime (shows seconds)
for pid in $(pgrep -f pytest); do
    echo "PID: $pid Runtime: $(ps -p $pid -o etimes=)s"
done
```

**Threshold**: Any pytest process running > 1800s (30 minutes) is suspicious.

### Symptom 2: Embedded Qdrant Lock Contention

**Indicators:**
- Multiple pytest processes on same project
- One process at 90%+ CPU (holding lock)
- Other process at 90%+ CPU (spinning on lock acquisition)
- No progress in test output

**Detection Command:**
```bash
# Check for multiple pytest processes on same project
pgrep -fa pytest | grep -E "mcp_memory|cachekit|cetana|prajna" | sort

# Check Qdrant lock files
find ~/gt -name "*.qdrant.lock" -exec ls -lh {} \;
```

### Symptom 3: Agent Kill Storm

**Indicators:**
- Multiple agent kills in short time window (< 1 hour)
- Kills concentrated on witness agents and mayor
- Town log shows repeated kill entries

**Detection Command:**
```bash
# Check kill storm in last hour
grep "\[kill\]" ~/gt/logs/town.log | grep "$(date -d '1 hour ago' +%Y-%m-%d)"

# Count kills by rig
grep "\[kill\]" ~/gt/logs/town.log | awk '{print $3}' | sort | uniq -c | sort -rn
```

**Threshold**: > 5 agent kills in 1 hour indicates systemic issue.

## Diagnosis Procedure

### Step 1: Identify Runaway Processes

```bash
# Get all pytest processes with runtime and CPU
ps aux | awk '/pytest/ && !/awk/ {print $2, $3, $10, $11, $12, $13}'

# Get full command line for each
for pid in $(pgrep pytest); do
    echo "=== PID $pid ==="
    ps -p $pid -o pid,ppid,etime,pcpu,cmd
done
```

### Step 2: Check Test Status

```bash
# Check pytest cache for failed tests
find ~/gt -name lastfailed -exec echo "=== {} ===" \; -exec cat {} \; -exec echo "" \;

# Check for pytest logs
find ~/gt -name "pytest.log" -o -name "test*.log" -mmin -60
```

### Step 3: Identify Lock Contention

```bash
# Check for Qdrant lock files
find ~/gt -type f -name "*.lock" | grep -i qdrant

# Check for multiple pytest on same rig
for rig in mcp_memory cachekit_saas cetana prajna tj1w; do
    count=$(pgrep -fc "pytest.*$rig")
    if [ "$count" -gt 1 ]; then
        echo "⚠️  $rig has $count pytest processes (lock contention likely)"
    fi
done
```

### Step 4: Review Logs

```bash
# Check overseer log for stuck agents
tail -50 ~/gt/overseer/overseer.log | grep -E "refinery|pytest"

# Check town log for related events
grep "$(date +%Y-%m-%d)" ~/gt/logs/town.log | grep -E "refinery|pytest|kill" | tail -30

# Check guardian log
tail -50 ~/gt/logs/guardian.log
```

## Resolution Procedures

### Immediate Response (Witness/Deacon)

**Goal**: Stop the bleeding - kill runaway processes

```bash
#!/bin/bash
# kill-runaway-pytest.sh

# Identify runaway pytest processes (> 30 min runtime)
echo "Scanning for runaway pytest processes..."
for pid in $(pgrep -f pytest); do
    runtime=$(ps -p $pid -o etimes= 2>/dev/null || echo 0)
    if [ "$runtime" -gt 1800 ]; then
        cmd=$(ps -p $pid -o cmd=)
        echo "⚠️  RUNAWAY: PID $pid (${runtime}s) - $cmd"
        echo "   Killing..."
        kill -TERM $pid
        sleep 2
        # Force kill if still alive
        if ps -p $pid > /dev/null 2>&1; then
            echo "   Force killing..."
            kill -9 $pid
        fi
        echo "   ✓ Killed PID $pid"

        # Notify mayor
        gt nudge hq-mayor "WITNESS: Killed runaway pytest PID $pid on $(hostname) after ${runtime}s runtime. Lock contention suspected."
    fi
done
```

**Execution**:
```bash
# As witness or deacon
chmod +x ~/gt/scripts/kill-runaway-pytest.sh
~/gt/scripts/kill-runaway-pytest.sh
```

### Root Cause Mitigation (Refinery/Mayor)

**Goal**: Prevent recurrence

#### 1. Verify Timeout Guards Installed

```bash
# Check if pytest-timeout is installed
uv pip list | grep pytest-timeout || echo "⚠️  pytest-timeout NOT installed"

# Check pytest.ini for timeout config
grep -E "^timeout|timeout_method" pytest.ini || echo "⚠️  No timeout config in pytest.ini"
```

**If missing**: Follow `TIMEOUT_GUARD_PLAN.md` to install and configure.

#### 2. Check Test Configuration

```bash
# Verify integration tests have timeout markers
grep -r "pytestmark.*timeout" tests/integration/ || echo "⚠️  No timeout markers on integration tests"

# Check for embedded Qdrant usage in tests
grep -r "QdrantClient.*path=" tests/ && echo "⚠️  Tests using embedded Qdrant (lock contention risk)"
```

#### 3. Add Pre-Flight Checks

If refinery auto-runs tests, add this check to the workflow:

```bash
# Before running pytest
if pgrep -f "pytest.*$(basename $PWD)" > /dev/null; then
    echo "⚠️  ERROR: pytest already running, aborting to prevent lock contention"
    pgrep -fa "pytest.*$(basename $PWD)"
    exit 1
fi
```

### Long-Term Prevention (Mayor)

**Goal**: Architectural fixes

#### 1. Move to Server-Mode Qdrant for Tests

- Spin up ephemeral Qdrant server for test runs
- Avoids exclusive lock contention
- Allows parallel test execution

```bash
# Start test Qdrant server
docker run -d --rm -p 6333:6333 --name qdrant-test qdrant/qdrant

# Run tests with server mode
export QDRANT_URL=http://localhost:6333
pytest tests/integration/

# Cleanup
docker stop qdrant-test
```

#### 2. Implement Test Isolation

- Each test run gets isolated Qdrant collection
- Cleanup after test completion
- No shared state between test runs

#### 3. Add Monitoring

**GT Guardian Enhancement**:

```bash
# Add to gt-guardian.sh
check_runaway_pytest() {
    for pid in $(pgrep -f pytest); do
        runtime=$(ps -p $pid -o etimes=)
        if [ "$runtime" -gt 1800 ]; then
            logger "GT-GUARDIAN: Runaway pytest detected PID=$pid runtime=${runtime}s"
            kill -TERM $pid
        fi
    done
}

# Call in main loop
check_runaway_pytest
```

## Escalation

### When to Escalate to Mayor

- Runaway pytest processes return after being killed (systemic issue)
- Kill storm continues after pytest termination (deeper problem)
- Multiple rigs affected simultaneously (infrastructure issue)

**Escalation Command**:
```bash
gt nudge hq-mayor "ESCALATION: Runaway pytest on $RIG_NAME. Root cause: [embedded Qdrant lock|no timeouts|unknown]. Killed PIDs: $PIDS. Recommend: [action]"
```

### When to Escalate to Deacon

- Guardian not killing runaway processes (monitoring failure)
- Overseer not detecting stuck agents (classification failure)
- Systemd services not restarting (infrastructure failure)

**Escalation Command**:
```bash
gt nudge deacon "ESCALATION: Guardian/Overseer failure. Runaway pytest not detected. PIDs: $PIDS. Manual intervention required."
```

## Post-Incident Actions

### Required

1. **Store incident in MCP memory**:
   ```bash
   # Store root cause analysis
   mcp-memory store "Incident: Runaway pytest on $RIG_NAME. Root cause: $CAUSE. Resolution: $RESOLUTION. Prevention: $PREVENTION" \
     --tags incident,testing,$RIG_NAME,pytest \
     --type note
   ```

2. **Update runbook** if new patterns discovered:
   ```bash
   # Add new detection signature or resolution procedure
   vim ~/gt/docs/runbooks/infinite-test-loop.md
   ```

3. **Create bead for permanent fix**:
   ```bash
   bd create "Implement pytest timeout guards on $RIG_NAME" \
     --type bug \
     --priority P1 \
     --description "See TIMEOUT_GUARD_PLAN.md for implementation details"
   ```

### Recommended

1. **Review similar risks** across all rigs:
   ```bash
   for rig in ~/gt/*/refinery/rig; do
       echo "=== $rig ==="
       grep -E "^timeout" "$rig/pytest.ini" 2>/dev/null || echo "⚠️  No timeout config"
   done
   ```

2. **Add alerting** for early detection:
   ```bash
   # Add to overseer or witness
   # Alert on pytest runtime > 600s (10 min)
   ```

3. **Document in project memory**:
   ```bash
   # Update rig-specific CLAUDE.md with lessons learned
   echo "## Testing Guardrails\n\n- ALWAYS use pytest-timeout\n- NEVER use embedded Qdrant in tests\n- ALWAYS add pre-flight checks before test runs" >> ~/gt/$RIG_NAME/refinery/CLAUDE.md
   ```

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ INFINITE TEST LOOP QUICK RESPONSE                           │
├─────────────────────────────────────────────────────────────┤
│ 1. DETECT                                                   │
│    ps aux | awk '$3 > 90 && /pytest/'                       │
│                                                             │
│ 2. IDENTIFY                                                 │
│    for pid in $(pgrep pytest); do                           │
│      ps -p $pid -o etime,pcpu,cmd                          │
│    done                                                     │
│                                                             │
│ 3. KILL                                                     │
│    kill -TERM <pid>  # Wait 2s                             │
│    kill -9 <pid>     # If still alive                      │
│                                                             │
│ 4. NOTIFY                                                   │
│    gt nudge hq-mayor "Killed runaway pytest PID <pid>"     │
│                                                             │
│ 5. PREVENT                                                  │
│    - Install pytest-timeout                                 │
│    - Add pre-flight checks                                 │
│    - Switch to server-mode Qdrant                          │
│                                                             │
│ THRESHOLDS:                                                 │
│ - CPU: > 90% for > 5 min = suspicious                      │
│ - Runtime: > 30 min = runaway                              │
│ - Kill storm: > 5 kills/hour = systemic issue              │
└─────────────────────────────────────────────────────────────┘
```

## Change Log

| Date       | Change                                  | Author   |
|------------|-----------------------------------------|----------|
| 2026-02-17 | Initial runbook created from mm-nt7lt   | furiosa  |

## See Also

- `ROOT_CAUSE_ANALYSIS.md` - Detailed investigation of mm-nt7lt
- `TIMEOUT_GUARD_PLAN.md` - Implementation plan for timeout guards
- `~/gt/AGENTS.md` - Agent roles and responsibilities
- `~/gt/overseer/README.md` - Overseer monitoring system
