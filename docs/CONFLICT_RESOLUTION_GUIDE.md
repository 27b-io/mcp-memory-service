# Memory Conflict Resolution - User Guide

## Overview

MCP Memory Service now detects and resolves conflicts between memories - situations where:
- New information contradicts existing memories (negation, antonyms, temporal changes)
- Same memory is updated concurrently
- Near-duplicate memories exist

**Conflict Resolution Strategies:**
1. **Automatic (CRDT)**: Last-write-wins based on timestamps (opt-in)
2. **Manual**: API-driven resolution with full control

## Quick Start

### Enabling Conflict Detection

Conflict detection is **enabled by default**. Contradictions are automatically detected when storing new memories and flagged with `CONTRADICTS` graph edges.

```bash
# Already enabled by default
MCP_CONFLICT_ENABLED=true
```

### Option 1: Automatic Resolution (CRDT)

**Last-write-wins** strategy - automatically resolves conflicts by keeping the most recently updated memory.

```bash
# Enable auto-resolution (disabled by default - opt-in)
MCP_CONFLICT_AUTO_RESOLVE=true
```

**Behavior:**
- Compares `updated_at` timestamps
- Winner keeps content, loser deleted
- Tags automatically merged (union)
- Resolution recorded in `conflict_history`

**When it triggers:**
1. **Store time**: When contradiction detected while storing new memory
2. **Consolidation**: Batch-resolves unresolved conflicts during periodic cleanup

### Option 2: Manual Resolution (API)

Keep auto-resolve disabled (default) and use the API to review and resolve conflicts manually.

## Configuration

### Environment Variables

```bash
# Core conflict resolution
MCP_CONFLICT_ENABLED=true                    # Enable conflict detection (default: true)
MCP_CONFLICT_AUTO_RESOLVE=false              # Auto-resolve using CRDT (default: false)

# Duplicate handling
MCP_CONFLICT_AUTO_RESOLVE_DUPLICATES=true    # Auto-merge duplicates (default: true)
MCP_CONFLICT_DUPLICATE_SIMILARITY_THRESHOLD=0.95  # Cosine similarity for duplicates (default: 0.95)
MCP_CONFLICT_QUEUE_DUPLICATES_FOR_REVIEW=false    # Queue duplicates for manual review (default: false)

# Advanced (future)
MCP_CONFLICT_OPTIMISTIC_LOCKING=false        # Version checking for concurrent updates (default: false)
```

### Python Configuration

```python
from mcp_memory_service.config import settings

# Check current config
print(settings.conflict.enabled)        # True
print(settings.conflict.auto_resolve)   # False (opt-in)

# Runtime configuration (if using Settings object)
from mcp_memory_service.config import Settings
custom_settings = Settings()
custom_settings.conflict.auto_resolve = True
```

## Manual Resolution API

### List Unresolved Conflicts

Get all memories with unresolved `CONTRADICTS` edges:

```bash
curl http://localhost:8001/api/manage/conflicts
```

**Response:**
```json
{
  "conflicts": [
    {
      "source_hash": "abc123...",
      "target_hash": "def456...",
      "confidence": 0.85,
      "signal_type": "negation",
      "created_at": 1234567890.0
    }
  ],
  "total": 1
}
```

**With full memory content:**
```bash
curl "http://localhost:8001/api/manage/conflicts?include_memories=true"
```

### Get Conflict Details

View both memories involved in a specific conflict:

```bash
curl http://localhost:8001/api/manage/conflicts/{source_hash}/{target_hash}
```

**Response:**
```json
{
  "source_hash": "abc123...",
  "target_hash": "def456...",
  "confidence": 0.85,
  "signal_type": "negation",
  "created_at": 1234567890.0,
  "source_memory": {
    "content": "The API supports pagination",
    "tags": ["api", "docs"],
    ...
  },
  "target_memory": {
    "content": "The API does not support pagination",
    "tags": ["api", "update"],
    ...
  }
}
```

### Resolve Conflict

Choose how to resolve the conflict:

```bash
curl -X POST http://localhost:8001/api/manage/conflicts/{source}/{target}/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "action": "keep_source",
    "merge_tags": true
  }'
```

**Resolution Actions:**

| Action | Behavior |
|--------|----------|
| `keep_source` | Keep source memory, delete target |
| `keep_target` | Keep target memory, delete source |
| `merge` | Merge tags into source, delete target |
| `dismiss` | Mark as resolved without changes |

**Response:**
```json
{
  "success": true,
  "action": "keep_source",
  "message": "Kept source memory abc123, deleted target def456",
  "kept_hash": "abc123...",
  "deleted_hash": "def456..."
}
```

## Understanding Conflicts

### Contradiction Types

**1. Negation Asymmetry**
- One memory negates what the other asserts
- Example: "API supports X" vs "API does not support X"
- Confidence based on topic overlap + negation strength

**2. Antonym Pairs**
- Memories use opposing terms for same concept
- Example: "enabled" vs "disabled", "success" vs "failure"
- Detected via predefined antonym vocabulary

**3. Temporal Supersession**
- New memory explicitly states something changed
- Phrases: "no longer", "switched from", "was X now Y"
- High confidence signal for outdated information

### Conflict Metadata

Each `CONTRADICTS` edge stores:
- `confidence`: 0.0-1.0 (how certain the contradiction is)
- `signal_type`: "negation", "antonym", "temporal"
- `similarity`: Cosine similarity between memories
- `created_at`: When contradiction was detected
- `resolved_at`: When resolved (null = unresolved)
- `resolved_by`: "auto" or "manual"
- `resolution_action`: How it was resolved

### Memory Conflict Fields

Memories track their conflict state:

```python
memory.conflict_status       # "detected", "auto_resolved", "manual_resolved", None
memory.conflict_version      # Increments on content changes (optimistic locking)
memory.conflict_history      # List of resolution events
```

## Examples

### Example 1: Auto-Resolve on Store

```python
from mcp_memory_service import MemoryService

# With auto-resolve enabled
result = await memory_service.store_memory(
    content="The API no longer supports pagination",
    tags=["api", "breaking-change"]
)

# Response includes auto-resolution info
print(result["interference"])  # Detected contradiction
print(result["auto_resolved"])  # {"count": 1, "strategy": "last_write_wins"}
```

### Example 2: Manual Review Workflow

```python
# 1. List conflicts
conflicts = await memory_service._graph.list_unresolved_conflicts(limit=10)

# 2. Review each conflict
for conflict in conflicts:
    source = await memory_service.storage.get_memory_by_hash(conflict["source"])
    target = await memory_service.storage.get_memory_by_hash(conflict["target"])

    print(f"Source: {source.content}")
    print(f"Target: {target.content}")
    print(f"Confidence: {conflict['confidence']}")

# 3. Resolve manually
await memory_service._graph.update_typed_edge_metadata(
    source_hash=conflict["source"],
    target_hash=conflict["target"],
    relation_type="CONTRADICTS",
    metadata={
        "resolved_at": time.time(),
        "resolved_by": "manual",
        "resolution_action": "keep_source"
    }
)
```

### Example 3: Consolidation with Auto-Resolve

```python
# Run periodic consolidation (includes conflict resolution if enabled)
stats = await memory_service.consolidate()

print(stats["conflicts_resolved"])  # Number of conflicts auto-resolved
print(stats["duplicates_merged"])   # Number of duplicates merged
```

## Best Practices

### When to Use Auto-Resolve

✅ **Good for:**
- High-volume automated systems
- Environments where latest info always wins
- Quick conflict cleanup during development
- Reducing manual overhead

❌ **Not good for:**
- Critical information that needs review
- Multi-user systems (concurrent edits by different users)
- Cases where both versions might be valid

### When to Use Manual Resolution

✅ **Good for:**
- Production systems with important data
- Multi-user collaboration
- Conflicts requiring human judgment
- Preserving conflict history for audit

### Conflict Prevention

1. **Use specific tags**: Better categorization reduces false positive contradictions
2. **Structured metadata**: Store facts in metadata, not just free-form content
3. **Temporal tags**: Tag memories with date/version to track changes over time
4. **Source tracking**: Use `client_hostname` to attribute memory sources

## Troubleshooting

### "Graph layer not enabled"

Conflict resolution requires FalkorDB. Enable it:

```bash
MCP_FALKORDB_ENABLED=true
MCP_FALKORDB_HOST=localhost
MCP_FALKORDB_PORT=6379
```

### Conflicts not being detected

Check interference detection is enabled:

```bash
MCP_INTERFERENCE_ENABLED=true
MCP_INTERFERENCE_SIMILARITY_THRESHOLD=0.7  # Lower = more sensitive
MCP_INTERFERENCE_MIN_CONFIDENCE=0.3        # Lower = more signals reported
```

### Auto-resolve not working

Verify configuration:

```python
from mcp_memory_service.config import settings

assert settings.conflict.enabled == True
assert settings.conflict.auto_resolve == True
```

### Conflicts re-appearing

If the same conflict keeps appearing:
- Check if both memories are being re-stored
- Verify resolution edges have `resolved_at` set
- Look at `conflict_history` to see resolution pattern

## Migration Guide

### Upgrading from Pre-Conflict Version

**No migration required** - changes are backward compatible:
- New fields have defaults (None, 0)
- Existing CONTRADICTS edges work without resolution metadata
- Auto-resolve disabled by default

**After upgrade:**
1. Existing CONTRADICTS edges are unresolved (resolved_at = null)
2. Run consolidation to batch-resolve if auto-resolve enabled
3. Use API to manually review and resolve existing conflicts

### Enabling Auto-Resolve for Existing System

```bash
# 1. Review existing conflicts first
curl http://localhost:8001/api/manage/conflicts?limit=100

# 2. Enable auto-resolve
export MCP_CONFLICT_AUTO_RESOLVE=true

# 3. Run consolidation to clean up backlog
curl -X POST http://localhost:8001/api/manage/system/consolidate

# 4. Monitor resolution stats
curl http://localhost:8001/api/health
```

## API Reference

See [CONFLICT_RESOLUTION_DESIGN.md](CONFLICT_RESOLUTION_DESIGN.md) for:
- Full architecture details
- Implementation phases
- Testing strategy
- Open questions and decisions
