# Memory Merge Conflicts - Design Document

## Overview

Extend MCP Memory Service with conflict detection and resolution capabilities. Support both automatic resolution (CRDT-style) and manual resolution via API.

## Problem Statement

Currently, the system:
- ✅ Detects contradictions (interference.py) and creates CONTRADICTS graph edges
- ✅ Merges duplicates automatically during consolidation
- ❌ Doesn't expose conflicts to users for manual resolution
- ❌ Doesn't handle concurrent updates to the same memory
- ❌ Doesn't provide configurable conflict resolution strategies

## Design Goals

1. **KISS**: Reuse existing infrastructure (graph edges, storage, interference detection)
2. **DRY**: Don't duplicate conflict detection logic
3. **YAGNI**: Build only what's needed - no complex CRDT protocols unless proven necessary
4. **Backward compatible**: No breaking changes to existing APIs

## Conflict Types

### 1. Contradictions (Already Detected)
- **Detection**: `utils/interference.py` - negation, antonym, temporal patterns
- **Storage**: FalkorDB CONTRADICTS edges
- **Current behavior**: Flagged but not resolved
- **New behavior**: Expose via API, allow resolution

### 2. Duplicates (Already Handled)
- **Detection**: Vector similarity in consolidation
- **Current behavior**: Auto-merged (older memory kept, tags merged)
- **New behavior**: Optionally queue for manual review before merging

### 3. Concurrent Updates (NEW)
- **Detection**: Optimistic locking - check updated_at hasn't changed
- **Storage**: Conflict metadata in Qdrant payload
- **New behavior**: Detect version conflicts, apply CRDT or queue for manual resolution

## Architecture

### Storage Schema

**No new tables/collections** - extend existing:

```python
# Memory model - add optional conflict tracking
@dataclass
class Memory:
    # ... existing fields ...
    conflict_status: str | None = None  # "detected", "auto_resolved", "manual_resolved", None
    conflict_version: int = 0  # Increment on each update (optimistic locking)
    conflict_history: list[dict] | None = None  # Track resolution history
```

**Graph edges** (FalkorDB):
```cypher
# Existing
(m1:Memory)-[r:CONTRADICTS {confidence: float, signal_type: str}]->(m2:Memory)

# Extended properties
r.resolved_at = timestamp  # When conflict was resolved
r.resolved_by = "auto" | "manual"  # How it was resolved
r.resolution_action = "keep_source" | "keep_target" | "merge" | "custom"
```

### Conflict Detection

#### At Store Time
```python
async def store_memory(content, tags, ...):
    # 1. Existing: Generate content_hash
    # 2. Existing: Detect contradictions via interference detection
    # 3. NEW: Check for concurrent updates (if updating existing memory)
    existing = await storage.get_memory_by_hash(content_hash)
    if existing and existing.conflict_version != expected_version:
        # Concurrent update detected - queue for resolution

    # 4. Store memory + create CONTRADICTS edges (existing)
    # 5. NEW: Add conflict metadata if conflicts detected
```

#### During Consolidation
```python
# Existing duplicate detection
# NEW: Optionally queue duplicates for manual review instead of auto-merge
```

### Resolution Strategies

#### 1. CRDT: Last-Write-Wins (Automatic)

Config: `MCP_MEMORY_AUTO_RESOLVE_CONFLICTS=true`

```python
async def auto_resolve_conflict(source_hash, target_hash):
    source = await storage.get_memory_by_hash(source_hash)
    target = await storage.get_memory_by_hash(target_hash)

    # Winner = most recent updated_at
    if source.updated_at > target.updated_at:
        winner, loser = source, target
    else:
        winner, loser = target, source

    # Merge tags (union)
    merged_tags = list(set(winner.tags) | set(loser.tags))

    # Keep winner's content, update tags
    winner.tags = merged_tags
    winner.conflict_history = [{
        "resolved_at": time.time(),
        "strategy": "last_write_wins",
        "merged_from": loser.content_hash,
        "merged_tags": True
    }]

    # Update winner, delete loser, mark edge as resolved
    await storage.update_memory_metadata(winner.content_hash, ...)
    await storage.delete(loser.content_hash)
    await graph.update_edge_metadata(source_hash, target_hash, "CONTRADICTS", {
        "resolved_at": time.time(),
        "resolved_by": "auto",
        "resolution_action": "last_write_wins"
    })
```

#### 2. Manual Resolution (API-driven)

```python
# API endpoints (web/api/manage.py)

@router.get("/conflicts")
async def list_conflicts(
    status: str = "unresolved",  # "unresolved", "all"
    limit: int = 10
):
    """List memories with unresolved CONTRADICTS edges."""
    # Query graph for CONTRADICTS edges where resolved_at is None
    # Return both source and target memory details

@router.get("/conflicts/{content_hash}")
async def get_conflict_details(content_hash: str):
    """Get full details of a conflict including both versions."""
    # Return memory + all CONTRADICTS edges + related memories

@router.post("/conflicts/{source_hash}/{target_hash}/resolve")
async def resolve_conflict(
    source_hash: str,
    target_hash: str,
    action: str,  # "keep_source", "keep_target", "merge", "custom"
    custom_content: str | None = None,
    custom_metadata: dict | None = None
):
    """Manually resolve a conflict."""
    # Apply resolution based on action
    # Update graph edge metadata (resolved_at, resolved_by, resolution_action)
    # Update or delete memories as needed
```

### Configuration

Add to `config.py`:

```python
class ConflictResolutionSettings(BaseSettings):
    """Conflict resolution configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable conflict detection and resolution"
    )

    auto_resolve: bool = Field(
        default=False,
        description="Automatically resolve conflicts using CRDT (last-write-wins)"
    )

    auto_resolve_duplicates: bool = Field(
        default=True,
        description="Automatically merge duplicates (existing behavior)"
    )

    duplicate_similarity_threshold: float = Field(
        default=0.95,
        description="Cosine similarity threshold for duplicate detection"
    )

    queue_duplicates_for_review: bool = Field(
        default=False,
        description="Queue duplicates for manual review instead of auto-merging"
    )

    optimistic_locking: bool = Field(
        default=False,
        description="Enable version checking for concurrent update detection"
    )

    model_config = SettingsConfigDict(env_prefix="MCP_MEMORY_CONFLICT_")

# Usage
class Settings:
    conflict: ConflictResolutionSettings = Field(default_factory=ConflictResolutionSettings)
```

## Implementation Plan

### Phase 1: Foundation (Current PR)
1. ✅ Add conflict_status, conflict_version, conflict_history to Memory model
2. ✅ Extend graph CONTRADICTS edges with resolution metadata
3. ✅ Add ConflictResolutionSettings to config
4. ✅ Create conflict query utilities (list conflicts from graph)

### Phase 2: CRDT Auto-Resolution
1. Implement auto_resolve_conflict() using last-write-wins
2. Add conflict resolution to consolidation cycle
3. Config flag to enable/disable auto-resolution
4. Tests for CRDT resolution

### Phase 3: Manual Resolution API
1. Add GET /conflicts endpoints
2. Add POST /conflicts/.../resolve endpoint
3. Resolution actions: keep_source, keep_target, merge
4. Tests for manual resolution

### Phase 4: Optimistic Locking (Optional Future)
1. Add version checking to update operations
2. Detect concurrent updates
3. Queue for resolution or auto-resolve

## Testing Strategy

```python
# tests/unit/test_conflict_resolution.py

def test_detect_contradiction_creates_conflict():
    """Interference detection creates CONTRADICTS edge."""

def test_auto_resolve_last_write_wins():
    """CRDT resolution keeps most recent memory."""

def test_auto_resolve_merges_tags():
    """CRDT resolution merges tags from both memories."""

def test_list_unresolved_conflicts():
    """API returns memories with unresolved CONTRADICTS edges."""

def test_manual_resolve_keep_source():
    """Manual resolution can keep source memory."""

def test_manual_resolve_merge():
    """Manual resolution can merge both memories."""

def test_concurrent_update_detection():
    """Optimistic locking detects version conflicts."""
```

## Migration Guide

**No schema migration required** - changes are additive:
- New fields in Memory have defaults (None, 0)
- Existing CONTRADICTS edges work without resolution metadata
- Config defaults preserve existing behavior

## Open Questions

1. **Duplicate review queue**: Store as CONTRADICTS edges or separate DUPLICATE edge type?
   - **Decision**: Use conflict_status metadata, not graph edges (duplicates are a storage concern, not knowledge graph concern)

2. **Version increment strategy**: On every update or only on content changes?
   - **Decision**: Content changes only (tags/metadata updates don't increment version)

3. **Conflict cascades**: What if resolving A→B creates new conflict B→C?
   - **Decision**: Detect and queue new conflicts, don't auto-resolve cascades

4. **Delete conflicts**: What if conflicting memory was deleted?
   - **Decision**: Mark edge as "resolved_obsolete", keep in graph for history

## References

- Existing interference detection: `src/mcp_memory_service/utils/interference.py`
- Existing duplicate merging: `src/mcp_memory_service/services/memory_service.py:_find_and_merge_duplicates`
- Graph client: `src/mcp_memory_service/graph/client.py`
- Storage base: `src/mcp_memory_service/storage/base.py`
