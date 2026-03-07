# Pydantic Models Migration — Design

**Date:** 2026-03-01
**Branch:** `feat/pydantic-models`
**Approach:** Big Bang — all layers converted in a single coordinated change

## Problem

Validation is scattered across the codebase in 6+ different patterns:
- Manual `isinstance()` checks (25+ locations)
- `dict.get()` with defaults (60+ locations)
- Range clamping (`max(1, min(k, 100))`) repeated 8+ times
- TypedDicts used as runtime type hints with no enforcement
- Memory dataclass with 200+ lines of manual `__post_init__` validation
- Service layer: 25 methods returning `dict[str, Any]`

Config and Web API layers already use Pydantic. The rest doesn't.

## Scope

| Layer | Action |
|-------|--------|
| `@src/mcp_memory_service/models/validators.py` | NEW — shared reusable types |
| `@src/mcp_memory_service/models/responses.py` | NEW — service response models |
| `@src/mcp_memory_service/models/mcp_inputs.py` | NEW — MCP tool input models |
| `@src/mcp_memory_service/models/memory.py` | REWRITE — dataclass → BaseModel |
| `@src/mcp_memory_service/hooks.py` | MODIFY — dataclass → BaseModel |
| `@src/mcp_memory_service/services/memory_service.py` | MODIFY — return response models |
| `@src/mcp_memory_service/mcp_server.py` | MODIFY — use input models |
| `@src/mcp_memory_service/storage/*.py` | MODIFY — `from_dict()` → `model_validate()` |
| `@tests/**` | MODIFY — update assertions |

**Out of scope:** Config (already Pydantic), Web API models (already Pydantic), storage ABC, graph client internals.

## Design

### 1. Shared Validators (`@src/mcp_memory_service/models/validators.py`)

Reusable `Annotated` types with `BeforeValidator`:

- `Tags` — normalizes `str | list[str] | None` → `list[str]`
- `UnitFloat` — `float` clamped to [0.0, 1.0]
- `ClampedInt` — `int` ≥ 0
- `ContentHash` — non-empty string
- `SearchMode` — Literal enum
- `MemoryType` — Literal enum
- `RelationType` — Literal enum
- `OutputFormat` — Literal enum

### 2. Memory Model (BaseModel)

Single Pydantic model replacing the dataclass. `@model_validator` replaces `__post_init__`.
`model_validate()` replaces `from_dict()`. `model_dump()` replaces `to_dict()`.

### 3. Service Response Models

Base `ServiceResult(success, error)` with operation-specific subclasses:
`StoreResult`, `DeleteResult`, `HealthResult`, `SearchResult`, `RelationResult`,
`FindDuplicatesResult`, `MergeDuplicatesResult`, `SupersedeResult`, `ContradictionsResult`.

### 4. MCP Input Models

Per-tool Pydantic models with declarative constraints:
`StoreMemoryParams`, `SearchParams`, `DeleteMemoryParams`, `RelationParams`,
`FindDuplicatesParams`, `SupersedeParams`, `MergeDuplicatesParams`.

Range clamping, mode validation, and required-field checks move from handler code to field definitions.

### 5. Hook Events

Swap `@dataclass` for `BaseModel` on `CreateEvent`, `DeleteEvent`, `UpdateEvent`, `RetrieveEvent`.

## Decisions

- **Single Memory model** (not input/storage/output split) — fields are identical, only population differs
- **`Annotated` types over `@field_validator`** — reusable across models without duplication
- **Service returns models, not dicts** — callers get type-safe attribute access
- **MCP handlers validate via Pydantic, then delegate** — thin handler pattern
