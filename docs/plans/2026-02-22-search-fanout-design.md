# Search Fan-Out: Semantic Tag Matching + Summary Vector Search

## Problem

Similarity scores for direct hits are too low (~0.47 for "proton bridge" against a "Proton Bridge IMAP Auth Fix" memory). Two root causes:

1. **Tag matching gap**: `extract_query_keywords` does exact token matching against tags. "proton bridge" doesn't match the tag `proton-bridge` (fixed in PR #118 for hyphenated compounds), but broader semantic matches (synonyms, related concepts) are still missed entirely.

2. **Document dilution**: Content embeddings represent the full memory (~200+ words). Short queries match poorly against long, multi-topic documents. Summaries (~50 tokens) are stored but never searched — they'd produce much higher cosine scores for topical queries.

## Solution

Two independent features feeding into the existing `combine_results_rrf_multi` pipeline:

```
Query "proton bridge"
  |
  +--- Content Vector Search (existing, weight=1.0)
  |      Qdrant search on "content" named vector
  |
  +--- Summary Vector Search (new, weight=0.8, configurable)
  |      Qdrant search on "summary" named vector
  |
  +--- Semantic Tag Matching (new, weight=0.5, configurable)
  |      Query embedding vs CacheKit-cached tag embeddings (k-NN)
  |      -> fan out matched tags -> search_by_tag
  |
  +--- Exact Tag Matching (existing, weight=0.3)
         extract_query_keywords -> search_by_tag

All result sets -> combine_results_rrf_multi -> boosts -> pagination -> response
```

Each feature is independently enabled/disabled with its own config section.

## Feature 1: Semantic Tag Matching

### Architecture

1. On query, reuse the existing query embedding
2. Load tag embeddings from CacheKit cache (Redis-backed, 1hr TTL)
3. Cache miss: `get_all_tags()` -> `generate_embeddings_batch(tags)` -> store in Redis
4. k-NN cosine similarity: query embedding vs tag embeddings (numpy dot product, <1000 tags x 1024 dims = ~4MB, trivial in-process computation)
5. Select tags above threshold (default 0.5 — tag strings are short, scores will be higher than document matches)
6. Union with exact-matched tags from `extract_query_keywords`
7. Fan out: `search_by_tag(semantic_tags)` -> feed into RRF as additional result set

### CacheKit Integration

CacheKit (`cachekit-io/cachekit-py`) provides Redis-backed caching with:
- L1 in-memory (~50ns) + L2 Redis (~2-7ms) two-tier cache
- Distributed locking (multi-pod safety)
- Circuit breaker (graceful Redis failure)
- Graceful degradation: falls back to L1-only if Redis unavailable

Replaces the manual `_tag_cache` tuple in `MemoryService` (lines 97, 879-890).

### Config

```python
class SemanticTagSettings(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.5
    max_tags: int = 10        # cap fan-out breadth
    rrf_weight: float = 0.5   # weight in RRF fusion
    cache_ttl: int = 3600     # tag embedding cache TTL (seconds)
```

Env vars: `MCP_SEMANTIC_TAG_ENABLED`, `MCP_SEMANTIC_TAG_THRESHOLD`, etc.

### Cache Invalidation

When `store_memory` adds a tag not in the cached tag set, invalidate the tag embeddings cache key. All pods pick up the change on next query.

## Feature 2: Summary Vector Search

### Qdrant Named Vector Migration

**Current**: Single unnamed vector collection `memories_arctic1024` (1024-dim Cosine).

**Target**: Named vectors collection with:
- `"content"` — 1024-dim Cosine (existing embeddings, migrated)
- `"summary"` — 1024-dim Cosine (new, backfilled during migration)

Qdrant cannot add named vectors to an existing unnamed-vector collection. Full migration required.

### Migration Strategy

CLI command `uv run memory migrate-vectors`:

1. Create new collection `memories_arctic1024_v2` with named vectors config
2. Scroll all points from old collection in batches of 100
3. For each batch:
   - Extract existing content vector
   - Generate summary embedding via `generate_embeddings_batch([summary for each memory])`
   - Memories without summaries get them auto-generated first (Gemini Flash)
   - Upsert to v2 with `{"content": existing_vector, "summary": summary_embedding}`
4. Verify point counts match
5. Update config to point at v2
6. Old collection preserved until explicit drop

CacheKit distributed lock prevents multiple pods from running simultaneously.

### Search Flow

1. Reuse existing query embedding
2. Search Qdrant on `"summary"` named vector
3. Return as additional result set into `combine_results_rrf_multi`

### On Store

When a new memory is stored, embed the summary alongside the content. Both vectors upserted to the same point as named vectors.

### Config

```python
class SummarySearchSettings(BaseModel):
    enabled: bool = True
    rrf_weight: float = 0.8
```

## Background Task Infrastructure

asyncio-based, in-process. Reuses the Hebbian write queue pattern (LPUSH/BRPOP from `graph/queue.py`) for coordination.

- **One-off tasks** (migration, backfill): `asyncio.create_task()` with progress tracking
- **Periodic tasks** (future: memory tending): `while True: await asyncio.sleep(interval)` loop
- **Multi-pod safety**: CacheKit distributed lock around background tasks

No Celery — YAGNI. asyncio is sufficient for current scale. arq is a clean step up if persistence is needed later.

## Delivery

| PR | Feature | Depends On |
|----|---------|------------|
| 1 | Semantic tag matching + CacheKit integration | None |
| 2 | Named vector migration + summary vector search | PR 1 (CacheKit) |
| 3 | Backfill + async task infrastructure | PR 2 (named vectors) |

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tag embedding storage | CacheKit (Redis-backed) | Multi-pod cache sharing, distributed locking, graceful degradation |
| Summary vector storage | Qdrant named vectors | Same point, same collection, native multi-vector search |
| Migration strategy | Full migration to named vectors | Clean architecture, no schema split |
| Background tasks | asyncio in-process | YAGNI — no Celery. Hebbian queue pattern exists. |
| RRF weights | Configurable per-signal | Different deployments need different tuning |
| CacheKit replaces _tag_cache | Yes | Multi-pod consistency, proper invalidation |
