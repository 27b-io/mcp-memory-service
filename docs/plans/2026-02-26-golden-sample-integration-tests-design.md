# Golden-Sample Integration Tests

## Problem

860 tests exist but none exercise the full search pipeline with all features enabled.
Every feature (Hebbian, semantic tags, intent inference, temporal decay, salience,
spaced repetition, spreading activation) is tested in isolation. The `test_fanout_e2e.py`
explicitly disables all boost stages. We cannot prove these features compose correctly.

## Design

### Test Data: Project Gutenberg Golden Samples

41 public domain passages from diverse sources (literature, philosophy, science,
political theory, history). Each passage is 200-800 characters with clear thematic
signal. Passages are grouped into **thematic clusters** where 2-3 passages share a
theme across different sources, enabling cross-source semantic retrieval tests.

Fixture file: `@tests/integration/fixtures/golden_samples.json`

Structure — a top-level JSON array of memory objects:
```json
[
  {
    "id": "gutenberg_1232_1",
    "content": "passage text",
    "source": "The Prince by Machiavelli",
    "themes": ["leadership", "power", "politics"],
    "tags": ["philosophy", "politics", "leadership"],
    "memory_type": "reference"
  },
  ...
]
```

### Test File: `@tests/integration/test_search_pipeline.py`

Single file, 7 test classes, ~30 test methods. All use real Qdrant `:memory:` storage
with real embeddings. No mocks for storage or embeddings.

### Test Classes

#### 1. `TestBaselineSemantic` — Pure vector search, all boosts disabled
- Store all golden samples
- Query with thematic queries ("nature of justice", "art of war", "scientific method")
- Assert: correct thematic cluster members appear in top-K
- Assert: cosine similarity scores are in expected range (0.3-0.8)
- This is the **control group** — all other tests compare against this baseline

#### 2. `TestTagMatching` — Exact + semantic tag search
- Query with tag-matching keywords
- Assert: exact tag matches rank higher than pure semantic matches
- Assert: semantic tag matching finds related tags (e.g., "governance" finds "politics")
- Compare RRF-fused results against baseline — tag signal should improve relevant rankings

#### 3. `TestTemporalDecay` — Time-based relevance degradation
- Store memories with controlled `created_at` timestamps (1 day, 7 days, 30 days, 90 days ago)
- Enable temporal decay (`temporal_decay_lambda=0.01`)
- Assert: recent memories score higher than old ones for equal semantic similarity
- Assert: decay factor is mathematically correct: `base + exp(-lambda * days) * (1 - base)`

#### 4. `TestSalienceBoost` — Emotional/importance scoring
- Store memories with varying `metadata.importance` (0.0, 0.5, 1.0)
- Enable salience boost
- Assert: high-importance memories rank above equivalent low-importance ones
- Assert: boost magnitude matches config (`boost_weight * salience_score`)

#### 5. `TestQueryIntentFanout` — Multi-concept query decomposition
- Query with multi-concept queries ("leadership philosophy and military strategy")
- Assert: results span multiple thematic clusters (not just the best single-vector match)
- Assert: memories matching multiple concepts rank higher (RRF multi-list weighting)

#### 6. `TestFullPipeline` — All features enabled simultaneously
- Default production config (all features on, FalkorDB off since no graph in test)
- Run the same queries as TestBaselineSemantic
- Assert: results are at least as good (Hit Rate >= baseline)
- Assert: no score exceeds 1.0 (cap works)
- Assert: min_similarity filter works (no results below threshold)
- Assert: pagination works correctly (page 1 + page 2 cover full result set, no overlaps)

#### 7. `TestPerformance` — Latency benchmarks
- Store all golden samples
- Time 100 sequential queries
- Assert: p50 < 100ms, p95 < 300ms, p99 < 500ms (in-memory Qdrant)
- Assert: no query takes > 1s
- Store timing distribution for regression tracking

### Fixture Architecture

```python
@pytest.fixture(scope="module")
async def seeded_storage():
    """Qdrant :memory: storage seeded with all golden samples."""
    storage = QdrantStorage(
        storage_path=":memory:",
        embedding_model="all-MiniLM-L6-v2",  # fast, deterministic
        collection_name=f"golden_{uuid.uuid4().hex[:8]}",
    )
    await storage.initialize()
    # Seed from golden_samples.json
    ...
    yield storage
    await storage.close()

@pytest.fixture(scope="module")
async def seeded_service(seeded_storage):
    """MemoryService wrapping seeded storage with all defaults."""
    return MemoryService(seeded_storage)
```

Module-scoped to avoid re-seeding per test (embedding generation is expensive).

### Assertion Strategy

**Ranking assertions** use relative ordering, not absolute positions:
```python
def assert_ranks_above(results, higher_hash, lower_hash):
    """Assert higher_hash appears before lower_hash in results."""
```

**Score assertions** use tolerance bands, not exact values:
```python
def assert_score_in_range(result, min_score, max_score):
```

**Hit assertions** check set membership in top-K:
```python
def assert_any_hit(results, expected_hashes, k=10):
    """At least one expected hash in top-K results."""
```

### What This Does NOT Test

- FalkorDB graph features (Hebbian, spreading activation) — requires graph database
- Encoding context boost — requires matching encoding contexts at store and retrieve time
- Spaced repetition — requires simulated access patterns over time
- Superseded memory filter — requires supersession relationships

These are documented as out-of-scope for Phase 1. Phase 2 can add a FalkorDB
integration test with `testcontainers` if needed.

## Files

| File | Purpose |
|------|---------|
| `@tests/integration/fixtures/golden_samples.json` | Gutenberg passages |
| `@tests/integration/test_search_pipeline.py` | Integration test suite |
