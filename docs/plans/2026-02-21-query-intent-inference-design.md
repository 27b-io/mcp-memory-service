# Query Intent Inference & Search Fan-Out

**Date**: 2026-02-21
**Issue**: #108
**Status**: Design approved

## Problem

Search treats user queries as literal embedding inputs. Multi-concept queries like
"dream cycle 3AM OpenClaw consolidation" produce a single averaged vector that matches
nothing well. Adding specificity makes results worse, not better.

The system is a **memory recommendation engine** — the user signals intent, and we
recommend relevant memories. We need to infer what they're looking for, not match strings.

### Evidence (post v11.10.0 cosine fix)

| Query | Top cosine | Results at 0.6 | Results at 0.5 |
|-------|-----------|-----------------|-----------------|
| `"dream"` | 0.61 | 1 | 1 |
| `"dream cycle 3AM OpenClaw consolidation"` | 0.56 | 0 | 3 |

More words = worse recall. This is backwards.

## Design

### Two-stage pluggable architecture

**Pre-search (NLP)**: Decompose query into concepts, fan out parallel vector searches.
Widens recall. Runs on every query. Fast, deterministic, no external calls.

**Post-search (LLM)**: Re-rank candidates by inferred intent. Improves precision.
Optional, off by default. Haiku via anthropic-lb when enabled.

### Pipeline

```
user query                                          wall clock
    │                                               ─────────
    ▼
┌─ STAGE 1: CONCEPT EXTRACTION (sync) ──────────┐
│  spaCy(query) → sub_queries[]                  │  ~15ms
│  en_core_web_sm: noun phrases, NER, compounds  │
│  Short-circuit: ≤1 concept → skip fan-out      │
└──────────┬─────────────────────────────────────┘
           │
           ▼
┌─ STAGE 2: BATCHED EMBEDDING (single forward) ─┐
│  model.encode([q1, q2, q3, original])          │  ~35ms
│  One forward pass, N vectors out               │
│  Avoids GIL serialisation of N sequential      │
│  encode() calls in thread pool                 │
└──────────┬─────────────────────────────────────┘
           │
           ▼
┌─ STAGE 3: PARALLEL FAN-OUT (asyncio.gather) ──┐
│  qdrant.search_by_vector(emb_1)  ──┐           │
│  qdrant.search_by_vector(emb_2)  ──┤           │
│  qdrant.search_by_vector(emb_3)  ──┤  parallel │  ~5ms
│  qdrant.search_by_vector(emb_og) ──┤  I/O      │
│  storage.search_by_tags(keywords)──┘           │
│                                                 │
│  Multi-set RRF merge (extend existing)          │
└──────────┬─────────────────────────────────────┘
           │
           ▼
┌─ STAGE 4: GRAPH INJECTION (parallel) ─────────┐
│  asyncio.gather(                                │
│    spreading_activation(top5_seeds),            │  ~10ms
│    hebbian_boosts(all_hashes),                  │
│  )                                              │
│  Fetch neighbor memories from Qdrant (batch)    │
│  Inject into candidate pool as new results      │
│  (not just boost existing — ADD new candidates) │
└──────────┬─────────────────────────────────────┘
           │
           ▼
┌─ STAGE 5: LLM RE-RANK (optional, async) ──────┐
│  Haiku via anthropic-lb                         │  ~300ms
│  Non-fatal: timeout/error → skip, use existing  │
└──────────┬─────────────────────────────────────┘
           │
           ▼
   existing boost pipeline (unchanged)              ~2ms
   (salience, spacing, context, temporal decay)
           │
           ▼
   min_similarity=0.5 filter → results

   TOTAL: ~65ms without LLM, ~365ms with LLM
   (vs current single-vector: ~40ms)
```

## Components

### 1. QueryIntentAnalyzer protocol

Pluggable interface for pre-search query decomposition.

```python
@dataclass
class QueryIntentResult:
    original_query: str
    sub_queries: list[str]        # For multi-vector fan-out
    concepts: list[str]           # Extracted concepts (for debug/logging)

class QueryIntentAnalyzer(Protocol):
    def analyze(self, query: str) -> QueryIntentResult: ...
```

### 2. SpaCyAnalyzer (default implementation)

- Model: `en_core_web_sm` (~15MB, loads in ~200ms, then cached)
- Extracts: noun phrases, named entities, compound nouns via dependency parse
- Generates sub-queries from concept clusters
- Lazy-loaded singleton (matches VADER analyzer pattern)
- Fallback: if spaCy unavailable, degrade to existing `extract_query_keywords()`
- Short-circuit: queries with ≤1 extracted concept skip fan-out entirely

### 3. LLMReranker protocol (optional post-search)

```python
class LLMReranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[dict]
    ) -> list[tuple[str, float]]:  # [(content_hash, relevance_score)]
        ...
```

### 4. AnthropicReranker

- Uses existing `MCP_SUMMARY_ANTHROPIC_BASE_URL` + API key (points at anthropic-lb)
- Model: `claude-haiku-4-5-20251001`
- Single call: query + candidate summaries → relevance scores
- Timeout: 2s hard limit
- Non-fatal: errors logged as warning, results pass through unmodified

### 5. New methods on QdrantStorage / MemoryStorage ABC

| Method | Purpose |
|--------|---------|
| `generate_embeddings_batch(texts: list[str]) -> list[list[float]]` | Single forward pass for N query vectors |
| `search_by_vector(embedding: list[float], n_results, min_similarity, **filters) -> list[MemoryQueryResult]` | Search with pre-computed vector, no embedding step |
| `get_memories_batch(content_hashes: list[str]) -> list[Memory]` | Batch fetch by hash for graph-injected neighbors |

### 6. Multi-set RRF merge

Extend `combine_results_rrf()` to accept N result lists instead of exactly 2.
Each list gets its own RRF ranking. Results appearing in multiple lists get
combined RRF scores (higher rank = more relevant across concepts).

```python
def combine_results_rrf_multi(
    result_sets: list[list[MemoryQueryResult]],
    weights: list[float],    # per-set weight (original query weighted higher)
    tag_matches: list[Memory],
    k: int = 60,
) -> list[tuple[Memory, float, dict]]:
```

### 7. Graph injection (not just boost)

Current: spreading activation finds neighbors, boosts scores of results already
in the set. Misses memories that are associatively connected but semantically
distant from the query.

New: after spreading activation, fetch the top-N activated neighbor hashes that
are NOT already in the result set. Batch-fetch their Memory objects from Qdrant.
Inject them into the candidate pool with their activation score as initial
similarity score. They then flow through the normal boost pipeline.

```python
# After spreading activation
neighbor_hashes = [
    h for h, score in graph_activation.items()
    if h not in result_hash_set and score >= min_activation
][:inject_limit]

if neighbor_hashes:
    neighbors = await storage.get_memories_batch(neighbor_hashes)
    for memory in neighbors:
        activation = graph_activation[memory.content_hash]
        combined.append((memory, activation, {"source": "graph_injection"}))
```

## Configuration

```python
class QueryIntentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_INTENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Master switch
    enabled: bool = True

    # NLP concept extraction
    spacy_model: str = "en_core_web_sm"
    max_sub_queries: int = 4
    min_query_tokens: int = 3  # Skip fan-out for short queries

    # Graph injection
    graph_inject: bool = True
    graph_inject_limit: int = 10
    graph_inject_min_activation: float = 0.05

    # LLM re-ranking (off by default)
    llm_rerank: bool = False
    llm_provider: str = "anthropic"  # Only option for now
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_ms: int = 2000
```

Default `min_similarity` threshold: **0.5** (changed from 0.6).

## Dependencies

- `spacy` — new dependency
- `en_core_web_sm` — spaCy model, downloaded at build time
- `anthropic` — already a dependency (summariser)

Dockerfile addition:
```dockerfile
RUN uv pip install spacy && python -m spacy download en_core_web_sm
```

## Graceful degradation

Every stage is non-fatal and independently toggleable:

| Component | Missing/failed | Behaviour |
|-----------|---------------|-----------|
| spaCy not installed | Fall back to `extract_query_keywords()` (existing) |
| spaCy model not found | Fall back to `extract_query_keywords()` |
| ≤1 concept extracted | Skip fan-out, use single-vector path |
| FalkorDB disabled | Skip graph injection + Hebbian boost |
| Graph query fails | Warning log, skip injection |
| LLM rerank disabled | Skip (default) |
| LLM rerank timeout | Warning log, pass results through |
| Anthropic API error | Warning log, pass results through |

## What this does NOT do

- Query caching — YAGNI, measure first
- Learned intent models — no training data yet
- Async spaCy — spaCy is sync-only, but fast enough at ~15ms
- SQLite-vec fan-out — ABC methods added but SQLite-vec implementation deferred
  (it's the dev/single-node backend; Qdrant is production)

## Metrics

Add to existing `SearchLog`:
- `intent_enabled: bool`
- `concepts_extracted: list[str]`
- `sub_queries_count: int`
- `graph_injected_count: int`
- `llm_reranked: bool`
- `stage_timings: dict[str, float]`  (per-stage ms for profiling)
