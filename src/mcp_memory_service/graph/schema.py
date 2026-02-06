"""
Graph schema for cognitive memory knowledge graph.

Defines the Cypher schema for FalkorDB: node labels, relationship types,
indices, and constraints. Schema is applied idempotently on startup.

Node Labels:
    :Memory  - Represents a stored memory (keyed by content_hash)

Relationship Types:
    :HEBBIAN     - Co-access association edge (implicit, high-frequency).
                   Weight increases on co-retrieval, decays over time.
    :RELATES_TO  - Generic semantic relationship (explicit, user-created).
    :PRECEDES    - Temporal/causal ordering: source precedes target.
    :CONTRADICTS - Conflicting information between two memories.

Indices:
    Memory(content_hash) - Unique lookup for memory nodes
    Memory(created_at)   - Temporal queries
"""

# Valid typed relationship types (whitelist for Cypher injection safety).
# FalkorDB doesn't support parameterized relationship types, so we validate
# against this set before string-formatting into queries.
RELATION_TYPES: frozenset[str] = frozenset({"RELATES_TO", "PRECEDES", "CONTRADICTS"})

# Cypher statements executed idempotently on graph initialization.
# FalkorDB supports CREATE INDEX IF NOT EXISTS syntax.
SCHEMA_STATEMENTS: list[str] = [
    # Exact-match index on content_hash for O(1) node lookup
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.content_hash)",
    # Range index on created_at for temporal traversals
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.created_at)",
]
