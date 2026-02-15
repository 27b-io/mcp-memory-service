#!/usr/bin/env python3
"""
Graph hydration script for Phase 3 of mm-fyr (gh-54).

Backfills FalkorDB with Memory nodes and edges for existing memories in Qdrant.

Usage:
    # Backfill nodes only
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/hydrate_graph.py --nodes-only

    # Simulate Hebbian edges (requires nodes to exist)
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/hydrate_graph.py --hebbian

    # Run contradiction scanning
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/hydrate_graph.py --contradictions

    # Do all three phases
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/hydrate_graph.py --all

    # Dry run mode
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/hydrate_graph.py --all --dry-run
"""

import argparse
import asyncio
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from mcp_memory_service.config import Settings
from mcp_memory_service.graph.client import GraphClient
from mcp_memory_service.utils.interference import detect_contradiction_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def backfill_nodes(
    qdrant_client: QdrantClient,
    graph_client: GraphClient,
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict:
    """
    Backfill Memory nodes in FalkorDB for all memories in Qdrant.

    Args:
        qdrant_client: Qdrant client
        graph_client: Graph client
        dry_run: Preview without making changes
        batch_size: Memories to process per batch

    Returns:
        Stats dict with node counts
    """
    logger.info("=" * 60)
    logger.info("Phase 3.1: Node Backfill")
    logger.info("=" * 60)

    stats = {"total_memories": 0, "nodes_created": 0, "errors": 0}

    # Count total memories
    collection_info = qdrant_client.get_collection("memories")
    total_count = collection_info.points_count
    stats["total_memories"] = total_count
    logger.info(f"Found {total_count:,} memories in Qdrant")

    if dry_run:
        logger.info(f"\n[DRY RUN] Would create {total_count:,} Memory nodes in FalkorDB")
        logger.info("No changes made (dry run)")
        return stats

    # Initialize graph
    await graph_client.initialize()

    # Get current node count
    graph_stats = await graph_client.get_graph_stats()
    existing_nodes = graph_stats.get("node_count", 0)
    logger.info(f"Current FalkorDB nodes: {existing_nodes:,}")

    # Scroll through all memories and create nodes
    offset = None
    batch_num = 0

    while True:
        batch_num += 1
        batch_start = time.time()

        points, next_offset = qdrant_client.scroll(
            collection_name="memories",
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        # Create nodes for batch
        for point in points:
            try:
                payload = point.payload
                content_hash = payload.get("content_hash")
                created_at = payload.get("created_at")

                if not content_hash or created_at is None:
                    logger.warning(f"Skipping point {point.id}: missing content_hash or created_at")
                    stats["errors"] += 1
                    continue

                # MERGE is idempotent - safe to call multiple times
                await graph_client.ensure_memory_node(content_hash, created_at)
                stats["nodes_created"] += 1

            except Exception as e:
                logger.error(f"Failed to create node for point {point.id}: {e}")
                stats["errors"] += 1

        batch_time = time.time() - batch_start
        logger.info(
            f"  Batch {batch_num}: {len(points)} memories → {stats['nodes_created']:,}/{total_count:,} nodes "
            f"({stats['nodes_created']/total_count*100:.1f}%) [{batch_time:.1f}s]"
        )

        if next_offset is None:
            break
        offset = next_offset

    # Verify final count
    final_stats = await graph_client.get_graph_stats()
    final_nodes = final_stats.get("node_count", 0)

    logger.info(f"\n✓ Node backfill complete:")
    logger.info(f"  Memories in Qdrant: {total_count:,}")
    logger.info(f"  Nodes in FalkorDB: {final_nodes:,}")
    logger.info(f"  Nodes created: {stats['nodes_created']:,}")
    logger.info(f"  Errors: {stats['errors']}")

    return stats


async def simulate_hebbian_edges(
    qdrant_client: QdrantClient,
    graph_client: GraphClient,
    model: SentenceTransformer,
    dry_run: bool = False,
    num_queries: int = 100,
    results_per_query: int = 10,
) -> dict:
    """
    Simulate memory retrievals to build Hebbian co-access edges.

    Generates synthetic queries from random memory content, performs vector search,
    and creates HEBBIAN edges between co-retrieved memories.

    Args:
        qdrant_client: Qdrant client
        graph_client: Graph client
        model: Embedding model for queries
        dry_run: Preview without making changes
        num_queries: Number of simulated queries to run
        results_per_query: Top-k results per query

    Returns:
        Stats dict with edge counts
    """
    logger.info("=" * 60)
    logger.info("Phase 3.2: Hebbian Edge Simulation")
    logger.info("=" * 60)

    stats = {"queries": 0, "edges_created": 0, "co_accesses": 0}

    if dry_run:
        logger.info(f"\n[DRY RUN] Would simulate {num_queries} queries × {results_per_query} results")
        logger.info(f"Expected co-access pairs: ~{num_queries * (results_per_query * (results_per_query - 1) // 2):,}")
        logger.info("No changes made (dry run)")
        return stats

    await graph_client.initialize()

    # Get sample memories to generate queries from
    logger.info(f"Fetching {num_queries} sample memories for query generation...")
    sample_points, _ = qdrant_client.scroll(
        collection_name="memories",
        limit=num_queries,
        with_payload=True,
        with_vectors=False,
    )

    if not sample_points:
        logger.error("No memories found in Qdrant")
        return stats

    logger.info(f"Simulating {num_queries} memory retrievals...")

    for i, point in enumerate(sample_points, 1):
        try:
            # Extract query text from memory content (first 200 chars)
            content = point.payload.get("content", "")
            if not content:
                continue

            query_text = content[:200]
            query_vector = model.encode(query_text).tolist()

            # Perform vector search
            results = qdrant_client.query_points(
                collection_name="memories",
                query=query_vector,
                limit=results_per_query,
            )

            if not results.points:
                continue

            # Extract content hashes from results
            result_hashes = []
            for r in results.points:
                h = r.payload.get("content_hash")
                if h:
                    result_hashes.append(h)

            if len(result_hashes) < 2:
                continue

            # Create Hebbian edges between all pairs in result set
            # In production, this goes through the write queue, but for bulk hydration
            # we create edges directly
            for j in range(len(result_hashes)):
                for k in range(j + 1, len(result_hashes)):
                    source = result_hashes[j]
                    target = result_hashes[k]

                    # Create bidirectional edges
                    # Check if edge exists first, update weight if so
                    existing = await graph_client.get_edge(source, target)

                    if existing:
                        # Edge exists - would normally strengthen it, but for simulation just count
                        pass
                    else:
                        # Create new edge with initial weight
                        # Note: This is a simplified version. Production uses HebbianWriteQueue
                        # For now, we'll use create_typed_edge as a proxy
                        # TODO: Add a bulk Hebbian edge creation method to GraphClient
                        pass

                    stats["co_accesses"] += 1

            stats["queries"] += 1

            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{num_queries} queries, {stats['co_accesses']:,} co-accesses")

        except Exception as e:
            logger.error(f"Error simulating query {i}: {e}")

    logger.info(f"\n✓ Hebbian simulation complete:")
    logger.info(f"  Queries simulated: {stats['queries']}")
    logger.info(f"  Co-access pairs: {stats['co_accesses']:,}")
    logger.info(f"\nNote: Full Hebbian edge creation requires HebbianWriteQueue integration")
    logger.info("This simulation identified co-access patterns for future implementation")

    return stats


async def scan_contradictions(
    qdrant_client: QdrantClient,
    graph_client: GraphClient,
    model: SentenceTransformer,
    config: Settings,
    dry_run: bool = False,
    batch_size: int = 50,
) -> dict:
    """
    Scan for contradictions between memories and create CONTRADICTS edges.

    Args:
        qdrant_client: Qdrant client
        graph_client: Graph client
        model: Embedding model for similarity
        config: Settings for interference detection
        dry_run: Preview without making changes
        batch_size: Memories to process per batch

    Returns:
        Stats dict with contradiction counts
    """
    logger.info("=" * 60)
    logger.info("Phase 3.3: Contradiction Scanning")
    logger.info("=" * 60)

    stats = {"memories_scanned": 0, "contradictions_found": 0, "edges_created": 0}

    if dry_run:
        logger.info("\n[DRY RUN] Would scan all memories for contradictions")
        logger.info("No changes made (dry run)")
        return stats

    await graph_client.initialize()

    # Get total count
    collection_info = qdrant_client.get_collection("memories")
    total_count = collection_info.points_count
    logger.info(f"Scanning {total_count:,} memories for contradictions...")

    offset = None
    batch_num = 0

    while True:
        batch_num += 1

        points, next_offset = qdrant_client.scroll(
            collection_name="memories",
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        for point in points:
            try:
                content = point.payload.get("content", "")
                content_hash = point.payload.get("content_hash")

                if not content or not content_hash:
                    continue

                # Use interference detection to find contradictions
                # Note: This is simplified - production interference detection is more sophisticated
                query_vector = point.vector if hasattr(point, 'vector') and point.vector else model.encode(content).tolist()

                # Search for similar memories
                similar = qdrant_client.query_points(
                    collection_name="memories",
                    query=query_vector,
                    limit=config.interference.max_candidates + 1,  # +1 to exclude self
                )

                for candidate in similar.points:
                    if candidate.id == point.id:
                        continue

                    # Check similarity threshold
                    if candidate.score < config.interference.similarity_threshold:
                        continue

                    candidate_content = candidate.payload.get("content", "")
                    candidate_hash = candidate.payload.get("content_hash")

                    if not candidate_hash:
                        continue

                    # Detect contradictions
                    contradictions = detect_contradiction_signals(
                        new_content=content,
                        existing_content=candidate_content,
                        existing_hash=candidate_hash,
                        similarity=candidate.score,
                        min_confidence=config.interference.min_confidence,
                    )

                    if contradictions and len(contradictions) > 0:
                        # Contradictions found, confidence already checked
                        if True:  # Simplified - confidence already checked in detect_contradiction_signals
                            # Create CONTRADICTS edge
                            created = await graph_client.create_typed_edge(
                                source_hash=content_hash,
                                target_hash=candidate_hash,
                                relation_type="CONTRADICTS",
                            )

                            if created:
                                stats["contradictions_found"] += 1
                                stats["edges_created"] += 1

                stats["memories_scanned"] += 1

            except Exception as e:
                logger.error(f"Error scanning memory {point.id}: {e}")

        logger.info(f"  Batch {batch_num}: {stats['memories_scanned']}/{total_count} scanned, {stats['contradictions_found']} contradictions")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"\n✓ Contradiction scanning complete:")
    logger.info(f"  Memories scanned: {stats['memories_scanned']:,}")
    logger.info(f"  Contradictions found: {stats['contradictions_found']}")
    logger.info(f"  CONTRADICTS edges created: {stats['edges_created']}")

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Hydrate FalkorDB graph with memory nodes and edges")
    parser.add_argument("--nodes-only", action="store_true", help="Only backfill Memory nodes")
    parser.add_argument("--hebbian", action="store_true", help="Only simulate Hebbian edges")
    parser.add_argument("--contradictions", action="store_true", help="Only scan for contradictions")
    parser.add_argument("--all", action="store_true", help="Run all three phases")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--qdrant-url", help="Qdrant URL (overrides MCP_QDRANT_URL)")
    parser.add_argument("--falkordb-host", default="localhost", help="FalkorDB host")
    parser.add_argument("--falkordb-port", type=int, default=6379, help="FalkorDB port")
    args = parser.parse_args()

    # Validate mode selection
    if not (args.nodes_only or args.hebbian or args.contradictions or args.all):
        parser.error("Must specify at least one mode: --nodes-only, --hebbian, --contradictions, or --all")

    # Load config
    config = Settings()

    # Connect to Qdrant
    qdrant_url = args.qdrant_url or os.environ.get("MCP_QDRANT_URL")
    if not qdrant_url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --qdrant-url")
        sys.exit(1)

    qdrant_client = QdrantClient(url=qdrant_url, timeout=120)

    # Connect to FalkorDB
    graph_client = GraphClient(
        host=args.falkordb_host,
        port=args.falkordb_port,
        graph_name="memory_graph",
    )

    try:
        start_time = time.time()

        # Phase 1: Node backfill
        if args.nodes_only or args.all:
            node_stats = await backfill_nodes(qdrant_client, graph_client, dry_run=args.dry_run)
            logger.info("")

        # Phase 2: Hebbian edges
        if args.hebbian or args.all:
            model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            hebbian_stats = await simulate_hebbian_edges(
                qdrant_client, graph_client, model, dry_run=args.dry_run
            )
            logger.info("")

        # Phase 3: Contradictions
        if args.contradictions or args.all:
            model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            contradiction_stats = await scan_contradictions(
                qdrant_client, graph_client, model, config, dry_run=args.dry_run
            )

        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info(f"Graph hydration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("=" * 60)

        # Final stats
        if not args.dry_run:
            final_stats = await graph_client.get_graph_stats()
            logger.info(f"\nFinal graph state:")
            logger.info(f"  Nodes: {final_stats.get('node_count', 0):,}")
            logger.info(f"  Hebbian edges: {final_stats.get('hebbian_edge_count', 0):,}")
            logger.info(f"  CONTRADICTS edges: {final_stats.get('typed_edge_counts', {}).get('contradicts', 0):,}")
            logger.info(f"  Total edges: {final_stats.get('edge_count', 0):,}")

    finally:
        await graph_client.close()
        qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
