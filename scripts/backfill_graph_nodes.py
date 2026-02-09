#!/usr/bin/env python3
"""Backfill graph nodes for existing memories in Qdrant.

Scans all memories in the Qdrant collection and creates corresponding
:Memory nodes in FalkorDB. Uses MERGE (idempotent) so safe to re-run.

This seeds the knowledge graph with nodes for memories that were created
before the graph layer was added (v11.x). Once seeded, Hebbian edges
will form naturally as memories are co-retrieved.

Usage:
    MCP_QDRANT_URL=http://... MCP_FALKORDB_HOST=... \
        uv run --python 3.12 python scripts/backfill_graph_nodes.py [--dry-run] [--batch-size N]

    # With defaults from environment (k8s port-forward or direct):
    MCP_QDRANT_URL=http://localhost:6333 \
    MCP_FALKORDB_HOST=localhost \
    MCP_FALKORDB_PORT=6379 \
        uv run --python 3.12 python scripts/backfill_graph_nodes.py
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"
GRAPH_NAME = "memory_graph"


async def backfill_graph_nodes(
    qdrant_url: str,
    falkordb_host: str,
    falkordb_port: int,
    falkordb_password: str | None,
    batch_size: int,
    dry_run: bool,
) -> dict[str, int]:
    """Scan Qdrant memories and create graph nodes in FalkorDB.

    Returns:
        Stats dict with created, skipped, errors counts.
    """
    from falkordb.asyncio import FalkorDB
    from redis.asyncio import BlockingConnectionPool

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, timeout=30)
    collection_info = client.get_collection(COLLECTION)
    total = collection_info.points_count
    logger.info(f"Qdrant collection '{COLLECTION}': {total} points")

    # Connect to FalkorDB
    pool = BlockingConnectionPool(
        host=falkordb_host,
        port=falkordb_port,
        password=falkordb_password,
        max_connections=4,
        timeout=None,
        decode_responses=True,
    )
    db = FalkorDB(connection_pool=pool)
    graph = db.select_graph(GRAPH_NAME)
    logger.info(f"FalkorDB connected: {falkordb_host}:{falkordb_port}/{GRAPH_NAME}")

    # Check existing node count
    try:
        result = await graph.query("MATCH (m:Memory) RETURN count(m) AS cnt")
        existing = result.result_set[0][0] if result.result_set else 0
        logger.info(f"Existing graph nodes: {existing}")
    except Exception as e:
        logger.warning(f"Could not count existing nodes: {e}")
        existing = 0

    stats = {"scanned": 0, "created": 0, "skipped": 0, "errors": 0}
    offset = None

    while True:
        # Scroll through all points
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            point_id = str(point.id)

            # Skip metadata point
            if point_id == METADATA_POINT_ID:
                continue

            stats["scanned"] += 1
            payload = point.payload or {}
            content_hash = payload.get("content_hash")
            created_at = payload.get("created_at")

            if not content_hash:
                logger.warning(f"Point {point_id}: no content_hash, skipping")
                stats["skipped"] += 1
                continue

            # Convert created_at to float timestamp if needed
            if created_at is None:
                created_at = 0.0
            elif isinstance(created_at, str):
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = dt.timestamp()
                except (ValueError, TypeError):
                    created_at = 0.0

            if dry_run:
                logger.debug(f"[DRY RUN] Would create node: {content_hash[:16]}...")
                stats["created"] += 1
                continue

            try:
                await graph.query(
                    "MERGE (m:Memory {content_hash: $hash}) "
                    "ON CREATE SET m.created_at = $ts",
                    params={"hash": content_hash, "ts": float(created_at)},
                )
                stats["created"] += 1
            except Exception as e:
                logger.error(f"Failed to create node for {content_hash[:16]}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Progress: {stats['scanned']}/{total} scanned, "
            f"{stats['created']} created, {stats['errors']} errors"
        )

        if offset is None:
            break

    # Final node count
    if not dry_run:
        try:
            result = await graph.query("MATCH (m:Memory) RETURN count(m) AS cnt")
            final = result.result_set[0][0] if result.result_set else 0
            logger.info(f"Final graph nodes: {final} (was {existing})")
        except Exception:
            pass

    await pool.aclose()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill FalkorDB graph nodes from Qdrant memories")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to FalkorDB")
    parser.add_argument("--batch-size", type=int, default=100, help="Qdrant scroll batch size (default: 100)")
    parser.add_argument("--qdrant-url", default=os.getenv("MCP_QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--falkordb-host", default=os.getenv("MCP_FALKORDB_HOST", "localhost"))
    parser.add_argument("--falkordb-port", type=int, default=int(os.getenv("MCP_FALKORDB_PORT", "6379")))
    parser.add_argument("--falkordb-password", default=os.getenv("MCP_FALKORDB_PASSWORD"))
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")

    start = time.monotonic()
    stats = asyncio.run(
        backfill_graph_nodes(
            qdrant_url=args.qdrant_url,
            falkordb_host=args.falkordb_host,
            falkordb_port=args.falkordb_port,
            falkordb_password=args.falkordb_password,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
    elapsed = time.monotonic() - start

    logger.info(f"Done in {elapsed:.1f}s: {stats}")


if __name__ == "__main__":
    main()
