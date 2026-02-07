#!/usr/bin/env python3
"""Backfill extractive summaries for existing memories in Qdrant.

Iterates all memories in the collection using Qdrant's native scroll API,
generates extractive summaries for entries missing them, and updates the
Qdrant payload via set_payload.

Idempotent: skips memories that already have a summary.

Usage:
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/backfill_summaries.py [--dry-run] [--batch-size N]
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/backfill_summaries.py --url http://localhost:6333
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Add project root to path for summariser import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.utils.summariser import extract_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
# Qdrant uses this point ID for internal metadata — skip it
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"


def backfill(client: QdrantClient, dry_run: bool = False, batch_size: int = 100) -> dict:
    """Run the backfill process using Qdrant's native scroll API.

    Uses offset-based pagination (next_page_offset) which is reliable
    regardless of duplicate field values.
    """
    stats = {"total": 0, "skipped": 0, "updated": 0, "errors": 0}
    offset = None  # First page

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            pid = str(point.id)
            if pid == METADATA_POINT_ID:
                continue

            stats["total"] += 1
            payload = point.payload or {}

            # Skip if summary already exists
            if payload.get("summary") is not None:
                stats["skipped"] += 1
                continue

            content = payload.get("content", "")
            if not content:
                stats["skipped"] += 1
                continue

            # Generate extractive summary
            summary = extract_summary(content)
            if summary is None:
                stats["skipped"] += 1
                continue

            content_hash = str(payload.get("content_hash", pid))[:12]

            if dry_run:
                logger.info(f"[DRY RUN] {content_hash}: {summary[:80]}")
                stats["updated"] += 1
                continue

            # Update Qdrant payload with summary (preserves all existing fields)
            try:
                client.set_payload(
                    collection_name=COLLECTION,
                    payload={"summary": summary},
                    points=PointIdsList(points=[point.id]),
                )
                stats["updated"] += 1
            except Exception as e:
                logger.warning(f"Failed to update {content_hash}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Progress: {stats['total']} scanned, "
            f"{stats['updated']} updated, "
            f"{stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

        if next_offset is None:
            break
        offset = next_offset

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory summaries")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of memories to process per batch (default: 100)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Qdrant server URL (overrides MCP_QDRANT_URL env var)",
    )
    args = parser.parse_args()

    # Resolve Qdrant URL
    url = args.url or os.environ.get("MCP_QDRANT_URL")
    if not url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
        sys.exit(1)

    logger.info(f"Connecting to Qdrant at {url}")
    client = QdrantClient(url=url, timeout=30)

    try:
        # Verify connection
        info = client.get_collection(COLLECTION)
        logger.info(f"Collection '{COLLECTION}': {info.points_count} points")

        # Run backfill
        mode = "DRY RUN" if args.dry_run else "LIVE"
        logger.info(f"Starting backfill ({mode}, batch_size={args.batch_size})")
        stats = backfill(client, dry_run=args.dry_run, batch_size=args.batch_size)
        logger.info(f"Backfill complete: {stats}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
