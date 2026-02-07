#!/usr/bin/env python3
"""Backfill extractive summaries for existing memories in Qdrant.

Iterates all memories in the collection, generates extractive summaries
for entries missing them, and updates the Qdrant payload in batch.

Idempotent: skips memories that already have a summary.

Usage:
    uv run --python 3.12 python scripts/backfill_summaries.py [--dry-run] [--batch-size N]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.storage.factory import create_storage_instance  # noqa: E402
from mcp_memory_service.utils.summariser import extract_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill(dry_run: bool = False, batch_size: int = 100) -> None:
    """Run the backfill process."""
    storage = await create_storage_instance()

    try:
        stats = {"total": 0, "skipped": 0, "updated": 0, "errors": 0}

        # Scroll through all memories in batches
        offset = 0
        while True:
            memories = await storage.get_all_memories(limit=batch_size, offset=offset)
            if not memories:
                break

            for memory in memories:
                stats["total"] += 1

                # Skip if summary already exists
                if memory.summary is not None:
                    stats["skipped"] += 1
                    continue

                # Generate extractive summary
                summary = extract_summary(memory.content)
                if summary is None:
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    logger.info(f"[DRY RUN] {memory.content_hash[:12]}: {summary[:80]}")
                    stats["updated"] += 1
                    continue

                # Update Qdrant payload with summary
                try:
                    await storage.update_memory_metadata(
                        memory.content_hash,
                        {"summary": summary},
                        preserve_timestamps=True,
                    )
                    stats["updated"] += 1
                except Exception as e:
                    logger.warning(f"Failed to update {memory.content_hash[:12]}: {e}")
                    stats["errors"] += 1

            offset += batch_size
            logger.info(
                f"Progress: {stats['total']} scanned, "
                f"{stats['updated']} updated, "
                f"{stats['skipped']} skipped, "
                f"{stats['errors']} errors"
            )

        logger.info(f"Backfill complete: {stats}")

    finally:
        if hasattr(storage, "close"):
            await storage.close()


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
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
