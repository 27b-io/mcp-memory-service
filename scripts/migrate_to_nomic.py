#!/usr/bin/env python3
"""
Migrate production Qdrant to nomic-embed-text-v1.5 (768 dims, 8K context).

Handles dimension change from 1024 (Snowflake Arctic) to 768 (nomic).
Creates new collection, re-embeds all memories, atomic swap.

Usage:
    # Dry run first
    MCP_QDRANT_URL=http://100.119.9.90:26333 uv run --python 3.12 python scripts/migrate_to_nomic.py --dry-run

    # Run migration
    MCP_QDRANT_URL=http://100.119.9.90:26333 uv run --python 3.12 python scripts/migrate_to_nomic.py
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OLD_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"  # 1024 dims
NEW_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # 768 dims, 8K context

async def migrate_embeddings(
    url: str,
    dry_run: bool = False,
    batch_size: int = 50,
    keep_backup: bool = True,
    old_collection: str = "memories",
):
    """
    Migrate memories to nomic-embed-text-v1.5.

    Process:
    1. Read all memories from 'memories' collection
    2. Create new collection 'memories_nomic768' with 768-dim vectors
    3. Re-embed all content with nomic model
    4. Store in new collection
    5. Atomic swap: memories → memories_backup, memories_nomic768 → memories

    Args:
        url: Qdrant server URL
        dry_run: Preview without making changes
        batch_size: Memories per batch
        keep_backup: Keep old collection after swap
    """
    start_time = time.time()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    new_collection = f"memories_nomic768"
    backup_collection = f"{old_collection}_backup_{timestamp}"

    logger.info("=" * 60)
    logger.info("Embedding Model Migration: Arctic → Nomic")
    logger.info("=" * 60)
    logger.info(f"Old model: {OLD_MODEL} (1024 dims)")
    logger.info(f"New model: {NEW_MODEL} (768 dims, 8K context)")
    logger.info(f"Old collection: {old_collection}")
    logger.info(f"New collection: {new_collection}")
    logger.info(f"Backup collection: {backup_collection}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Keep backup: {keep_backup}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    client = QdrantClient(url=url, timeout=120)

    try:
        # Get total count
        collection_info = client.get_collection(old_collection)
        total_count = collection_info.points_count
        logger.info(f"✓ Found {total_count:,} memories in {old_collection}")

        if dry_run:
            time_per_memory = 0.1  # Conservative estimate
            estimated_time = total_count * time_per_memory
            logger.info(f"\n[DRY RUN] Migration plan:")
            logger.info(f"  Memories to migrate: {total_count:,}")
            logger.info(f"  Batches ({batch_size} each): {(total_count + batch_size - 1) // batch_size}")
            logger.info(f"  Estimated time: {estimated_time:.1f}s ({estimated_time / 60:.1f} min)")
            logger.info(f"  Old dims (1024) → New dims (768)")
            logger.info(f"\nNo changes made (dry run)")
            return {
                "dry_run": True,
                "total_memories": total_count,
                "estimated_time_seconds": estimated_time,
            }

        # Load new model
        logger.info(f"\nLoading {NEW_MODEL}...")
        model = SentenceTransformer(NEW_MODEL, trust_remote_code=True)
        logger.info(f"✓ Model loaded: {model.get_sentence_embedding_dimension()} dims, max_seq={model.max_seq_length}")

        # Create new collection with 768 dimensions
        logger.info(f"\nCreating new collection: {new_collection} (768 dims)...")
        client.create_collection(
            collection_name=new_collection,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"✓ Collection created")

        # Migrate in batches
        logger.info(f"\nMigrating {total_count:,} memories...")
        migrated = 0
        failed = 0
        offset = None
        batch_num = 0

        while True:
            batch_num += 1
            batch_start = time.time()

            # Scroll batch
            points, next_offset = client.scroll(
                collection_name=old_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            # Re-embed batch
            new_points = []
            for point in points:
                try:
                    payload = point.payload
                    content = payload.get("content", "")

                    # Generate new embedding with nomic
                    embedding = model.encode(content).tolist()

                    # Create new point with same ID and payload
                    new_point = PointStruct(
                        id=point.id,
                        vector=embedding,
                        payload=payload,
                    )
                    new_points.append(new_point)
                    migrated += 1

                except Exception as e:
                    logger.error(f"Failed to re-embed point {point.id}: {e}")
                    failed += 1

            # Upsert batch to new collection
            if new_points:
                client.upsert(collection_name=new_collection, points=new_points)

            batch_time = time.time() - batch_start
            logger.info(
                f"  Batch {batch_num}: {len(points)} memories → {len(new_points)} migrated "
                f"({migrated:,}/{total_count:,}, {migrated/total_count*100:.1f}%) "
                f"[{batch_time:.1f}s, {len(points)/batch_time:.1f} mem/s]"
            )

            if next_offset is None:
                break
            offset = next_offset

        # Verify counts
        new_count = client.get_collection(new_collection).points_count
        logger.info(f"\n✓ Migration complete:")
        logger.info(f"  Source: {total_count:,} memories")
        logger.info(f"  Target: {new_count:,} memories")
        logger.info(f"  Migrated: {migrated:,}")
        logger.info(f"  Failed: {failed}")

        if new_count != total_count:
            raise ValueError(f"Count mismatch: expected {total_count}, got {new_count}")

        # Atomic swap
        logger.info(f"\nPerforming atomic collection swap...")
        logger.info(f"  Step 1: {old_collection} → {backup_collection}")
        client.update_collection_aliases(
            change_aliases_operations=[
                {
                    "rename_alias": {
                        "old_collection_name": old_collection,
                        "new_collection_name": backup_collection,
                    }
                }
            ]
        )

        logger.info(f"  Step 2: {new_collection} → {old_collection}")
        client.update_collection_aliases(
            change_aliases_operations=[
                {
                    "rename_alias": {
                        "old_collection_name": new_collection,
                        "new_collection_name": old_collection,
                    }
                }
            ]
        )

        logger.info(f"✓ Collections swapped")

        if not keep_backup:
            logger.info(f"\nDeleting backup collection: {backup_collection}")
            client.delete_collection(backup_collection)
            logger.info(f"✓ Backup deleted")
        else:
            logger.info(f"\n✓ Backup preserved: {backup_collection}")

        elapsed = time.time() - start_time
        logger.info(f"\n" + "=" * 60)
        logger.info(f"Migration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"New model: {NEW_MODEL} (768 dims, 8K context)")
        logger.info(f"Active collection: {old_collection} ({new_count:,} memories)")
        logger.info("=" * 60)

        return {
            "success": True,
            "total_memories": total_count,
            "migrated": migrated,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "backup_collection": backup_collection if keep_backup else None,
        }

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate to nomic-embed-text-v1.5")
    parser.add_argument("--url", help="Qdrant URL (overrides MCP_QDRANT_URL)")
    parser.add_argument("--old-collection", default="memories", help="Source collection name (default: memories)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--batch-size", type=int, default=50, help="Memories per batch")
    parser.add_argument("--no-backup", action="store_true", help="Delete backup after swap")
    args = parser.parse_args()

    url = args.url or os.environ.get("MCP_QDRANT_URL")
    if not url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
        sys.exit(1)

    result = asyncio.run(migrate_embeddings(
        url=url,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        keep_backup=not args.no_backup,
        old_collection=args.old_collection,
    ))

    if result.get("success"):
        logger.info("\n✓ Migration successful")
    elif result.get("dry_run"):
        logger.info("\n✓ Dry run complete - use without --dry-run to migrate")


if __name__ == "__main__":
    main()
