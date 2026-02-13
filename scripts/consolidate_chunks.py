#!/usr/bin/env python3
"""Consolidate legacy chunk fragments back into parent memories.

Chunks were created by old SQLite/Cloudflare backends that had content length limits.
Qdrant has no such limit, so chunks are unnecessary. This script reassembles them.

Each chunk has metadata: chunk_index (ordering) and original_hash (parent ID).
Parent memories no longer exist — chunks are all that remains.

Usage:
    # Preview what would happen
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/consolidate_chunks.py --dry-run

    # Run consolidation (all parents)
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/consolidate_chunks.py

    # Run in batches of 50 parents (idempotent - can resume)
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/consolidate_chunks.py --batch-size 50

    # With LLM summaries for consolidated memories
    MCP_SUMMARY_ANTHROPIC_BASE_URL=http://lab:8082 MCP_QDRANT_URL=http://... \\
        uv run --python 3.12 python scripts/consolidate_chunks.py --resummarise
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_memory_service.config import Settings
from mcp_memory_service.utils.summariser import summarise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"


def discover_chunks(client: QdrantClient) -> dict[str, list[dict]]:
    """Scan all memories and group chunks by original_hash."""
    parents: dict[str, list[dict]] = defaultdict(list)
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
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

            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            if "chunk_index" in metadata:
                parents[metadata.get("original_hash", "unknown")].append(
                    {
                        "point_id": pid,
                        "chunk_index": metadata["chunk_index"],
                        "content": payload.get("content", ""),
                        "content_hash": payload.get("content_hash", ""),
                        "tags": payload.get("tags", []),
                        "metadata": metadata,
                        "created_at": payload.get("created_at"),
                    }
                )

        if next_offset is None:
            break
        offset = next_offset

    return dict(parents)


def reassemble(chunks: list[dict]) -> dict:
    """Reassemble chunks into a single memory."""
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_index"])

    # Concatenate content
    content = "\n".join(c["content"] for c in sorted_chunks)

    # Use tags from first chunk (they're all the same)
    tags = sorted_chunks[0].get("tags", [])

    # Earliest created_at
    created_at = min(c["created_at"] for c in sorted_chunks if c.get("created_at"))

    # Build audit metadata
    metadata = {
        "consolidated_from": [c["content_hash"] for c in sorted_chunks],
        "original_hash": sorted_chunks[0]["metadata"].get("original_hash", ""),
        "chunk_count": len(sorted_chunks),
        "consolidation_timestamp": time.time(),
    }

    return {
        "content": content,
        "tags": tags,
        "metadata": metadata,
        "created_at": created_at,
    }


async def consolidate(
    client: QdrantClient,
    dry_run: bool = True,
    resummarise: bool = False,
    config: Settings | None = None,
    batch_size: int | None = None,
) -> dict:
    """Run the consolidation process."""
    import hashlib
    import uuid

    from qdrant_client.models import PointStruct
    from sentence_transformers import SentenceTransformer

    stats = {"parents": 0, "chunks_removed": 0, "consolidated": 0, "errors": 0, "skipped": 0}

    logger.info("Discovering chunks...")
    parents = discover_chunks(client)
    stats["parents"] = len(parents)

    total_chunks = sum(len(chunks) for chunks in parents.values())
    logger.info(f"Found {len(parents)} parents with {total_chunks} total chunks")

    # Apply batch limit if specified
    parents_to_process = dict(parents)
    if batch_size is not None and batch_size > 0:
        parents_to_process = dict(list(parents.items())[:batch_size])
        logger.info(f"Processing batch of {len(parents_to_process)} parents (batch_size={batch_size})")

    # Load embedding model once
    model = None
    if not dry_run:
        model_name = os.environ.get("MCP_MEMORY_EMBEDDING_MODEL", "intfloat/e5-base-v2")
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    for i, (original_hash, chunks) in enumerate(parents_to_process.items(), 1):
        try:
            reassembled = reassemble(chunks)
            content = reassembled["content"]

            if dry_run:
                logger.info(
                    f"[DRY RUN] [{i}/{len(parents_to_process)}] {original_hash[:12]}: {len(chunks)} chunks → "
                    f"{len(content)} chars, tags={reassembled['tags'][:3]}"
                )
                stats["consolidated"] += 1
                stats["chunks_removed"] += len(chunks)
                continue

            # Generate content hash for consolidated memory
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Check if consolidated memory already exists
            existing = client.scroll(
                collection_name=COLLECTION,
                limit=1,
                scroll_filter={"must": [{"key": "content_hash", "match": {"value": content_hash}}]},
                with_payload=False,
                with_vectors=False,
            )
            if existing[0]:
                logger.info(f"[{i}/{len(parents_to_process)}] Skipping {original_hash[:12]}: already consolidated")
                stats["skipped"] += 1
                continue

            # Generate summary
            summary = None
            if resummarise and config:
                try:
                    summary = await summarise(content, config=config)
                except Exception as e:
                    logger.warning(f"Summary generation failed for {original_hash[:12]}: {e}")

            if not summary:
                # Extractive fallback
                summary = await summarise(content)

            # Generate embedding
            embedding = model.encode(content).tolist()

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "content_hash": content_hash,
                    "tags": reassembled["tags"],
                    "metadata": reassembled["metadata"],
                    "created_at": reassembled["created_at"],
                    "updated_at": time.time(),
                    "memory_type": "consolidated",
                    "summary": summary,
                },
            )

            client.upsert(collection_name=COLLECTION, points=[point])

            # Delete original chunks
            chunk_point_ids = [c["point_id"] for c in chunks]
            client.delete(
                collection_name=COLLECTION,
                points_selector=PointIdsList(points=chunk_point_ids),
            )

            stats["consolidated"] += 1
            stats["chunks_removed"] += len(chunks)
            logger.info(f"[{i}/{len(parents_to_process)}] {original_hash[:12]}: {len(chunks)} chunks → {len(content)} chars ✓")

        except Exception as e:
            logger.error(f"[{i}/{len(parents_to_process)}] Error consolidating {original_hash[:12]}: {e}")
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Consolidate legacy chunk fragments")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--resummarise", action="store_true", help="Generate LLM summaries for consolidated memories")
    parser.add_argument("--batch-size", type=int, help="Process at most N parent groups per run (default: all)")
    parser.add_argument("--url", help="Qdrant URL (overrides MCP_QDRANT_URL)")
    args = parser.parse_args()

    url = args.url or os.environ.get("MCP_QDRANT_URL")
    if not url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
        sys.exit(1)

    config = Settings() if args.resummarise else None
    client = QdrantClient(url=url, timeout=60)

    try:
        stats = asyncio.run(
            consolidate(
                client,
                dry_run=args.dry_run,
                resummarise=args.resummarise,
                config=config,
                batch_size=args.batch_size,
            )
        )

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Consolidation complete:")
        print(f"  Parents found: {stats['parents']}")
        print(f"  Consolidated: {stats['consolidated']}")
        print(f"  Chunks removed: {stats['chunks_removed']}")
        print(f"  Skipped (already exists): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
