#!/usr/bin/env python3
"""Migrate embedding model by re-embedding all memories.

Handles dimension changes (e.g. 768→1024) by recreating the Qdrant collection.
Preserves all payloads. Generates new vectors with the target model.

Usage:
    # Preview
    CUDA_VISIBLE_DEVICES="" MCP_QDRANT_URL=http://... \
        uv run --python 3.12 python scripts/migrate_embeddings.py \
        --model Snowflake/snowflake-arctic-embed-l-v2.0 --dry-run

    # Run migration
    CUDA_VISIBLE_DEVICES="" MCP_QDRANT_URL=http://... \
        uv run --python 3.12 python scripts/migrate_embeddings.py \
        --model Snowflake/snowflake-arctic-embed-l-v2.0
"""

import argparse
import logging
import os
import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"
BATCH_SIZE = 50


def dump_all_memories(client: QdrantClient) -> list[dict]:
    """Extract all memories with payloads (no vectors needed)."""
    memories = []
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

        for p in points:
            memories.append({"id": str(p.id), "payload": p.payload or {}})

        if next_offset is None:
            break
        offset = next_offset

    return memories


def migrate(client: QdrantClient, model_name: str, dry_run: bool = True) -> dict:
    """Run the full migration."""
    from sentence_transformers import SentenceTransformer

    stats = {"total": 0, "embedded": 0, "metadata_preserved": 0, "errors": 0}

    # Step 1: Dump all memories
    logger.info("Step 1: Dumping all memories...")
    memories = dump_all_memories(client)
    stats["total"] = len(memories)
    logger.info(f"Dumped {len(memories)} points (including metadata point)")

    if dry_run:
        # Load model to check dimensions
        logger.info(f"Loading model {model_name} to check dimensions...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        test_emb = model.encode("test")
        new_dims = len(test_emb)

        # Check current collection dims
        info = client.get_collection(COLLECTION)
        current_dims = info.config.params.vectors.size
        logger.info(f"Current dimensions: {current_dims}")
        logger.info(f"New dimensions: {new_dims}")
        logger.info(f"Collection rebuild needed: {current_dims != new_dims}")
        logger.info(f"Memories to re-embed: {sum(1 for m in memories if m['id'] != METADATA_POINT_ID)}")
        print(f"\n[DRY RUN] Would migrate {len(memories)} points from {current_dims}→{new_dims} dims using {model_name}")
        return stats

    # Step 2: Load new embedding model
    logger.info(f"Step 2: Loading model {model_name}...")
    t0 = time.time()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    test_emb = model.encode("test")
    new_dims = len(test_emb)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s — {new_dims} dimensions")

    # Step 3: Check if rebuild needed
    info = client.get_collection(COLLECTION)
    current_dims = info.config.params.vectors.size
    needs_rebuild = current_dims != new_dims

    if needs_rebuild:
        logger.info(f"Step 3: Rebuilding collection ({current_dims}→{new_dims} dims)...")
        client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=new_dims, distance=Distance.COSINE),
        )
        logger.info("Collection recreated")
    else:
        logger.info(f"Step 3: Dimensions match ({current_dims}), updating vectors in-place")

    # Step 4: Re-embed and insert in batches
    logger.info(f"Step 4: Re-embedding {len(memories)} points...")
    batch = []
    batch_num = 0

    for i, mem in enumerate(memories):
        point_id = mem["id"]
        payload = mem["payload"]

        # Metadata point — preserve without vector
        if point_id == METADATA_POINT_ID:
            # Re-insert metadata with a zero vector
            batch.append(PointStruct(
                id=point_id,
                vector=[0.0] * new_dims,
                payload=payload,
            ))
            stats["metadata_preserved"] += 1
            continue

        # Regular memory — re-embed content
        content = payload.get("content", "")
        if not content:
            logger.warning(f"[{i+1}/{len(memories)}] Empty content for {point_id[:12]}, skipping")
            stats["errors"] += 1
            continue

        try:
            # Nomic/Arctic models use prefixes for better retrieval
            # For storage, use "search_document: " prefix
            prefixed = f"search_document: {content}"
            embedding = model.encode(prefixed).tolist()

            batch.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))
            stats["embedded"] += 1

        except Exception as e:
            logger.error(f"[{i+1}/{len(memories)}] Error embedding {point_id[:12]}: {e}")
            stats["errors"] += 1

        # Flush batch
        if len(batch) >= BATCH_SIZE:
            batch_num += 1
            client.upsert(collection_name=COLLECTION, points=batch)
            logger.info(f"  Batch {batch_num}: {len(batch)} points inserted ({stats['embedded']}/{stats['total']})")
            batch = []

    # Flush remaining
    if batch:
        batch_num += 1
        client.upsert(collection_name=COLLECTION, points=batch)
        logger.info(f"  Batch {batch_num}: {len(batch)} points inserted ({stats['embedded']}/{stats['total']})")

    # Step 5: Update metadata point with new model info
    logger.info("Step 5: Updating collection metadata...")
    try:
        client.set_payload(
            collection_name=COLLECTION,
            payload={
                "embedding_model": model_name,
                "vector_size": new_dims,
                "migrated_at": time.time(),
                "migrated_from": f"intfloat/e5-base-v2 ({current_dims}d)",
            },
            points=[METADATA_POINT_ID],
        )
    except Exception as e:
        logger.warning(f"Failed to update metadata point: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate embedding model")
    parser.add_argument("--model", required=True, help="Target embedding model name")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--url", help="Qdrant URL (overrides MCP_QDRANT_URL)")
    args = parser.parse_args()

    url = args.url or os.environ.get("MCP_QDRANT_URL")
    if not url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
        sys.exit(1)

    client = QdrantClient(url=url, timeout=120)

    try:
        t0 = time.time()
        stats = migrate(client, model_name=args.model, dry_run=args.dry_run)
        elapsed = time.time() - t0

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Migration {'preview' if args.dry_run else 'complete'} in {elapsed:.1f}s:")
        print(f"  Total points: {stats['total']}")
        print(f"  Embedded: {stats['embedded']}")
        print(f"  Metadata preserved: {stats['metadata_preserved']}")
        print(f"  Errors: {stats['errors']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
