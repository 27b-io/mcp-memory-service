#!/usr/bin/env python3
"""Backfill summaries for existing memories in Qdrant.

Supports both extractive (zero-cost) and LLM-powered (Gemini) summarisation.

Iterates all memories in the collection using Qdrant's native scroll API,
generates summaries for entries missing them, and updates the Qdrant payload
via set_payload.

Idempotent: skips memories that already have a summary.

Usage:
    # Extractive mode (default)
    MCP_QDRANT_URL=http://... uv run --python 3.12 python scripts/backfill_summaries.py [--dry-run] [--batch-size N]

    # LLM mode (requires API key)
    MCP_SUMMARY_API_KEY=... MCP_QDRANT_URL=http://... \\
        uv run --python 3.12 python scripts/backfill_summaries.py --mode llm [--rate-limit 10]
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.config import Settings  # noqa: E402
from mcp_memory_service.utils.summariser import extract_summary, llm_summarise  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
# Qdrant uses this point ID for internal metadata — skip it
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"


async def _generate_summary_llm(content: str, config: Settings) -> str | None:
    """Generate LLM summary with fallback to extractive."""
    if not config.summary.api_key:
        logger.warning("LLM mode requested but no API key configured - using extractive")
        return extract_summary(content)

    summary = await llm_summarise(
        content,
        api_key=config.summary.api_key.get_secret_value(),
        model=config.summary.model,
        max_tokens=config.summary.max_tokens,
        timeout=config.summary.timeout_seconds,
    )

    # Fallback to extractive on error
    if summary is None:
        logger.debug("LLM summarise returned None, falling back to extractive")
        return extract_summary(content)

    return summary


def backfill(
    client: QdrantClient,
    mode: str = "extractive",
    dry_run: bool = False,
    batch_size: int = 100,
    rate_limit: float = 0.0,
    config: Settings | None = None,
) -> dict:
    """Run the backfill process using Qdrant's native scroll API.

    Args:
        client: Qdrant client instance.
        mode: 'extractive' or 'llm' summarisation mode.
        dry_run: If True, log changes without updating Qdrant.
        batch_size: Number of memories to fetch per scroll batch.
        rate_limit: Requests per second for LLM mode (0 = no limit).
        config: Settings object for LLM configuration.

    Uses offset-based pagination (next_page_offset) which is reliable
    regardless of duplicate field values.
    """
    stats = {"total": 0, "skipped": 0, "updated": 0, "errors": 0}
    offset = None  # First page
    last_request_time = 0.0

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

            # Rate limiting for LLM mode
            if mode == "llm" and rate_limit > 0:
                time_since_last = time.time() - last_request_time
                min_interval = 1.0 / rate_limit
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
                last_request_time = time.time()

            # Generate summary based on mode
            if mode == "llm":
                summary = asyncio.run(_generate_summary_llm(content, config))
            else:
                summary = extract_summary(content)

            if summary is None:
                stats["skipped"] += 1
                continue

            content_hash = str(payload.get("content_hash", pid))[:12]

            if dry_run:
                logger.info(f"[DRY RUN] [{mode.upper()}] {content_hash}: {summary[:80]}")
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extractive", "llm"],
        default="extractive",
        help="Summary mode: 'extractive' (default, zero-cost) or 'llm' (Gemini, requires API key)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=10.0,
        help="LLM requests per second (default: 10, only applies to --mode llm)",
    )
    args = parser.parse_args()

    # Resolve Qdrant URL
    url = args.url or os.environ.get("MCP_QDRANT_URL")
    if not url:
        logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
        sys.exit(1)

    # Load config for LLM mode
    config = None
    if args.mode == "llm":
        config = Settings()
        if not config.summary.api_key:
            logger.error("LLM mode requires MCP_SUMMARY_API_KEY environment variable")
            sys.exit(1)
        logger.info(f"LLM mode: using model {config.summary.model}")
        logger.info(f"Rate limit: {args.rate_limit} requests/second")

    logger.info(f"Connecting to Qdrant at {url}")
    client = QdrantClient(url=url, timeout=30)

    try:
        # Verify connection
        info = client.get_collection(COLLECTION)
        logger.info(f"Collection '{COLLECTION}': {info.points_count} points")

        # Run backfill
        run_mode = "DRY RUN" if args.dry_run else "LIVE"
        logger.info(f"Starting backfill ({run_mode}, mode={args.mode}, batch_size={args.batch_size})")
        stats = backfill(
            client,
            mode=args.mode,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            rate_limit=args.rate_limit if args.mode == "llm" else 0.0,
            config=config,
        )
        logger.info(f"Backfill complete: {stats}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
