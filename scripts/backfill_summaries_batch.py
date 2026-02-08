#!/usr/bin/env python3
"""Backfill summaries using Claude Batch API (50% discount).

Workflow:
1. Fetch all memories from Qdrant
2. Generate batch requests (with custom_id = memory UUID)
3. Submit batch to Anthropic API
4. Poll for completion
5. Parse JSONL results
6. Update Qdrant payloads

Usage:
    # Create and submit batch
    MCP_SUMMARY_ANTHROPIC_BASE_URL=http://lab:8082 \\
    MCP_QDRANT_URL=http://... \\
    uv run --python 3.12 python scripts/backfill_summaries_batch.py --submit

    # Check status
    uv run --python 3.12 python scripts/backfill_summaries_batch.py --status <batch_id>

    # Apply results (after batch completes)
    uv run --python 3.12 python scripts/backfill_summaries_batch.py --apply <batch_id>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.config import Settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "memories"
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"


def create_batch_requests(client: QdrantClient, config: Settings, force: bool = False) -> list[dict]:
    """Generate batch requests for all memories needing summaries."""
    requests = []
    offset = None
    total_scanned = 0
    total_included = 0

    logger.info("Scanning memories to generate batch requests...")

    while True:
        scroll_result = client.scroll(
            collection_name=COLLECTION,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = scroll_result

        if not points:
            break

        for point in points:
            pid = str(point.id)
            if pid == METADATA_POINT_ID:
                continue

            total_scanned += 1
            payload = point.payload or {}

            # Skip if summary exists (unless --force)
            if not force and payload.get("summary") is not None:
                continue

            content = payload.get("content", "")
            if not content:
                continue

            # Size-based model selection
            content_len = len(content)
            model = (
                config.summary.anthropic_model_small
                if content_len < config.summary.anthropic_size_threshold
                else config.summary.anthropic_model_large
            )

            # Generate prompt
            prompt = (
                f"Summarise this memory in one sentence (max {config.summary.max_tokens} tokens). "
                f"Capture the key decision, fact, or conclusion:\n\n{content}"
            )

            requests.append(
                {
                    "custom_id": pid,
                    "params": {
                        "model": model,
                        "max_tokens": config.summary.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                    },
                }
            )
            total_included += 1

        if total_scanned % 500 == 0:
            logger.info(f"Scanned: {total_scanned}, included: {total_included}")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Batch generation complete: {total_scanned} scanned, {total_included} requests created")
    return requests


async def submit_batch(requests: list[dict], config: Settings) -> dict:
    """Submit batch to Anthropic API."""
    url = f"{config.summary.anthropic_base_url.rstrip('/')}/v1/messages/batches"

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Add API key if provided (optional for proxy)
    if config.summary.anthropic_api_key:
        headers["x-api-key"] = config.summary.anthropic_api_key.get_secret_value()

    payload = {"requests": requests}

    logger.info(f"Submitting batch with {len(requests)} requests to {url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    logger.info(f"Batch submitted successfully: {data.get('id')}")
    logger.info(f"Status: {data.get('processing_status')}")

    return data


async def get_batch_status(batch_id: str, config: Settings) -> dict:
    """Get batch status from Anthropic API."""
    url = f"{config.summary.anthropic_base_url.rstrip('/')}/v1/messages/batches/{batch_id}"

    headers = {
        "anthropic-version": "2023-06-01",
    }

    if config.summary.anthropic_api_key:
        headers["x-api-key"] = config.summary.anthropic_api_key.get_secret_value()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

    return data


async def download_results(results_url: str, config: Settings) -> list[dict]:
    """Download batch results JSONL."""
    headers = {}
    if config.summary.anthropic_api_key:
        headers["x-api-key"] = config.summary.anthropic_api_key.get_secret_value()

    logger.info(f"Downloading results from {results_url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(results_url, headers=headers)
        response.raise_for_status()
        text = response.text

    # Parse JSONL
    results = []
    for line in text.strip().split("\n"):
        if line:
            results.append(json.loads(line))

    logger.info(f"Downloaded {len(results)} results")
    return results


def apply_results(results: list[dict], client: QdrantClient) -> dict:
    """Apply batch results to Qdrant."""
    stats = {"total": 0, "updated": 0, "errors": 0}

    for result in results:
        stats["total"] += 1
        custom_id = result.get("custom_id")
        result_data = result.get("result")

        if not result_data:
            logger.warning(f"No result data for {custom_id}")
            stats["errors"] += 1
            continue

        # Check result type (succeeded, errored, canceled, expired)
        result_type = result_data.get("type")
        if result_type != "succeeded":
            logger.warning(f"Non-success result for {custom_id}: type={result_type}")
            stats["errors"] += 1
            continue

        # Extract summary from Batch API response (nested under message.content)
        message = result_data.get("message", {})
        content_blocks = message.get("content", [])
        text_block = next((block for block in content_blocks if block.get("type") == "text"), None)

        if not text_block:
            logger.warning(f"No text block in result for {custom_id}")
            stats["errors"] += 1
            continue

        summary = text_block.get("text", "").strip()
        if not summary:
            logger.warning(f"Empty summary for {custom_id}")
            stats["errors"] += 1
            continue

        # Update Qdrant
        try:
            client.set_payload(
                collection_name=COLLECTION,
                payload={"summary": summary},
                points=PointIdsList(points=[custom_id]),
            )
            stats["updated"] += 1

            if stats["updated"] % 100 == 0:
                logger.info(f"Progress: {stats['updated']} updated, {stats['errors']} errors")

        except Exception as e:
            logger.warning(f"Failed to update {custom_id}: {e}")
            stats["errors"] += 1

    logger.info(f"Apply complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill summaries via Claude Batch API")
    parser.add_argument("--submit", action="store_true", help="Create and submit batch")
    parser.add_argument("--status", metavar="BATCH_ID", help="Check batch status")
    parser.add_argument("--apply", metavar="BATCH_ID", help="Apply batch results to Qdrant")
    parser.add_argument("--force", action="store_true", help="Regenerate all summaries (not just missing)")
    parser.add_argument("--url", help="Qdrant URL (overrides MCP_QDRANT_URL)")
    parser.add_argument("--poll", action="store_true", help="Poll until batch completes (use with --status)")

    args = parser.parse_args()

    # Load config
    config = Settings()

    if args.submit:
        # Connect to Qdrant
        url = args.url or os.environ.get("MCP_QDRANT_URL")
        if not url:
            logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
            sys.exit(1)

        client = QdrantClient(url=url, timeout=30)
        try:
            # Generate batch requests
            requests = create_batch_requests(client, config, force=args.force)

            if not requests:
                logger.info("No requests to submit")
                sys.exit(0)

            # Submit batch
            batch_data = asyncio.run(submit_batch(requests, config))
            print(f"\nBatch ID: {batch_data.get('id')}")
            print(f"Status: {batch_data.get('processing_status')}")
            print(f"\nCheck status with: --status {batch_data.get('id')}")
        finally:
            client.close()

    elif args.status:
        batch_id = args.status

        if args.poll:

            async def poll_until_complete(bid: str, cfg: Settings) -> None:
                logger.info(f"Polling batch {bid} until completion...")
                while True:
                    status_data = await get_batch_status(bid, cfg)
                    processing_status = status_data.get("processing_status")
                    request_counts = status_data.get("request_counts", {})

                    print(f"\nStatus: {processing_status}")
                    print(f"Requests: {request_counts}")

                    if processing_status in ["ended", "failed", "expired"]:
                        if processing_status == "ended":
                            results_url = status_data.get("results_url")
                            print("\n✅ Batch complete!")
                            print(f"Results URL: {results_url}")
                            print(f"\nApply results with: --apply {bid}")
                        else:
                            print(f"\n❌ Batch {processing_status}")
                        break

                    await asyncio.sleep(60)  # Poll every minute

            asyncio.run(poll_until_complete(batch_id, config))
        else:
            status_data = asyncio.run(get_batch_status(batch_id, config))
            print(json.dumps(status_data, indent=2))

    elif args.apply:
        batch_id = args.apply

        # Get batch status to retrieve results URL
        status_data = asyncio.run(get_batch_status(batch_id, config))
        processing_status = status_data.get("processing_status")

        if processing_status != "ended":
            logger.error(f"Batch not complete yet (status: {processing_status})")
            sys.exit(1)

        results_url = status_data.get("results_url")
        if not results_url:
            logger.error("No results_url in batch status")
            sys.exit(1)

        # Download results
        results = asyncio.run(download_results(results_url, config))

        # Connect to Qdrant
        url = args.url or os.environ.get("MCP_QDRANT_URL")
        if not url:
            logger.error("No Qdrant URL. Set MCP_QDRANT_URL or use --url")
            sys.exit(1)

        client = QdrantClient(url=url, timeout=30)
        try:
            # Apply results
            stats = apply_results(results, client)
            print(f"\n✅ Applied {stats['updated']} summaries ({stats['errors']} errors)")
        finally:
            client.close()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
