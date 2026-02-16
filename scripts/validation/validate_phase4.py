#!/usr/bin/env python3
"""
Phase 4 Validation - Memory Cleanup Project (Issue #54)

Validates all acceptance criteria:
1. Zero chunk fragments remain
2. Consolidated memories have valid embeddings and summaries
3. Embedding model handles 8K+ tokens without truncation
4. Graph node count matches memory count
5. Hebbian + CONTRADICTS edges populated
6. Search quality maintained or improved
7. check_database_health() all green

Usage:
    MCP_QDRANT_URL=http://... MCP_FALKORDB_HOST=... \\
        uv run --python 3.12 python scripts/validation/validate_phase4.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"


async def validate_no_chunks(client: QdrantClient, collection: str) -> dict:
    """AC1: Zero chunk fragments remain."""
    logger.info("=== AC1: Checking for chunk fragments ===")

    chunk_count = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            if str(point.id) == METADATA_POINT_ID:
                continue
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            if "chunk_index" in metadata:
                chunk_count += 1
                if chunk_count <= 5:  # Show first few examples
                    logger.warning(f"  Found chunk: {payload.get('content_hash', 'unknown')[:16]}...")

        if offset is None:
            break

    result = {
        "criterion": "Zero chunk fragments remain",
        "passed": chunk_count == 0,
        "chunk_count": chunk_count,
    }

    if result["passed"]:
        logger.info(f"✅ AC1 PASSED: No chunks found")
    else:
        logger.error(f"❌ AC1 FAILED: {chunk_count} chunks remain")

    return result


async def validate_embeddings_and_summaries(client: QdrantClient, collection: str) -> dict:
    """AC2: Consolidated memories have valid embeddings and summaries."""
    logger.info("=== AC2: Checking embeddings and summaries ===")

    stats = {
        "total": 0,
        "missing_vector": 0,
        "missing_summary": 0,
        "invalid_vector_dim": 0,
        "consolidated": 0,
    }

    collection_info = client.get_collection(collection)
    expected_dim = collection_info.config.params.vectors.size

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        for point in points:
            if str(point.id) == METADATA_POINT_ID:
                continue

            stats["total"] += 1
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            if "consolidated_from" in metadata:
                stats["consolidated"] += 1

            if not point.vector or len(point.vector) == 0:
                stats["missing_vector"] += 1
            elif len(point.vector) != expected_dim:
                stats["invalid_vector_dim"] += 1

            if not payload.get("summary"):
                stats["missing_summary"] += 1

        if offset is None:
            break

    result = {
        "criterion": "Valid embeddings and summaries",
        "passed": stats["missing_vector"] == 0 and stats["missing_summary"] == 0 and stats["invalid_vector_dim"] == 0,
        "stats": stats,
        "expected_dim": expected_dim,
    }

    if result["passed"]:
        logger.info(f"✅ AC2 PASSED: All {stats['total']} memories have valid embeddings and summaries")
        logger.info(f"   ({stats['consolidated']} are consolidated memories)")
    else:
        logger.error(f"❌ AC2 FAILED: Issues found")
        logger.error(f"   Missing vectors: {stats['missing_vector']}")
        logger.error(f"   Missing summaries: {stats['missing_summary']}")
        logger.error(f"   Invalid vector dims: {stats['invalid_vector_dim']}")

    return result


async def validate_embedding_model(client: QdrantClient, collection: str) -> dict:
    """AC3: Embedding model handles 8K+ tokens without truncation."""
    logger.info("=== AC3: Checking embedding model capabilities ===")

    # Get model info from metadata point
    try:
        metadata_point = client.retrieve(
            collection_name=collection,
            ids=[METADATA_POINT_ID],
            with_payload=True,
        )

        if metadata_point:
            payload = metadata_point[0].payload or {}
            model_name = payload.get("embedding_model", "unknown")
            vector_size = payload.get("vector_size", 0)

            # Check if model is nomic (supports 8K tokens)
            is_nomic = "nomic" in model_name.lower()
            has_8k_support = is_nomic  # nomic-embed-text-v1.5 supports 8K context

            result = {
                "criterion": "Embedding model handles 8K+ tokens",
                "passed": has_8k_support,
                "model_name": model_name,
                "vector_size": vector_size,
                "supports_8k": has_8k_support,
            }

            if result["passed"]:
                logger.info(f"✅ AC3 PASSED: {model_name} supports 8K+ tokens")
            else:
                logger.error(f"❌ AC3 FAILED: {model_name} does not support 8K tokens")
                logger.error(f"   Expected: nomic-ai/nomic-embed-text-v1.5")

            return result
    except Exception as e:
        logger.warning(f"Could not retrieve metadata point: {e}")

    return {
        "criterion": "Embedding model handles 8K+ tokens",
        "passed": False,
        "error": "Could not retrieve model info",
    }


async def validate_graph_nodes(
    qdrant_client: QdrantClient,
    collection: str,
    falkordb_host: str,
    falkordb_port: int,
    falkordb_password: str | None,
) -> dict:
    """AC4: Graph node count matches memory count."""
    logger.info("=== AC4: Checking graph node count ===")

    from falkordb.asyncio import FalkorDB
    from redis.asyncio import BlockingConnectionPool

    # Get Qdrant memory count
    collection_info = qdrant_client.get_collection(collection)
    memory_count = collection_info.points_count - 1  # Exclude metadata point

    # Get FalkorDB node count
    pool = BlockingConnectionPool(
        host=falkordb_host,
        port=falkordb_port,
        password=falkordb_password,
        decode_responses=True,
    )
    db = FalkorDB(connection_pool=pool)
    graph = db.select_graph("memory_graph")

    try:
        result = await graph.query("MATCH (m:Memory) RETURN count(m) AS cnt")
        graph_node_count = result.result_set[0][0] if result.result_set else 0

        passed = memory_count == graph_node_count

        result = {
            "criterion": "Graph node count matches memory count",
            "passed": passed,
            "memory_count": memory_count,
            "graph_node_count": graph_node_count,
            "difference": abs(memory_count - graph_node_count),
        }

        if passed:
            logger.info(f"✅ AC4 PASSED: {memory_count} memories = {graph_node_count} graph nodes")
        else:
            logger.error(f"❌ AC4 FAILED: Memory count ({memory_count}) != Graph nodes ({graph_node_count})")
            logger.error(f"   Difference: {result['difference']}")

        await pool.aclose()
        return result

    except Exception as e:
        await pool.aclose()
        return {
            "criterion": "Graph node count matches memory count",
            "passed": False,
            "error": str(e),
        }


async def validate_graph_edges(
    falkordb_host: str,
    falkordb_port: int,
    falkordb_password: str | None,
) -> dict:
    """AC5: Hebbian + CONTRADICTS edges populated."""
    logger.info("=== AC5: Checking graph edges ===")

    from falkordb.asyncio import FalkorDB
    from redis.asyncio import BlockingConnectionPool

    pool = BlockingConnectionPool(
        host=falkordb_host,
        port=falkordb_port,
        password=falkordb_password,
        decode_responses=True,
    )
    db = FalkorDB(connection_pool=pool)
    graph = db.select_graph("memory_graph")

    try:
        # Count Hebbian edges
        hebbian_result = await graph.query("MATCH ()-[r:HEBBIAN]->() RETURN count(r) AS cnt")
        hebbian_count = hebbian_result.result_set[0][0] if hebbian_result.result_set else 0

        # Count CONTRADICTS edges
        contradicts_result = await graph.query("MATCH ()-[r:CONTRADICTS]->() RETURN count(r) AS cnt")
        contradicts_count = contradicts_result.result_set[0][0] if contradicts_result.result_set else 0

        # Both should be > 0 for a healthy populated graph
        passed = hebbian_count > 0 and contradicts_count > 0

        result = {
            "criterion": "Hebbian + CONTRADICTS edges populated",
            "passed": passed,
            "hebbian_count": hebbian_count,
            "contradicts_count": contradicts_count,
        }

        if passed:
            logger.info(f"✅ AC5 PASSED: {hebbian_count} Hebbian edges, {contradicts_count} CONTRADICTS edges")
        else:
            logger.error(f"❌ AC5 FAILED: Insufficient edge population")
            logger.error(f"   Hebbian: {hebbian_count}, CONTRADICTS: {contradicts_count}")

        await pool.aclose()
        return result

    except Exception as e:
        await pool.aclose()
        return {
            "criterion": "Hebbian + CONTRADICTS edges populated",
            "passed": False,
            "error": str(e),
        }


async def validate_search_quality(client: QdrantClient, collection: str) -> dict:
    """AC6: Search quality maintained or improved (basic check)."""
    logger.info("=== AC6: Basic search quality check ===")

    # Simple smoke test: search for common terms
    test_queries = [
        "python code",
        "configuration settings",
        "error handling",
    ]

    search_results = []
    for query in test_queries:
        try:
            from sentence_transformers import SentenceTransformer

            # Load model from metadata
            metadata_point = client.retrieve(
                collection_name=collection,
                ids=[METADATA_POINT_ID],
                with_payload=True,
            )
            model_name = metadata_point[0].payload.get("embedding_model", "intfloat/e5-base-v2")

            model = SentenceTransformer(model_name)
            query_vector = model.encode(query).tolist()

            results = client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=5,
            )

            search_results.append({
                "query": query,
                "result_count": len(results),
                "has_results": len(results) > 0,
            })
        except Exception as e:
            logger.warning(f"Search test for '{query}' failed: {e}")
            search_results.append({
                "query": query,
                "error": str(e),
                "has_results": False,
            })

    passed = all(r.get("has_results", False) for r in search_results)

    result = {
        "criterion": "Search quality maintained",
        "passed": passed,
        "test_queries": search_results,
    }

    if passed:
        logger.info(f"✅ AC6 PASSED: All {len(test_queries)} search queries returned results")
    else:
        logger.error(f"❌ AC6 FAILED: Some searches returned no results")

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 Validation - Memory Cleanup Project")
    parser.add_argument("--qdrant-url", default=os.getenv("MCP_QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("MCP_COLLECTION_NAME", "memories"))
    parser.add_argument("--falkordb-host", default=os.getenv("MCP_FALKORDB_HOST", "localhost"))
    parser.add_argument("--falkordb-port", type=int, default=int(os.getenv("MCP_FALKORDB_PORT", "6379")))
    parser.add_argument("--falkordb-password", default=os.getenv("MCP_FALKORDB_PASSWORD"))
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Phase 4 Validation - Memory Cleanup Project (Issue #54)")
    logger.info("=" * 70)
    logger.info(f"Qdrant: {args.qdrant_url}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"FalkorDB: {args.falkordb_host}:{args.falkordb_port}")
    logger.info("=" * 70)

    client = QdrantClient(url=args.qdrant_url, timeout=60)

    results = []

    # Run all validation checks
    results.append(await validate_no_chunks(client, args.collection))
    results.append(await validate_embeddings_and_summaries(client, args.collection))
    results.append(await validate_embedding_model(client, args.collection))
    results.append(await validate_graph_nodes(
        client, args.collection,
        args.falkordb_host, args.falkordb_port, args.falkordb_password
    ))
    results.append(await validate_graph_edges(
        args.falkordb_host, args.falkordb_port, args.falkordb_password
    ))
    results.append(await validate_search_quality(client, args.collection))

    # AC7: check_database_health() - would require MCP server
    logger.info("=== AC7: check_database_health() ===")
    logger.info("⚠️  Run via MCP: mcp__memory__check_database_health()")

    # Summary
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)

    for r in results:
        status = "✅ PASS" if r.get("passed", False) else "❌ FAIL"
        logger.info(f"{status}: {r['criterion']}")

    logger.info("=" * 70)
    logger.info(f"Result: {passed_count}/{total_count} checks passed")

    if passed_count == total_count:
        logger.info("🎉 All validation checks PASSED!")
        sys.exit(0)
    else:
        logger.error(f"⚠️  {total_count - passed_count} validation check(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
