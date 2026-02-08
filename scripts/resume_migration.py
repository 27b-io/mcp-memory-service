#!/usr/bin/env python3
"""Resume embedding migration - only process missing points."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import time

COLLECTION = "memories"
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000000"
MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
BATCH = 10  # Small batches

url = os.environ.get("MCP_QDRANT_URL")
client = QdrantClient(url=url, timeout=120)

print("Loading model...")
model = SentenceTransformer(MODEL, trust_remote_code=True)
print("Model loaded")

# Get all point IDs that exist
existing_ids = set()
offset = None
while True:
    points, next_offset = client.scroll(COLLECTION, limit=100, offset=offset, with_payload=False, with_vectors=False)
    existing_ids.update(str(p.id) for p in points)
    if next_offset is None:
        break
    offset = next_offset

print(f"Existing points: {len(existing_ids)}")

# Get all memories from dump (we dumped before migration started)
# Actually, we don't have a dump. Let me just check what's in the original collection...
# Wait, the collection was already rebuilt. We can't recover the missing ones easily.

# Alternative: just finish by getting the schema of what should be there from the backup
print("ERROR: Can't resume - need original memory list. Restore from snapshot instead.")
sys.exit(1)
