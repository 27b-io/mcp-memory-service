# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Storage backend factory for the MCP Memory Service.

Creates and initializes the Qdrant storage backend.
"""

import logging

from .base import MemoryStorage
from .qdrant_storage import QdrantStorage

logger = logging.getLogger(__name__)


async def create_storage_instance() -> MemoryStorage:
    """
    Create and initialize the Qdrant storage backend instance.

    Returns:
        Initialized QdrantStorage instance
    """
    from ..config import EMBEDDING_MODEL_NAME, settings

    logger.info("Creating Qdrant storage backend instance...")

    # Determine mode: server (URL) or embedded (path)
    if settings.qdrant.url:
        storage = QdrantStorage(
            url=settings.qdrant.url,
            embedding_model=EMBEDDING_MODEL_NAME,
            collection_name=settings.qdrant.COLLECTION_NAME,
            quantization_enabled=settings.qdrant.quantization_enabled,
            distance_metric=settings.qdrant.DISTANCE_METRIC,
        )
        logger.info(f"Initialized Qdrant storage in server mode: {settings.qdrant.url}")
    else:
        storage = QdrantStorage(
            storage_path=settings.qdrant.storage_path,
            embedding_model=EMBEDDING_MODEL_NAME,
            collection_name=settings.qdrant.COLLECTION_NAME,
            quantization_enabled=settings.qdrant.quantization_enabled,
            distance_metric=settings.qdrant.DISTANCE_METRIC,
        )
        logger.info(f"Initialized Qdrant storage in embedded mode: {settings.qdrant.storage_path}")

    await storage.initialize()
    logger.info("QdrantStorage initialized successfully")

    return storage
