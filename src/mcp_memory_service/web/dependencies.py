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
FastAPI dependencies for the HTTP interface.
"""

import logging

from fastapi import Depends, HTTPException

from ..config import settings
from ..services.memory_service import MemoryService
from ..services.quota_service import QuotaService
from ..shared_storage import get_graph_client, get_write_queue
from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)

# Global storage instance
_storage: MemoryStorage | None = None
# Global quota service instance
_quota_service: QuotaService | None = None


def set_storage(storage: MemoryStorage) -> None:
    """Set the global storage instance and initialize quota service if enabled."""
    global _storage, _quota_service
    _storage = storage

    # Initialize quota service if enabled in config
    if settings.quota.enabled:
        logger.info("Initializing QuotaService (quota enforcement enabled)")
        _quota_service = QuotaService(
            storage=storage,
            settings=settings.quota,
        )
    else:
        logger.info("QuotaService disabled (quota.enabled=False)")
        _quota_service = None


def get_storage() -> MemoryStorage:
    """Get the global storage instance."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    return _storage


def get_memory_service(storage: MemoryStorage = Depends(get_storage)) -> MemoryService:
    """Get a MemoryService instance with the configured storage backend and quota service."""
    return MemoryService(
        storage,
        graph_client=get_graph_client(),
        write_queue=get_write_queue(),
        quota_service=_quota_service,
    )


async def create_storage_backend() -> MemoryStorage:
    """
    Create and initialize storage backend for web interface based on configuration.

    Returns:
        Initialized storage backend
    """
    from ..config import DATABASE_PATH
    from ..storage.factory import create_storage_instance

    logger.info("Creating storage backend for web interface...")

    # Use shared factory with DATABASE_PATH for web interface
    return await create_storage_instance(DATABASE_PATH)
