"""Unit test conftest — clear CacheKit L1 caches between tests.

CacheKit's L1CacheManager holds all L1Cache instances globally. When
``backend=None`` (L1-only mode used in CI/tests), cached results from one
test bleed into the next. ``clear_all()`` nukes every L1Cache instance
registered with the global manager — no closure-walking needed.
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_cachekit_l1():
    """Clear all CacheKit L1 caches before and after every unit test."""

    def _clear():
        try:
            from cachekit.l1_cache import get_l1_cache_manager
        except ImportError:
            return
        get_l1_cache_manager().clear_all()

    _clear()
    yield
    _clear()
