"""Unit test conftest — clear CacheKit L1 caches between tests.

CacheKit's ``ainvalidate_cache()`` is key-specific; it does NOT clear all
entries for a function.  When ``backend=None`` (L1-only mode), cached
embeddings from one test can bleed into the next.  This fixture reaches
into the wrapper closure and calls ``L1Cache.clear()`` directly.
"""

import pytest


def _get_l1(fn):
    """Extract the L1Cache instance from a CacheKit wrapper's closure."""
    if not hasattr(fn, "__closure__") or fn.__closure__ is None:
        return None
    for cell in fn.__closure__:
        try:
            v = cell.cell_contents
            if hasattr(v, "__class__") and v.__class__.__name__ == "L1Cache":
                return v
        except ValueError:
            pass
    return None


@pytest.fixture(autouse=True)
def _clear_cachekit_l1():
    """Clear all CacheKit L1 caches before and after every unit test."""
    def _clear():
        try:
            from mcp_memory_service.services.memory_service import (
                _cached_corpus_count,
                _cached_embed,
                _cached_extract_keywords,
                _cached_fetch_all_tags,
            )
            for fn in (_cached_fetch_all_tags, _cached_corpus_count, _cached_embed, _cached_extract_keywords):
                l1 = _get_l1(fn)
                if l1 is not None:
                    l1.clear()
        except Exception:  # noqa: BLE001
            pass

    _clear()
    yield
    _clear()
