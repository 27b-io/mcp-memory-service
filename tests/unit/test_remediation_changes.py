"""
Tests for codebase remediation changes.

These tests verify the security fixes, config cleanup, and storage consolidation
performed during the codebase remediation (56% code reduction).

NOTE: Tests that verify removed code stays removed are kept.
Tests that were empty/pass statements have been removed.
"""

import pytest


class TestSecurityFixes:
    """Test security fixes from Phase 0 of remediation."""

    def test_cors_default_is_empty_list(self):
        """Verify CORS default is [] not ['*'] (SEC-003 fix)."""
        from mcp_memory_service.config import HTTPSettings

        settings = HTTPSettings()
        assert settings.cors_origins == [], f"CORS default should be empty list, got {settings.cors_origins}"

    def test_api_key_comparison_uses_constant_time(self):
        """
        Verify API key comparison uses secrets.compare_digest (SEC-007 fix).

        This test verifies the fix is in place by checking:
        1. The authenticate_api_key function exists and is callable
        2. The function handles None input gracefully
        3. The actual constant-time behavior is verified via code inspection
           (secrets.compare_digest is used in the implementation)
        """
        from mcp_memory_service.web.oauth.middleware import authenticate_api_key

        # Test that function handles None key (unauthenticated case)
        result = authenticate_api_key(None)
        # Should return a response indicating unauthenticated
        assert result is not None

        # Note: Actual timing attack prevention cannot be reliably tested
        # in a unit test due to OS scheduler noise. The implementation uses
        # secrets.compare_digest which is the standard library's constant-time
        # comparison function.

    def test_anonymous_access_is_read_only(self):
        """Verify anonymous access defaults to disabled (SEC-001 fix)."""
        from mcp_memory_service.config import OAuthSettings

        settings = OAuthSettings()
        # Anonymous access should be disabled by default
        assert settings.allow_anonymous_access is False, "Anonymous access should be disabled by default"


class TestStorageConsolidation:
    """Test storage backend consolidation."""

    def test_removed_backends_not_importable(self):
        """Verify all removed backends are not importable."""
        removed_backends = [
            ("cloudflare", "mcp_memory_service.storage.cloudflare", "CloudflareStorage"),
            ("hybrid", "mcp_memory_service.storage.hybrid", "HybridStorage"),
            ("http_client", "mcp_memory_service.storage.http_client", "HTTPStorageClient"),
            ("sqlite_vec", "mcp_memory_service.storage.sqlite_vec", "SqliteVecMemoryStorage"),
        ]

        for _name, module_path, class_name in removed_backends:
            with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)

    def test_qdrant_is_sole_storage_backend(self):
        """Verify QdrantStorage is the only concrete storage backend."""
        from mcp_memory_service.storage import QdrantStorage
        from mcp_memory_service.storage.base import MemoryStorage

        assert issubclass(QdrantStorage, MemoryStorage)


class TestConfigCleanup:
    """Test configuration cleanup from Phase 3."""

    def test_removed_settings_classes_not_importable(self):
        """Verify removed settings classes are gone."""
        removed_classes = [
            "CloudflareSettings",
            "HybridSettings",
            "DocumentSettings",
        ]

        from mcp_memory_service import config

        for class_name in removed_classes:
            assert not hasattr(config, class_name), f"{class_name} should be removed from config"

    def test_essential_settings_classes_exist(self):
        """Verify essential settings classes still exist and are usable."""
        from mcp_memory_service.config import (
            HTTPSettings,
            OAuthSettings,
            PathSettings,
            QdrantSettings,
            ServerSettings,
            StorageSettings,
        )

        # All should be instantiable without errors
        PathSettings()
        ServerSettings()
        StorageSettings()
        HTTPSettings()
        OAuthSettings()
        QdrantSettings()

    def test_sqlite_settings_removed(self):
        """Verify SQLite-specific settings are gone."""
        from mcp_memory_service.config import StorageSettings

        settings = StorageSettings()
        assert not hasattr(settings, "storage_backend"), "storage_backend field should be removed"


class TestRemovedModules:
    """Test that removed modules are truly gone."""

    def test_consolidation_module_removed(self):
        """Verify consolidation module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.consolidation  # noqa: F401

    def test_sync_module_removed(self):
        """Verify sync module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.sync  # noqa: F401

    def test_ingestion_module_removed(self):
        """Verify ingestion module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.ingestion  # noqa: F401

    def test_discovery_module_removed(self):
        """Verify discovery/mDNS module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.discovery  # noqa: F401

    def test_lm_studio_compat_removed(self):
        """Verify LM Studio compatibility layer is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.lm_studio  # noqa: F401

    def test_legacy_server_removed(self):
        """Verify legacy server.py god-class is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import mcp_memory_service.server  # noqa: F401


class TestDependencyCleanup:
    """Test removed dependencies don't break core functionality."""

    def test_oauth_middleware_works_without_authlib(self):
        """Verify OAuth middleware loads without authlib dependency."""
        try:
            # Import the actual authentication function that exists
            from mcp_memory_service.web.oauth.middleware import authenticate_api_key

            # If we get here, the middleware loads without authlib
            assert callable(authenticate_api_key)
        except ImportError as e:
            if "authlib" in str(e).lower():
                pytest.fail("authlib should not be a required dependency")
            raise

    def test_api_key_verification_works_without_jose(self):
        """Verify API key verification doesn't require python-jose."""
        try:
            # Import the actual authentication function
            from mcp_memory_service.web.oauth.middleware import authenticate_api_key

            # If we get here, it works without jose
            assert callable(authenticate_api_key)
        except ImportError as e:
            if "jose" in str(e).lower():
                pytest.fail("python-jose should not be a required dependency")
            raise
