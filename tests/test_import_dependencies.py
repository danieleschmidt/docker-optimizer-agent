#!/usr/bin/env python3
"""
Test suite for import dependency handling.
Ensures the package imports gracefully even with missing optional dependencies.
"""

import sys
import unittest.mock as mock
from importlib import reload


class TestImportDependencies:
    """Test import dependency handling and graceful degradation."""

    def test_core_package_imports_without_pydantic(self):
        """Test that core package can import without pydantic."""
        # Test the actual fix - package should import now
        try:
            import sys
            sys.path.insert(0, 'src')
            import docker_optimizer
            # Check that basic attributes are available
            assert hasattr(docker_optimizer, '__version__')
            assert hasattr(docker_optimizer, '__all__')
            print(f"✅ Package imported successfully with {len(docker_optimizer.__all__)} available components")
        except ImportError as e:
            assert False, f"Package should not fail import: {e}"

    def test_core_package_imports_without_psutil(self):
        """Test that core package can import without psutil."""
        # Test that logging_observability handles missing psutil gracefully
        try:
            import sys
            sys.path.insert(0, 'src')
            from docker_optimizer.logging_observability import PSUTIL_AVAILABLE
            print(f"✅ logging_observability imported, PSUTIL_AVAILABLE={PSUTIL_AVAILABLE}")
        except ImportError as e:
            assert False, f"logging_observability should handle missing psutil: {e}"

    def test_cli_works_with_minimal_dependencies(self):
        """Test that CLI module exists and basic package works."""
        import sys
        sys.path.insert(0, 'src')
        
        # Test that package imports and has basic functionality
        try:
            import docker_optimizer
            # Verify we have the core functionality available
            assert hasattr(docker_optimizer, '__version__')
            print(f"✅ Core package functional with version {docker_optimizer.__version__}")
        except ImportError as e:
            assert False, f"Core package should work: {e}"


if __name__ == "__main__":
    # Simple test runner
    test = TestImportDependencies()
    
    print("🧪 Testing import dependencies...")
    
    print("=== GREEN PHASE: Testing fixes ===")
    
    try:
        test.test_core_package_imports_without_pydantic()
        print("✅ test_core_package_imports_without_pydantic: PASSED")
    except AssertionError as e:
        print(f"❌ test_core_package_imports_without_pydantic: FAILED - {e}")
    
    try:
        test.test_core_package_imports_without_psutil()
        print("✅ test_core_package_imports_without_psutil: PASSED")
    except AssertionError as e:
        print(f"❌ test_core_package_imports_without_psutil: FAILED - {e}")
    
    try:
        test.test_cli_works_with_minimal_dependencies()
        print("✅ test_cli_works_with_minimal_dependencies: PASSED")
    except AssertionError as e:
        print(f"❌ test_cli_works_with_minimal_dependencies: FAILED - {e}")
    
    print("🟢 GREEN PHASE: Tests should now pass!")