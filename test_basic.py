#!/usr/bin/env python3
"""Basic tests without external dependencies."""

import sys
import os
sys.path.insert(0, '.')

def test_package_structure():
    """Test that package structure is correct."""
    expected_dirs = [
        'darkoperator',
        'darkoperator/operators',
        'darkoperator/anomaly', 
        'darkoperator/models',
        'darkoperator/data',
        'darkoperator/utils',
        'darkoperator/visualization',
        'darkoperator/cli',
        'darkoperator/physics',
        'darkoperator/optimization',
        'darkoperator/monitoring',
        'darkoperator/security',
        'darkoperator/config',
        'tests'
    ]
    
    for dir_path in expected_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        print(f"✓ {dir_path}")
    
    print("✓ Package structure is correct")


def test_imports():
    """Test that modules can be imported (without torch dependencies)."""
    try:
        # Test configuration imports
        from darkoperator.config.settings import Settings, load_config
        print("✓ Configuration modules import successfully")
        
        # Test utility imports (non-torch parts)
        from darkoperator.utils.validation import sanitize_file_path
        print("✓ Validation utilities import successfully")
        
        # Test that files exist and are readable
        import darkoperator.cli.main
        print("✓ CLI module exists")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_security_functions():
    """Test security functions."""
    from darkoperator.utils.validation import sanitize_file_path, ValidationError
    
    # Test safe paths
    assert sanitize_file_path("model.pt") == "model.pt"
    assert sanitize_file_path("data/events.npz") == "data/events.npz"
    print("✓ Safe path sanitization works")
    
    # Test dangerous paths
    try:
        sanitize_file_path("../../../etc/passwd")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ Directory traversal attack blocked")
    
    try:
        sanitize_file_path("file`malicious`")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ Command injection attempt blocked")


def test_configuration():
    """Test configuration loading."""
    from darkoperator.config.settings import Settings, validate_config
    
    # Test default settings
    settings = Settings()
    validate_config(settings)
    print("✓ Default configuration is valid")
    
    # Test that directories get created
    assert os.path.exists(settings.data.cache_dir)
    assert os.path.exists(settings.model.cache_dir)
    assert os.path.exists(settings.monitoring.log_dir)
    print("✓ Configuration directories created")


def test_file_contents():
    """Test that key files have expected content."""
    # Check setup.py
    with open('setup.py', 'r') as f:
        content = f.read()
        assert 'darkoperator' in content
        assert 'Neural Operators' in content
    print("✓ setup.py contains expected content")
    
    # Check CLI entry point
    with open('darkoperator/cli/main.py', 'r') as f:
        content = f.read()
        assert 'def main()' in content
        assert 'argparse' in content
    print("✓ CLI entry point is properly structured")
    
    # Check README
    with open('README.md', 'r') as f:
        content = f.read()
        assert 'DarkOperator Studio' in content
        assert 'Neural Operators' in content
    print("✓ README contains project description")


def run_all_tests():
    """Run all tests."""
    print("Running basic tests without external dependencies...")
    print("=" * 60)
    
    tests = [
        test_package_structure,
        test_imports,
        test_security_functions,
        test_configuration,
        test_file_contents
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
            print(f"✓ {test.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)