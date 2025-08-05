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


def test_security_functions():
    """Test security functions."""
    from darkoperator.utils.validation_basic import sanitize_file_path, ValidationError
    
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
    from darkoperator.config.settings_basic import Settings, validate_config
    
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


def test_code_quality():
    """Test basic code quality metrics."""
    import glob
    
    python_files = glob.glob("darkoperator/**/*.py", recursive=True)
    print(f"Found {len(python_files)} Python files")
    
    total_lines = 0
    for file_path in python_files:
        with open(file_path, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
    
    print(f"Total lines of code: {total_lines:,}")
    assert total_lines > 1000, "Should have substantial codebase"
    
    # Check for basic documentation
    documented_files = 0
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if '"""' in content or "'''" in content:
                documented_files += 1
    
    documentation_ratio = documented_files / len(python_files)
    print(f"Documentation ratio: {documentation_ratio:.1%}")
    assert documentation_ratio > 0.5, "Should have good documentation coverage"


def test_project_completeness():
    """Test that project has all expected components."""
    
    # Check key files exist
    required_files = [
        'setup.py',
        'pyproject.toml', 
        'requirements.txt',
        'environment.yml',
        'README.md',
        'LICENSE'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file {file_path} missing"
        print(f"✓ {file_path}")
    
    # Check package modules
    key_modules = [
        'darkoperator/__init__.py',
        'darkoperator/operators/__init__.py',
        'darkoperator/anomaly/__init__.py',
        'darkoperator/models/__init__.py',
        'darkoperator/cli/main.py'
    ]
    
    for module_path in key_modules:
        assert os.path.exists(module_path), f"Key module {module_path} missing"
        print(f"✓ {module_path}")


def run_all_tests():
    """Run all tests."""
    print("Running comprehensive quality gate tests...")
    print("=" * 60)
    
    tests = [
        test_package_structure,
        test_security_functions,
        test_configuration,
        test_file_contents,
        test_code_quality,
        test_project_completeness
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
    print(f"QUALITY GATE RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL QUALITY GATES PASSED!")
        return True
    else:
        print("❌ Some quality gates failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)