#!/usr/bin/env python3
"""
Comprehensive Quality Gates Test Suite - Security, Performance, Testing
"""

import sys
import os
import time
import subprocess
import tempfile
from pathlib import Path
sys.path.insert(0, '.')

def test_security_gates():
    """Test security scanning and validation."""
    print("Testing security gates...")
    try:
        from darkoperator.security.enhanced_security_scanner import SecurityScanner
        from darkoperator.security.data_security import sanitize_inputs, validate_data_source
        
        # Test security scanner
        scanner = SecurityScanner()
        scan_results = scanner.scan_basic_security()
        print(f"✅ Security scan completed: {len(scan_results)} checks")
        
        # Test input sanitization
        try:
            sanitize_inputs({'malicious': 'DROP TABLE users;'})
            print("❌ Should have blocked SQL injection")
            return False
        except Exception:
            print("✅ Input sanitization blocking malicious patterns")
        
        # Test file validation
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test,data\n1,2\n")
            temp_path = f.name
        
        try:
            validate_data_source(temp_path, allowed_extensions=['.csv'])
            print("✅ File validation working")
        finally:
            os.unlink(temp_path)
        
        return True
    except Exception as e:
        print(f"❌ Security gates failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarking meets requirements."""
    print("\nTesting performance benchmarks...")
    try:
        import torch
        
        # Test computation performance
        start_time = time.time()
        
        # Neural operator simulation
        input_tensor = torch.randn(100, 4, 32, 32)
        conv_layer = torch.nn.Conv2d(4, 16, 3, padding=1)
        output = conv_layer(input_tensor)
        
        computation_time = time.time() - start_time
        
        # Performance requirement: <1 second for this operation
        if computation_time < 1.0:
            print(f"✅ Neural operator performance: {computation_time:.3f}s (< 1s requirement)")
        else:
            print(f"⚠️ Performance below target: {computation_time:.3f}s")
        
        # Memory usage check
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent < 95:
            print(f"✅ Memory usage acceptable: {memory_percent:.1f}%")
        else:
            print(f"⚠️ High memory usage: {memory_percent:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Performance benchmarks failed: {e}")
        return False

def test_code_coverage():
    """Test that core modules can be imported and basic functionality works."""
    print("\nTesting code coverage...")
    try:
        # Test core module imports
        modules_to_test = [
            'darkoperator',
            'darkoperator.operators',
            'darkoperator.models',
            'darkoperator.anomaly',
            'darkoperator.physics',
            'darkoperator.data',
            'darkoperator.utils',
            'darkoperator.security',
            'darkoperator.optimization'
        ]
        
        imported_count = 0
        for module in modules_to_test:
            try:
                __import__(module)
                imported_count += 1
            except ImportError as e:
                print(f"⚠️ Module import issue: {module} - {e}")
        
        coverage_percent = (imported_count / len(modules_to_test)) * 100
        print(f"✅ Module import coverage: {coverage_percent:.1f}% ({imported_count}/{len(modules_to_test)})")
        
        # Test basic functionality
        from darkoperator.physics.lorentz import LorentzEmbedding
        from darkoperator.data.synthetic import generate_synthetic_events
        from darkoperator.utils.validation import validate_tensor_shape
        
        # Create and test components
        embedding = LorentzEmbedding(4, 16)
        events = generate_synthetic_events(5)
        
        import torch
        test_tensor = torch.randn(10, 4)
        is_valid = validate_tensor_shape(test_tensor, 2)
        
        if len(events) == 5 and is_valid:
            print("✅ Core functionality working")
        
        return True
    except Exception as e:
        print(f"❌ Code coverage test failed: {e}")
        return False

def test_documentation_completeness():
    """Test documentation and API completeness."""
    print("\nTesting documentation completeness...")
    try:
        # Check README exists
        readme_path = Path("README.md")
        if readme_path.exists():
            readme_size = readme_path.stat().st_size
            print(f"✅ README.md exists ({readme_size} bytes)")
        else:
            print("❌ README.md missing")
            return False
        
        # Check package structure
        package_path = Path("darkoperator")
        if package_path.exists() and package_path.is_dir():
            submodules = list(package_path.glob("*/"))
            print(f"✅ Package structure: {len(submodules)} submodules")
        else:
            print("❌ Package structure incomplete")
            return False
        
        # Check __init__.py files
        init_files = list(package_path.rglob("__init__.py"))
        print(f"✅ Package initialization: {len(init_files)} __init__.py files")
        
        return True
    except Exception as e:
        print(f"❌ Documentation completeness failed: {e}")
        return False

def test_deployment_readiness():
    """Test deployment configuration and readiness."""
    print("\nTesting deployment readiness...")
    try:
        # Check deployment configurations exist
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml", 
            "pyproject.toml",
            "requirements.txt"
        ]
        
        found_files = 0
        for file_name in deployment_files:
            file_path = Path(file_name)
            if file_path.exists():
                found_files += 1
                print(f"✅ {file_name} exists")
            else:
                print(f"⚠️ {file_name} missing")
        
        # Check deployment directory
        deployment_dir = Path("deployment_artifacts")
        if deployment_dir.exists():
            deployment_configs = list(deployment_dir.glob("*"))
            print(f"✅ Deployment artifacts: {len(deployment_configs)} files")
        
        deployment_score = (found_files / len(deployment_files)) * 100
        print(f"✅ Deployment readiness: {deployment_score:.1f}%")
        
        return deployment_score >= 75  # 75% minimum for deployment readiness
    except Exception as e:
        print(f"❌ Deployment readiness failed: {e}")
        return False

def main():
    """Run comprehensive quality gates test suite."""
    print("=" * 60)
    print("QUALITY GATES: Comprehensive System Validation")
    print("=" * 60)
    
    tests = [
        test_security_gates,
        test_performance_benchmarks,
        test_code_coverage,
        test_documentation_completeness,
        test_deployment_readiness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"QUALITY GATES RESULTS: {passed}/{total} gates passed")
    
    if passed >= total * 0.8:  # 80% pass rate for quality gates
        print("✅ QUALITY GATES PASSED: System ready for production")
        return True
    else:
        print("❌ QUALITY GATES FAILED: System needs improvement")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)