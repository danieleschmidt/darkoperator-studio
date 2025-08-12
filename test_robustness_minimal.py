#!/usr/bin/env python3
"""
Minimal test script for robustness integration without external dependencies.

Tests core robustness functionality without requiring torch/numpy.
"""

import sys
import traceback
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def test_error_handling_utilities():
    """Test error handling utilities directly."""
    print("Testing error handling utilities...")
    
    try:
        # Import error handling directly
        sys.path.insert(0, str(repo_root / 'darkoperator' / 'utils'))
        from robust_error_handling import (
            robust_physics_operation,
            RobustPhysicsLogger,
            ErrorRecoveryManager,
            RobustPhysicsError,
            ErrorContext,
            ErrorSeverity
        )
        
        print("✓ Error handling imports successful")
        
        # Test error context creation
        context = ErrorContext("test_component", "test_operation")
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        print("✓ Error context creation works")
        
        # Test error severity enum
        assert ErrorSeverity.LOW == ErrorSeverity.LOW
        assert ErrorSeverity.CRITICAL != ErrorSeverity.LOW
        print("✓ Error severity enum works")
        
        # Test robust logger creation
        logger = RobustPhysicsLogger("test_component")
        assert logger.component_name == "test_component"
        print("✓ Robust logger creation works")
        
        # Test error recovery manager
        manager = ErrorRecoveryManager()
        assert manager.get_recovery_rate() == 1.0  # No errors yet
        print("✓ Error recovery manager works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling utilities test failed: {e}")
        traceback.print_exc()
        return False

def test_decorator_functionality():
    """Test the robust operation decorator."""
    print("\nTesting decorator functionality...")
    
    try:
        sys.path.insert(0, str(repo_root / 'darkoperator' / 'utils'))
        from robust_error_handling import robust_physics_operation
        
        # Test successful operation
        @robust_physics_operation(max_retries=1, fallback_strategy='graceful_degradation', log_errors=False)
        def successful_operation(x):
            return x * 2
        
        result = successful_operation(5)
        assert result == 10
        print("✓ Successful operation works")
        
        # Test failing operation with graceful degradation
        @robust_physics_operation(max_retries=1, fallback_strategy='graceful_degradation', log_errors=False)
        def failing_operation():
            raise ValueError("Test error")
        
        result = failing_operation()
        # Should not raise exception due to graceful degradation
        print(f"✓ Failing operation handled gracefully: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Decorator functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("\nTesting memory monitoring...")
    
    try:
        sys.path.insert(0, str(repo_root / 'darkoperator' / 'utils'))
        from robust_error_handling import _get_memory_usage, _get_device_info
        
        # Test memory usage collection
        memory_info = _get_memory_usage()
        assert 'cpu_memory_percent' in memory_info
        assert 'cpu_memory_available_gb' in memory_info
        print("✓ Memory usage monitoring works")
        
        # Test device info collection
        device_info = _get_device_info()
        assert 'cpu_count' in device_info
        assert 'cuda_available' in device_info
        print("✓ Device info collection works")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_health_check():
    """Test system health check."""
    print("\nTesting system health check...")
    
    try:
        sys.path.insert(0, str(repo_root / 'darkoperator' / 'utils'))
        from robust_error_handling import get_system_health_check
        
        health_status = get_system_health_check()
        
        # Validate structure
        assert 'overall_status' in health_status
        assert 'checks' in health_status
        assert 'timestamp' in health_status
        
        # Check that we have expected health checks
        checks = health_status['checks']
        assert 'memory' in checks
        assert 'error_recovery' in checks
        
        print(f"✓ System health check works: {health_status['overall_status']}")
        print(f"  Checks: {list(checks.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ System health check failed: {e}")
        traceback.print_exc()
        return False

def test_error_classification():
    """Test error severity classification."""
    print("\nTesting error classification...")
    
    try:
        sys.path.insert(0, str(repo_root / 'darkoperator' / 'utils'))
        from robust_error_handling import (
            _classify_error_severity,
            ErrorContext,
            ErrorSeverity,
            ConservationViolationError,
            QuantumCircuitError
        )
        
        context = ErrorContext("test", "test")
        
        # Test memory error classification
        memory_error = RuntimeError("CUDA out of memory")
        severity = _classify_error_severity(memory_error, context)
        assert severity == ErrorSeverity.CRITICAL
        print("✓ Memory error classified as CRITICAL")
        
        # Test physics error classification
        physics_error = ConservationViolationError("Energy not conserved", context)
        severity = _classify_error_severity(physics_error, context)
        assert severity == ErrorSeverity.MEDIUM
        print("✓ Physics error classified as MEDIUM")
        
        # Test quantum error classification
        quantum_error = QuantumCircuitError("Quantum circuit failed", context)
        severity = _classify_error_severity(quantum_error, context)
        assert severity == ErrorSeverity.HIGH
        print("✓ Quantum error classified as HIGH")
        
        return True
        
    except Exception as e:
        print(f"✗ Error classification test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run minimal robustness tests."""
    print("=" * 60)
    print("MINIMAL ROBUSTNESS INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_error_handling_utilities,
        test_decorator_functionality,
        test_memory_monitoring,
        test_health_check,
        test_error_classification
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Unexpected error in {test_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All minimal robustness tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())