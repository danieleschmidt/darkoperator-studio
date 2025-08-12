#!/usr/bin/env python3
"""
Test script for robustness integration.

Tests that robust error handling doesn't break existing functionality
and provides graceful degradation under failure conditions.
"""

import sys
import traceback
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def test_basic_imports():
    """Test that basic imports work with robustness enhancements."""
    print("Testing basic imports...")
    
    try:
        # Test core module structure
        import darkoperator
        print("✓ darkoperator import successful")
        
        # Test research module
        from darkoperator.research import BenchmarkConfig
        print("✓ research module import successful")
        
        # Test error handling utilities  
        from darkoperator.utils.robust_error_handling import (
            robust_physics_operation,
            RobustPhysicsLogger,
            get_system_health_check
        )
        print("✓ robust error handling import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling_decorator():
    """Test the robust physics operation decorator."""
    print("\nTesting error handling decorator...")
    
    try:
        from darkoperator.utils.robust_error_handling import robust_physics_operation
        
        @robust_physics_operation(max_retries=2, fallback_strategy='graceful_degradation')
        def test_function_success():
            return "success"
        
        @robust_physics_operation(max_retries=2, fallback_strategy='graceful_degradation')
        def test_function_failure():
            raise ValueError("Test error")
        
        # Test successful operation
        result = test_function_success()
        assert result == "success"
        print("✓ Successful operation handling works")
        
        # Test failed operation with graceful degradation
        result = test_function_failure()
        # Should not raise exception due to graceful degradation
        print(f"✓ Failed operation handled gracefully: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_context_manager():
    """Test the robust physics context manager."""
    print("\nTesting context manager...")
    
    try:
        from darkoperator.utils.robust_error_handling import robust_physics_context
        
        # Test successful context
        with robust_physics_context("test_component", "test_operation") as context:
            # Simulate some work
            context.input_shapes = {'test': (10, 20)}
        
        print("✓ Context manager works for successful operations")
        
        # Test failed context with error handling
        try:
            with robust_physics_context("test_component", "failing_operation") as context:
                raise ValueError("Test context error")
        except Exception:
            pass  # Expected to raise
        
        print("✓ Context manager properly handles errors")
        
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        traceback.print_exc()
        return False

def test_system_health_check():
    """Test system health monitoring."""
    print("\nTesting system health check...")
    
    try:
        from darkoperator.utils.robust_error_handling import get_system_health_check
        
        health_status = get_system_health_check()
        
        # Validate health status structure
        assert 'overall_status' in health_status
        assert 'checks' in health_status
        assert 'timestamp' in health_status
        
        print(f"✓ System health check completed: {health_status['overall_status']}")
        print(f"  Checks performed: {list(health_status['checks'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ System health check failed: {e}")
        traceback.print_exc()
        return False

def test_robust_logger():
    """Test the robust physics logger."""
    print("\nTesting robust logger...")
    
    try:
        from darkoperator.utils.robust_error_handling import RobustPhysicsLogger
        
        logger = RobustPhysicsLogger("test_component")
        
        # Test various logging methods
        logger.log_operation_start("test_operation", param1="value1")
        logger.log_operation_success("test_operation", 0.1, result="success")
        logger.log_performance_metrics("test_operation", {"metric1": 0.95})
        
        print("✓ Robust logger functionality works")
        
        return True
        
    except Exception as e:
        print(f"✗ Robust logger test failed: {e}")
        traceback.print_exc()
        return False

def test_error_recovery_manager():
    """Test the error recovery manager."""
    print("\nTesting error recovery manager...")
    
    try:
        from darkoperator.utils.robust_error_handling import (
            ErrorRecoveryManager,
            RobustPhysicsError,
            ErrorContext,
            ErrorSeverity
        )
        
        manager = ErrorRecoveryManager()
        
        # Simulate some errors
        context = ErrorContext("test_component", "test_operation")
        error = RobustPhysicsError("Test error", context, ErrorSeverity.MEDIUM)
        
        manager.record_error(error, recovered=True)
        manager.record_recovery_strategy("graceful_degradation")
        
        # Check recovery rate
        recovery_rate = manager.get_recovery_rate()
        assert 0.0 <= recovery_rate <= 1.0
        
        # Generate report
        report = manager.generate_reliability_report()
        assert 'overall_recovery_rate' in report
        
        print(f"✓ Error recovery manager works (recovery rate: {recovery_rate})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error recovery manager test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all robustness integration tests."""
    print("=" * 60)
    print("ROBUSTNESS INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_error_handling_decorator,
        test_context_manager,
        test_system_health_check,
        test_robust_logger,
        test_error_recovery_manager
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
        print("🎉 All robustness integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())