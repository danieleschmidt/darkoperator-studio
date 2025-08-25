#!/usr/bin/env python3
"""
Generation 2 Robustness Testing Suite

Tests enhanced error handling, monitoring, and resilience features
implemented in Generation 2 of the TERRAGON SDLC v4.0.
"""

import sys
import time
import json
import traceback
from pathlib import Path

# Add darkoperator to path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_error_handling():
    """Test enhanced error handling capabilities."""
    print("🛡️ Testing Enhanced Error Handling...")
    
    try:
        from darkoperator.utils.enhanced_error_handling import (
            RobustOperatorWrapper, robust_operation, HealthMonitor,
            get_health_monitor, validate_input_data, ValidationError
        )
        
        # Test 1: Health monitoring
        print("  📊 Testing health monitoring...")
        health_monitor = get_health_monitor()
        health_results = health_monitor.check_health()
        
        print(f"  ✓ Health check completed: {len(health_results)} components")
        for component, status in health_results.items():
            print(f"    - {component}: {status['status']}")
        
        health_score = health_monitor.get_system_health_score()
        print(f"  ✓ System health score: {health_score:.2%}")
        
        # Test 2: Robust decorator
        print("  🔄 Testing robust operation decorator...")
        
        @robust_operation(max_retries=2, timeout=5.0)
        def test_robust_function(should_fail=False, delay=0):
            if delay:
                time.sleep(delay)
            if should_fail:
                raise ValueError("Intentional test error")
            return "Success!"
        
        # Test successful operation
        result = test_robust_function(should_fail=False)
        print(f"  ✓ Robust operation succeeded: {result}")
        
        print("  ✅ Enhanced error handling tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_advanced_monitoring():
    """Test advanced monitoring capabilities.""" 
    print("📊 Testing Advanced Monitoring...")
    
    try:
        from darkoperator.monitoring.advanced_monitoring import (
            MetricsCollector, PerformanceProfiler, AlertManager,
            get_metrics_collector, get_profiler, get_alert_manager
        )
        
        # Test 1: Metrics collection
        print("  📈 Testing metrics collection...")
        collector = get_metrics_collector()
        
        # Record various metric types
        collector.increment_counter("test_operations", 5.0)
        collector.set_gauge("system_load", 0.75)
        collector.record_histogram("response_time", 23.5)
        
        print("  ✓ Metrics collection working")
        
        # Test 2: Performance profiling
        print("  ⚡ Testing performance profiling...")
        profiler = get_profiler()
        
        @profiler.profile_operation("test_computation")
        def sample_computation():
            result = sum(i**2 for i in range(100))
            time.sleep(0.01)
            return result
        
        result = sample_computation()
        print(f"  ✓ Profiled operation result: {result}")
        
        print("  ✅ Advanced monitoring tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Monitoring test failed: {e}")
        traceback.print_exc()
        return False


def run_generation2_tests():
    """Run all Generation 2 robustness tests."""
    print("🚀 TERRAGON SDLC v4.0 - Generation 2 Robustness Testing")
    print("=" * 60)
    
    results = {}
    start_time = time.time()
    
    # Run test suites
    test_suites = [
        ("Enhanced Error Handling", test_enhanced_error_handling),
        ("Advanced Monitoring", test_advanced_monitoring)
    ]
    
    passed_tests = 0
    total_tests = len(test_suites)
    
    for suite_name, test_func in test_suites:
        print(f"\n🧪 Running {suite_name} Tests...")
        try:
            result = test_func()
            results[suite_name] = {"passed": result, "error": None}
            if result:
                passed_tests += 1
                print(f"✅ {suite_name} tests PASSED")
            else:
                print(f"❌ {suite_name} tests FAILED")
        except Exception as e:
            results[suite_name] = {"passed": False, "error": str(e)}
            print(f"💥 {suite_name} tests CRASHED: {e}")
    
    # Generate summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📋 GENERATION 2 ROBUSTNESS TEST SUMMARY")
    print("=" * 60)
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    print(f"Execution Time: {duration:.2f} seconds")
    
    # Save results
    test_results = {
        "generation": 2,
        "test_type": "robustness",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": duration,
        "summary": {
            "total_suites": total_tests,
            "passed_suites": passed_tests,
            "failed_suites": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests
        },
        "detailed_results": results
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "generation2_robustness_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Assessment
    if passed_tests == total_tests:
        print("🎉 GENERATION 2 COMPLETE: All robustness tests passed!")
        print("🚀 Ready for Generation 3 optimization implementation.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  GENERATION 2 MOSTLY COMPLETE: Minor issues detected.")
    else:
        print("❌ GENERATION 2 NEEDS WORK: Major robustness issues detected.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    try:
        success = run_generation2_tests()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n❌ Testing interrupted by user")
        exit_code = 2
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        exit_code = 3
    
    print(f"\n🏁 Testing completed with exit code: {exit_code}")
    sys.exit(exit_code)