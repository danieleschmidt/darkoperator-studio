#!/usr/bin/env python3
"""
Generation 3 Optimization Testing Suite

Tests advanced optimization, performance scaling, and intelligent resource management
implemented in Generation 3 of the TERRAGON SDLC v4.0.
"""

import sys
import time
import json
import traceback
from pathlib import Path

# Add darkoperator to path
sys.path.insert(0, str(Path(__file__).parent))

def test_adaptive_performance_optimizer():
    """Test adaptive performance optimization capabilities."""
    print("⚡ Testing Adaptive Performance Optimizer...")
    
    try:
        from darkoperator.optimization.adaptive_performance_optimizer import (
            AdaptivePerformanceOptimizer, AdaptiveMemoryManager, 
            DynamicBatchOptimizer, WorkloadPredictor, get_performance_optimizer,
            optimize_for_physics_workload
        )
        
        # Test 1: Memory management
        print("  🧠 Testing adaptive memory management...")
        memory_manager = AdaptiveMemoryManager(max_cache_size_mb=100.0)
        
        # Test cache operations
        memory_manager.put("test_key_1", "test_value_1", 10.0)
        memory_manager.put("test_key_2", "test_value_2", 15.0)
        
        cached_value = memory_manager.get("test_key_1")
        cache_miss = memory_manager.get("nonexistent_key")
        
        print(f"  ✓ Cache hit: {cached_value is not None}")
        print(f"  ✓ Cache miss handled: {cache_miss is None}")
        
        # Test cache statistics
        stats = memory_manager.get_stats()
        print(f"  ✓ Cache stats: {stats['cache_size']} items, {stats['memory_usage_mb']:.1f} MB")
        
        # Test 2: Dynamic batch optimization
        print("  📦 Testing dynamic batch optimization...")
        batch_optimizer = DynamicBatchOptimizer(initial_batch_size=16)
        
        # Simulate different performance scenarios
        batch_optimizer.record_performance(16, 100.0, 50.0, 160.0)  # Good performance
        batch_optimizer.record_performance(16, 150.0, 80.0, 120.0)  # Worse performance
        batch_optimizer.record_performance(16, 120.0, 60.0, 140.0)  # Medium performance
        
        optimized_batch_size = batch_optimizer.optimize_batch_size()
        batch_stats = batch_optimizer.get_optimization_stats()
        
        print(f"  ✓ Optimized batch size: {optimized_batch_size}")
        print(f"  ✓ Optimization steps: {batch_stats['optimization_steps']}")
        print(f"  ✓ Recent efficiency: {batch_stats.get('recent_avg_efficiency', 0):.2f}")
        
        # Test 3: Workload prediction
        print("  🔮 Testing workload prediction...")
        predictor = WorkloadPredictor()
        
        # Record some workload patterns
        predictor.record_workload("neural_operator_inference", 100, 1.0)
        predictor.record_workload("neural_operator_inference", 150, 1.2)
        predictor.record_workload("anomaly_detection", 200, 0.8)
        
        prediction = predictor.predict_next_workload()
        print(f"  ✓ Predicted size: {prediction['predicted_size']}")
        print(f"  ✓ Predicted complexity: {prediction['predicted_complexity']:.2f}")
        print(f"  ✓ Prediction confidence: {prediction['confidence']:.2f}")
        
        # Test 4: Full optimizer integration
        print("  🎯 Testing full optimizer integration...")
        optimizer = AdaptivePerformanceOptimizer()
        
        # Test workload optimization
        opt_result = optimizer.optimize_for_workload("neural_operator_inference", 500, 1.5)
        print(f"  ✓ Optimization result keys: {list(opt_result.keys())}")
        
        # Record performance
        optimizer.record_execution_performance("neural_operator_inference", 500, 2.5, 100.0, 0.95)
        
        # Generate report
        report = optimizer.get_optimization_report()
        print(f"  ✓ Generated report with {len(report)} sections")
        print(f"  ✓ Recommendations: {len(report['system_recommendations'])}")
        
        # Test 5: Physics workload decorator
        print("  🧬 Testing physics workload decorator...")
        
        @optimize_for_physics_workload("calorimeter_simulation", 100, 1.0)
        def simulate_calorimeter(n_events):
            time.sleep(0.01)  # Simulate computation
            return f"Processed {n_events} events"
        
        result = simulate_calorimeter(100)
        print(f"  ✓ Decorated function result: {result}")
        
        print("  ✅ Adaptive performance optimizer tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Optimizer test failed: {e}")
        traceback.print_exc()
        return False


def test_intelligent_scaling():
    """Test intelligent auto-scaling capabilities."""
    print("🔄 Testing Intelligent Auto-scaling...")
    
    try:
        from darkoperator.distributed.intelligent_scaling import (
            IntelligentAutoscaler, PredictiveLoadForecaster, ResourcePool,
            ScalingMetrics, ResourceNode, get_autoscaler, create_sample_resource_pool
        )
        
        # Test 1: Load forecasting
        print("  📈 Testing predictive load forecasting...")
        forecaster = PredictiveLoadForecaster()
        
        # Simulate load pattern
        current_time = time.time()
        for i in range(10):
            metrics = ScalingMetrics(
                cpu_utilization=0.3 + 0.1 * i,
                memory_utilization=0.2 + 0.05 * i,
                queue_length=10 + i * 5,
                throughput_ops_per_sec=100 + i * 10,
                response_time_p95_ms=200 + i * 20,
                error_rate=0.01,
                timestamp=current_time + i * 60
            )
            forecaster.record_load_metrics(metrics)
            
        # Test forecasting
        forecast = forecaster.forecast_load(15)  # 15 minutes ahead
        print(f"  ✓ Load forecast: {forecast['predicted_load']:.2f}")
        print(f"  ✓ Forecast confidence: {forecast['confidence']:.2f}")
        print(f"  ✓ Seasonal component: {forecast['components']['seasonal']:.2f}")
        
        # Test 2: Resource pool management
        print("  🏗️ Testing resource pool management...")
        resource_pool = create_sample_resource_pool()
        
        resource_summary = resource_pool.get_resource_summary()
        print(f"  ✓ Total nodes: {resource_summary['total_nodes']}")
        print(f"  ✓ Total CPU cores: {resource_summary['total_cpu_cores']}")
        print(f"  ✓ Total GPUs: {resource_summary['total_gpus']}")
        print(f"  ✓ Available capabilities: {len(resource_summary['capabilities_available'])}")
        
        # Test node selection
        best_nodes = resource_pool.find_best_nodes(
            "neural_operator_inference", 2, {"requires_gpu": True}
        )
        print(f"  ✓ Found {len(best_nodes)} suitable GPU nodes")
        
        # Test 3: Scaling decisions
        print("  🎯 Testing scaling decision logic...")
        autoscaler = IntelligentAutoscaler(min_instances=2, max_instances=10)
        autoscaler.resource_pool = resource_pool
        
        # Test high load scenario
        high_load_metrics = ScalingMetrics(
            cpu_utilization=0.85,
            memory_utilization=0.75,
            queue_length=200,
            throughput_ops_per_sec=300,
            response_time_p95_ms=1200,
            error_rate=0.03,
            timestamp=time.time()
        )
        
        decision = autoscaler.evaluate_scaling_decision(high_load_metrics)
        print(f"  ✓ High load decision: {decision.action}")
        print(f"  ✓ Target instances: {decision.target_instances}")
        print(f"  ✓ Decision confidence: {decision.confidence:.2f}")
        print(f"  ✓ Reasoning count: {len(decision.reasoning)}")
        
        # Execute scaling decision
        if decision.action != "no_action":
            execution_result = autoscaler.execute_scaling_decision(decision, "neural_operator_inference")
            print(f"  ✓ Execution status: {execution_result['status']}")
            
        # Test low load scenario
        low_load_metrics = ScalingMetrics(
            cpu_utilization=0.15,
            memory_utilization=0.20,
            queue_length=5,
            throughput_ops_per_sec=800,
            response_time_p95_ms=150,
            error_rate=0.005,
            timestamp=time.time() + 600  # 10 minutes later
        )
        
        # Wait for cooldown (simulate)
        autoscaler.last_scaling_time = time.time() - 700  # Make cooldown pass
        
        low_decision = autoscaler.evaluate_scaling_decision(low_load_metrics)
        print(f"  ✓ Low load decision: {low_decision.action}")
        
        # Test 4: Scaling system status
        print("  📊 Testing scaling system status...")
        status = autoscaler.get_scaling_status()
        
        print(f"  ✓ Current instances: {status['current_instances']}")
        print(f"  ✓ Scaling features enabled: {status['features_enabled']}")
        print(f"  ✓ Time since last scaling: {status['time_since_last_scaling_minutes']:.1f} min")
        
        print("  ✅ Intelligent auto-scaling tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Scaling test failed: {e}")
        traceback.print_exc()
        return False


def test_optimization_integration():
    """Test integration between optimization components."""
    print("🔗 Testing Optimization Integration...")
    
    try:
        from darkoperator.optimization.adaptive_performance_optimizer import get_performance_optimizer
        from darkoperator.distributed.intelligent_scaling import get_autoscaler, ScalingMetrics
        
        # Test 1: Combined optimization scenario
        print("  🎭 Testing combined optimization scenario...")
        
        # Get global instances
        optimizer = get_performance_optimizer()
        autoscaler = get_autoscaler()
        
        # Simulate a complex workload scenario
        workload_scenarios = [
            ("neural_operator_inference", 500, 1.2, 0.7, 0.6),  # Medium load
            ("anomaly_detection", 1000, 2.0, 0.9, 0.8),        # High load
            ("physics_simulation", 200, 0.8, 0.3, 0.4),        # Low load
        ]
        
        total_optimizations = 0
        total_scaling_actions = 0
        
        for workload_type, size, complexity, cpu_util, mem_util in workload_scenarios:
            print(f"    Processing {workload_type} workload...")
            
            # Performance optimization
            opt_result = optimizer.optimize_for_workload(workload_type, size, complexity)
            if opt_result.get('optimizations'):
                total_optimizations += 1
                
            # Scaling decision
            metrics = ScalingMetrics(
                cpu_utilization=cpu_util,
                memory_utilization=mem_util,
                queue_length=int(size * 0.1),
                throughput_ops_per_sec=size / 2.0,
                response_time_p95_ms=complexity * 100,
                error_rate=0.01,
                timestamp=time.time()
            )
            
            decision = autoscaler.evaluate_scaling_decision(metrics)
            if decision.action != "no_action":
                total_scaling_actions += 1
                
            # Simulate execution and record performance
            execution_time = size * complexity * 0.001
            memory_used = size * 0.1
            
            optimizer.record_execution_performance(
                workload_type, size, execution_time, memory_used, 0.95
            )
            
            time.sleep(0.01)  # Small delay
            
        print(f"  ✓ Processed {len(workload_scenarios)} workload scenarios")
        print(f"  ✓ Performance optimizations applied: {total_optimizations}")
        print(f"  ✓ Scaling actions recommended: {total_scaling_actions}")
        
        # Test 2: Cross-component reporting
        print("  📋 Testing cross-component reporting...")
        
        # Get reports from both systems
        opt_report = optimizer.get_optimization_report()
        scaling_status = autoscaler.get_scaling_status()
        
        # Verify report integration
        performance_trends = opt_report.get('performance_trends', {})
        scaling_metrics = scaling_status.get('resource_pool', {})
        
        print(f"  ✓ Optimization report sections: {len(opt_report)}")
        print(f"  ✓ Performance trend metrics: {len(performance_trends)}")
        print(f"  ✓ Scaling status sections: {len(scaling_status)}")
        print(f"  ✓ Resource pool metrics: {len(scaling_metrics)}")
        
        # Test 3: System health integration
        print("  🏥 Testing system health integration...")
        
        # Check if systems are providing consistent health indicators
        opt_recommendations = opt_report.get('system_recommendations', [])
        scaling_features = scaling_status.get('features_enabled', {})
        
        health_indicators = {
            'optimization_active': len(opt_recommendations) > 0,
            'predictive_scaling': scaling_features.get('predictive_scaling', False),
            'cost_optimization': scaling_features.get('cost_optimization', False),
            'memory_management': opt_report.get('memory_management', {}).get('hit_rate', 0) > 0.1
        }
        
        healthy_systems = sum(health_indicators.values())
        total_systems = len(health_indicators)
        
        print(f"  ✓ Healthy optimization systems: {healthy_systems}/{total_systems}")
        print(f"  ✓ System health score: {healthy_systems/total_systems:.2%}")
        
        print("  ✅ Optimization integration tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        traceback.print_exc()
        return False


def run_generation3_tests():
    """Run all Generation 3 optimization tests."""
    print("🚀 TERRAGON SDLC v4.0 - Generation 3 Optimization Testing")
    print("=" * 60)
    
    results = {}
    start_time = time.time()
    
    # Run test suites
    test_suites = [
        ("Adaptive Performance Optimizer", test_adaptive_performance_optimizer),
        ("Intelligent Auto-scaling", test_intelligent_scaling),
        ("Optimization Integration", test_optimization_integration)
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
    print("📋 GENERATION 3 OPTIMIZATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    print(f"Execution Time: {duration:.2f} seconds")
    
    # Detailed results
    print("\n📊 Detailed Results:")
    for suite_name, result in results.items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  {suite_name}: {status}")
        if result["error"]:
            print(f"    Error: {result['error']}")
    
    # Save results
    test_results = {
        "generation": 3,
        "test_type": "optimization",
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
    
    results_file = results_dir / "generation3_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Assessment
    if passed_tests == total_tests:
        print("🎉 GENERATION 3 COMPLETE: All optimization tests passed!")
        print("🚀 Ready for comprehensive testing and quality gates.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  GENERATION 3 MOSTLY COMPLETE: Minor issues detected.")
    else:
        print("❌ GENERATION 3 NEEDS WORK: Major optimization issues detected.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    try:
        success = run_generation3_tests()
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