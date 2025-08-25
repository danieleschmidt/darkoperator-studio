#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - Comprehensive Validation Suite

Comprehensive testing and validation across all three generations of enhancements:
- Generation 1: Basic functionality validation
- Generation 2: Robustness and error handling
- Generation 3: Optimization and scaling

This suite ensures full system integration and production readiness.
"""

import sys
import time
import json
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add darkoperator to path
sys.path.insert(0, str(Path(__file__).parent))

def run_subprocess_test(script_path: str, description: str) -> Tuple[bool, Dict[str, Any]]:
    """Run a subprocess test and capture results."""
    try:
        print(f"    📝 Running {description}...")
        
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=Path(__file__).parent,
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout
        )
        
        # Parse the result
        success = result.returncode == 0
        
        # Try to extract JSON results if available
        test_results = {}
        if success and "Results saved to:" in result.stdout:
            # Look for results file
            results_pattern = "results/"
            if results_pattern in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Results saved to:" in line and results_pattern in line:
                        results_file = line.split("Results saved to: ")[1].strip()
                        try:
                            with open(results_file) as f:
                                test_results = json.load(f)
                        except:
                            pass
                        break
        
        return success, {
            'returncode': result.returncode,
            'stdout_length': len(result.stdout),
            'stderr_length': len(result.stderr),
            'test_results': test_results,
            'execution_successful': success
        }
        
    except subprocess.TimeoutExpired:
        return False, {'error': 'Test timeout (120s)', 'returncode': -1}
    except Exception as e:
        return False, {'error': str(e), 'returncode': -1}


def test_core_functionality():
    """Test core DarkOperator functionality."""
    print("🎯 Testing Core Functionality...")
    
    try:
        # Test basic imports and module availability
        print("  📦 Testing core module imports...")
        
        # Test core darkoperator import
        import darkoperator as do
        print(f"    ✓ DarkOperator version: {getattr(do, '__version__', 'unknown')}")
        
        # Test key components availability
        components_tested = 0
        components_available = 0
        
        test_imports = [
            ("darkoperator.operators", "Neural operators"),
            ("darkoperator.anomaly", "Anomaly detection"),
            ("darkoperator.models", "Neural models"),
            ("darkoperator.utils", "Utility functions"),
            ("darkoperator.monitoring", "Monitoring systems"),
            ("darkoperator.optimization", "Optimization tools"),
            ("darkoperator.distributed", "Distributed computing")
        ]
        
        for module_name, description in test_imports:
            components_tested += 1
            try:
                __import__(module_name)
                components_available += 1
                print(f"    ✓ {description}: Available")
            except ImportError as e:
                print(f"    ⚠ {description}: Limited ({str(e)[:50]}...)")
        
        availability_rate = components_available / components_tested
        print(f"  📊 Module availability: {availability_rate:.1%} ({components_available}/{components_tested})")
        
        # Test example functionality
        print("  🧪 Testing example workflows...")
        
        # Run quickstart example
        quickstart_success, quickstart_results = run_subprocess_test(
            "examples/quickstart.py", "Quickstart example"
        )
        
        if quickstart_success:
            print("    ✓ Quickstart example executed successfully")
        else:
            print(f"    ✗ Quickstart example failed: {quickstart_results.get('error', 'Unknown error')}")
        
        # Summary
        core_tests_passed = (availability_rate >= 0.7 and quickstart_success)
        
        print(f"  {'✅' if core_tests_passed else '❌'} Core functionality tests {'PASSED' if core_tests_passed else 'FAILED'}")
        
        return core_tests_passed, {
            'module_availability_rate': availability_rate,
            'components_available': components_available,
            'components_tested': components_tested,
            'quickstart_success': quickstart_success,
            'quickstart_results': quickstart_results
        }
        
    except Exception as e:
        print(f"  ✗ Core functionality test failed: {e}")
        traceback.print_exc()
        return False, {'error': str(e)}


def test_generation_progression():
    """Test all three generations in sequence."""
    print("🚀 Testing Generation Progression...")
    
    generation_results = {}
    overall_success = True
    
    # Generation 1: Basic functionality
    print("  📡 Testing Generation 1: Basic Functionality...")
    gen1_success, gen1_results = run_subprocess_test(
        "examples/quickstart.py", "Generation 1 validation"
    )
    generation_results['generation_1'] = {
        'success': gen1_success,
        'results': gen1_results
    }
    if not gen1_success:
        overall_success = False
        print("    ❌ Generation 1 validation failed")
    else:
        print("    ✅ Generation 1 validation passed")
    
    # Generation 2: Robustness
    print("  🛡️ Testing Generation 2: Robustness and Error Handling...")
    gen2_success, gen2_results = run_subprocess_test(
        "test_gen2_robustness.py", "Generation 2 robustness"
    )
    generation_results['generation_2'] = {
        'success': gen2_success,
        'results': gen2_results
    }
    if not gen2_success:
        overall_success = False
        print("    ❌ Generation 2 robustness tests failed")
    else:
        print("    ✅ Generation 2 robustness tests passed")
    
    # Generation 3: Optimization
    print("  ⚡ Testing Generation 3: Optimization and Scaling...")
    gen3_success, gen3_results = run_subprocess_test(
        "test_gen3_optimization.py", "Generation 3 optimization"
    )
    generation_results['generation_3'] = {
        'success': gen3_success,
        'results': gen3_results
    }
    if not gen3_success:
        overall_success = False
        print("    ❌ Generation 3 optimization tests failed")
    else:
        print("    ✅ Generation 3 optimization tests passed")
    
    # Calculate overall progression score
    passed_generations = sum(1 for gen in generation_results.values() if gen['success'])
    progression_score = passed_generations / 3.0
    
    print(f"  📊 Generation Progression: {passed_generations}/3 ({progression_score:.1%})")
    print(f"  {'🎉' if overall_success else '⚠️'} All generations {'PASSED' if overall_success else 'INCOMPLETE'}")
    
    return overall_success, {
        'generation_results': generation_results,
        'passed_generations': passed_generations,
        'progression_score': progression_score,
        'all_generations_passed': overall_success
    }


def test_system_integration():
    """Test system integration across components."""
    print("🔗 Testing System Integration...")
    
    try:
        integration_scores = {}
        
        # Test 1: Cross-module compatibility
        print("  🧩 Testing cross-module compatibility...")
        
        try:
            # Test optimization + monitoring integration
            from darkoperator.optimization.adaptive_performance_optimizer import get_performance_optimizer
            from darkoperator.monitoring.advanced_monitoring import get_metrics_collector
            
            optimizer = get_performance_optimizer()
            metrics = get_metrics_collector()
            
            # Test basic interaction
            metrics.increment_counter("test_integration", 1.0)
            opt_result = optimizer.optimize_for_workload("test_integration", 100, 1.0)
            
            integration_scores['optimization_monitoring'] = 1.0
            print("    ✓ Optimization ↔ Monitoring integration working")
            
        except Exception as e:
            integration_scores['optimization_monitoring'] = 0.0
            print(f"    ✗ Optimization ↔ Monitoring integration failed: {e}")
        
        # Test 2: Error handling + monitoring integration
        print("  🛡️ Testing error handling + monitoring integration...")
        
        try:
            from darkoperator.utils.enhanced_error_handling import get_health_monitor
            from darkoperator.monitoring.advanced_monitoring import get_alert_manager
            
            health_monitor = get_health_monitor()
            alert_manager = get_alert_manager()
            
            # Test basic interaction
            health_results = health_monitor.check_health()
            test_metrics = {'error_rate': 0.02, 'memory_usage_percent': 85}
            alert_manager.check_alerts(test_metrics)
            
            integration_scores['error_monitoring'] = 1.0
            print("    ✓ Error Handling ↔ Monitoring integration working")
            
        except Exception as e:
            integration_scores['error_monitoring'] = 0.0
            print(f"    ✗ Error Handling ↔ Monitoring integration failed: {e}")
        
        # Test 3: Scaling + optimization integration  
        print("  🔄 Testing scaling + optimization integration...")
        
        try:
            from darkoperator.distributed.intelligent_scaling import get_autoscaler
            
            autoscaler = get_autoscaler()
            optimizer = get_performance_optimizer()
            
            # Test coordinated optimization and scaling
            opt_result = optimizer.optimize_for_workload("scaling_test", 500, 1.5)
            
            # Create scaling metrics
            from darkoperator.distributed.intelligent_scaling import ScalingMetrics
            metrics = ScalingMetrics(
                cpu_utilization=0.6,
                memory_utilization=0.7,
                queue_length=50,
                throughput_ops_per_sec=200,
                response_time_p95_ms=300,
                error_rate=0.01,
                timestamp=time.time()
            )
            
            scaling_decision = autoscaler.evaluate_scaling_decision(metrics)
            
            integration_scores['scaling_optimization'] = 1.0
            print("    ✓ Scaling ↔ Optimization integration working")
            
        except Exception as e:
            integration_scores['scaling_optimization'] = 0.0
            print(f"    ✗ Scaling ↔ Optimization integration failed: {e}")
        
        # Calculate integration score
        avg_integration_score = sum(integration_scores.values()) / len(integration_scores) if integration_scores else 0.0
        integration_success = avg_integration_score >= 0.8
        
        print(f"  📊 Integration Score: {avg_integration_score:.1%}")
        print(f"  {'✅' if integration_success else '❌'} System integration {'PASSED' if integration_success else 'FAILED'}")
        
        return integration_success, {
            'integration_scores': integration_scores,
            'avg_integration_score': avg_integration_score,
            'integration_success': integration_success
        }
        
    except Exception as e:
        print(f"  ✗ System integration test failed: {e}")
        traceback.print_exc()
        return False, {'error': str(e)}


def test_production_readiness():
    """Test production readiness indicators."""
    print("🏭 Testing Production Readiness...")
    
    readiness_checks = {}
    
    # Check 1: Configuration management
    print("  ⚙️ Testing configuration management...")
    
    config_files = [
        "darkoperator/config/production.json",
        "darkoperator/config/development.json",
        "docker-compose.yml",
        "Dockerfile"
    ]
    
    config_score = 0
    for config_file in config_files:
        if Path(config_file).exists():
            config_score += 1
            print(f"    ✓ Found {config_file}")
        else:
            print(f"    ⚠ Missing {config_file}")
    
    readiness_checks['configuration'] = config_score / len(config_files)
    
    # Check 2: Deployment artifacts
    print("  🚀 Testing deployment artifacts...")
    
    deployment_files = [
        "deployment_artifacts/docker-compose-prod.yml",
        "deployment_artifacts/k8s_deployment_final.yaml",
        "deployment_artifacts/main.tf"
    ]
    
    deployment_score = 0
    for deploy_file in deployment_files:
        if Path(deploy_file).exists():
            deployment_score += 1
            print(f"    ✓ Found {deploy_file}")
        else:
            print(f"    ⚠ Missing {deploy_file}")
    
    readiness_checks['deployment'] = deployment_score / len(deployment_files)
    
    # Check 3: Monitoring and observability
    print("  📊 Testing monitoring setup...")
    
    monitoring_files = [
        "monitoring/grafana_dashboard.json",
        "monitoring/prometheus.yml"
    ]
    
    monitoring_score = 0
    for monitor_file in monitoring_files:
        if Path(monitor_file).exists():
            monitoring_score += 1
            print(f"    ✓ Found {monitor_file}")
        else:
            print(f"    ⚠ Missing {monitor_file}")
    
    readiness_checks['monitoring'] = monitoring_score / len(monitoring_files)
    
    # Check 4: Documentation
    print("  📚 Testing documentation...")
    
    doc_files = [
        "README.md",
        "DEPLOYMENT.md",
        "CONTRIBUTING.md",
        "docs/README.md"
    ]
    
    doc_score = 0
    for doc_file in doc_files:
        if Path(doc_file).exists():
            doc_score += 1
            print(f"    ✓ Found {doc_file}")
        else:
            print(f"    ⚠ Missing {doc_file}")
    
    readiness_checks['documentation'] = doc_score / len(doc_files)
    
    # Calculate overall readiness
    overall_readiness = sum(readiness_checks.values()) / len(readiness_checks)
    production_ready = overall_readiness >= 0.8
    
    print(f"  📊 Production Readiness Score: {overall_readiness:.1%}")
    print(f"  {'🏭' if production_ready else '⚠️'} Production readiness {'EXCELLENT' if production_ready else 'NEEDS WORK'}")
    
    return production_ready, {
        'readiness_checks': readiness_checks,
        'overall_readiness': overall_readiness,
        'production_ready': production_ready
    }


def run_comprehensive_validation():
    """Run the complete comprehensive validation suite."""
    print("🚀 TERRAGON SDLC v4.0 - Comprehensive System Validation")
    print("=" * 70)
    
    start_time = time.time()
    validation_results = {}
    
    # Test suites
    test_suites = [
        ("Core Functionality", test_core_functionality),
        ("Generation Progression", test_generation_progression),
        ("System Integration", test_system_integration),
        ("Production Readiness", test_production_readiness)
    ]
    
    passed_suites = 0
    total_suites = len(test_suites)
    
    for suite_name, test_func in test_suites:
        print(f"\n🧪 {suite_name} Validation")
        print("-" * 50)
        
        try:
            success, results = test_func()
            validation_results[suite_name] = {
                'success': success,
                'results': results,
                'error': None
            }
            
            if success:
                passed_suites += 1
                print(f"✅ {suite_name}: PASSED")
            else:
                print(f"❌ {suite_name}: FAILED")
                
        except Exception as e:
            validation_results[suite_name] = {
                'success': False,
                'results': {},
                'error': str(e)
            }
            print(f"💥 {suite_name}: CRASHED - {e}")
    
    # Generate comprehensive summary
    end_time = time.time()
    duration = end_time - start_time
    success_rate = passed_suites / total_suites
    
    print("\n" + "=" * 70)
    print("📋 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Suites: {total_suites}")
    print(f"Passed Suites: {passed_suites}")
    print(f"Failed Suites: {total_suites - passed_suites}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Execution Time: {duration:.2f} seconds")
    
    # Detailed breakdown
    print("\n📊 Detailed Results:")
    for suite_name, result in validation_results.items():
        status = "PASSED" if result['success'] else "FAILED" 
        print(f"  {suite_name}: {status}")
        if result['error']:
            print(f"    Error: {result['error']}")
    
    # Save comprehensive results
    comprehensive_results = {
        "validation_type": "comprehensive_sdlc",
        "terragon_version": "4.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": duration,
        "summary": {
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": total_suites - passed_suites,
            "success_rate": success_rate
        },
        "detailed_results": validation_results,
        "overall_assessment": {
            "system_status": "PRODUCTION_READY" if success_rate >= 0.8 else "NEEDS_IMPROVEMENT",
            "confidence_level": "HIGH" if success_rate >= 0.9 else "MEDIUM" if success_rate >= 0.7 else "LOW",
            "recommendations": generate_recommendations(validation_results, success_rate)
        }
    }
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "comprehensive_sdlc_validation.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("-" * 30)
    
    if success_rate >= 0.9:
        print("🎉 EXCELLENT: System is production-ready with high confidence!")
        print("🚀 Ready for deployment and community release.")
    elif success_rate >= 0.8:
        print("✅ GOOD: System is largely ready with minor improvements needed.")
        print("🔧 Address failing tests before full production deployment.")
    elif success_rate >= 0.6:
        print("⚠️ MODERATE: System has significant capabilities but needs work.")
        print("🛠️ Focus on failed test areas before production consideration.")
    else:
        print("❌ POOR: System needs substantial improvement.")
        print("🚧 Major development work required before production readiness.")
    
    return success_rate >= 0.8, comprehensive_results


def generate_recommendations(validation_results: Dict[str, Any], success_rate: float) -> List[str]:
    """Generate recommendations based on validation results."""
    recommendations = []
    
    # Analysis recommendations based on results
    for suite_name, result in validation_results.items():
        if not result['success']:
            if suite_name == "Core Functionality":
                recommendations.append("Install missing dependencies and fix core module imports")
            elif suite_name == "Generation Progression":
                recommendations.append("Debug failed generation tests and ensure progressive enhancement works")
            elif suite_name == "System Integration":
                recommendations.append("Fix cross-component integration issues")
            elif suite_name == "Production Readiness":
                recommendations.append("Complete missing deployment artifacts and documentation")
    
    # General recommendations based on success rate
    if success_rate < 0.8:
        recommendations.append("Prioritize fixing failing test suites before production deployment")
    
    if success_rate >= 0.9:
        recommendations.append("System ready for community release and production deployment")
        recommendations.append("Consider adding advanced features and optimizations")
    
    if not recommendations:
        recommendations.append("System performing excellently across all validation areas")
    
    return recommendations


if __name__ == "__main__":
    try:
        success, results = run_comprehensive_validation()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        exit_code = 2
        
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        traceback.print_exc()
        exit_code = 3
    
    print(f"\n🏁 Comprehensive validation completed with exit code: {exit_code}")
    sys.exit(exit_code)