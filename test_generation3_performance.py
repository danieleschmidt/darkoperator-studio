#!/usr/bin/env python3
"""
Comprehensive test suite for Generation 3: MAKE IT SCALE
Tests performance optimization, parallel processing, caching, and distributed computing.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import as_completed

from darkoperator.optimization.performance_optimizer import (
    PerformanceProfiler, GPUOptimizer, ParallelProcessor, 
    CacheOptimizer, BatchOptimizer
)
from darkoperator.distributed.auto_scaling import (
    ResourceMonitor, AutoScaler, DistributedCoordinator, 
    DistributedWorker, WorkerConfig
)
from darkoperator.models.autoencoder import ConvolutionalAutoencoder
from darkoperator.data.synthetic import generate_background_events


def test_performance_profiler():
    """Test performance profiling capabilities."""
    print("🧪 Testing performance profiler...")
    
    with PerformanceProfiler(enabled=True) as profiler:
        # Test function profiling
        @profiler.profile_function("matrix_multiply")
        def matrix_multiply(size: int):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            return torch.mm(a, b)
        
        # Run multiple operations
        for size in [100, 200, 300]:
            for _ in range(5):
                result = matrix_multiply(size)
        
        # Get performance report
        report = profiler.get_performance_report()
        assert 'matrix_multiply' in report, "Should profile function calls"
        assert report['matrix_multiply']['total_calls'] == 15, "Should track all calls"
        
        print(f"✓ Profiler tracked {report['matrix_multiply']['total_calls']} calls")
        print(f"  Average time: {report['matrix_multiply']['average_time']:.4f}s")
    
    print("📊 Performance profiler tests completed")


def test_gpu_optimization():
    """Test GPU-specific optimizations."""
    print("🧪 Testing GPU optimization...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_optimizer = GPUOptimizer(device)
    
    # Create test model
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=64)
    model = model.to(device)
    
    # Test model optimization for inference
    optimized_model = gpu_optimizer.optimize_model_for_inference(model)
    print("✓ Model optimization for inference completed")
    
    # Test batch inference
    test_inputs = [torch.randn(1, 50, 50) for _ in range(20)]
    
    start_time = time.perf_counter()
    results = gpu_optimizer.batch_inference(
        optimized_model, 
        test_inputs, 
        batch_size=8,
        use_amp=torch.cuda.is_available()
    )
    inference_time = time.perf_counter() - start_time
    
    assert len(results) == len(test_inputs), "Should process all inputs"
    print(f"✓ Batch inference: {len(results)} inputs in {inference_time:.3f}s")
    
    # Test memory-efficient forward pass
    large_input = torch.randn(4, 1, 50, 50).to(device)
    memory_efficient_result = gpu_optimizer.memory_efficient_forward(
        optimized_model, large_input, checkpoint_segments=2
    )
    print("✓ Memory-efficient forward pass completed")
    
    print("📊 GPU optimization tests completed")


def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("🧪 Testing parallel processing...")
    
    with ParallelProcessor(max_workers=4) as processor:
        # Test parallel map with CPU-intensive task
        def cpu_intensive_task(x):
            result = 0
            for i in range(x * 1000):
                result += i ** 0.5
            return result
        
        test_data = list(range(1, 21))  # 1 to 20
        
        # Sequential processing
        start_time = time.perf_counter()
        sequential_results = [cpu_intensive_task(x) for x in test_data]
        sequential_time = time.perf_counter() - start_time
        
        # Parallel processing
        start_time = time.perf_counter()
        parallel_results = processor.parallel_map(cpu_intensive_task, test_data)
        parallel_time = time.perf_counter() - start_time
        
        speedup = sequential_time / parallel_time
        print(f"✓ Parallel speedup: {speedup:.2f}x ({sequential_time:.3f}s -> {parallel_time:.3f}s)")
        
        assert len(parallel_results) == len(test_data), "Should process all data"
        
        # Test parallel batch processing with models
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=32)
        test_batches = [torch.randn(2, 1, 50, 50) for _ in range(8)]
        
        batch_results = processor.parallel_batch_process(
            model, test_batches, device_ids=[0] if torch.cuda.is_available() else None
        )
        
        assert len(batch_results) == len(test_batches), "Should process all batches"
        print("✓ Parallel batch processing completed")
    
    print("📊 Parallel processing tests completed")


def test_caching_optimization():
    """Test caching optimization."""
    print("🧪 Testing caching optimization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "test_cache.pkl"
        cache_optimizer = CacheOptimizer(cache_size=100, persist_path=cache_file)
        
        # Test cached computation
        @cache_optimizer.cached_computation()
        def expensive_physics_computation(energy: float, particles: int):
            # Simulate expensive computation
            time.sleep(0.01)  # 10ms computation
            return energy * particles + np.random.randn() * 0.1
        
        # First call (cache miss)
        start_time = time.perf_counter()
        result1 = expensive_physics_computation(100.0, 1000)
        first_call_time = time.perf_counter() - start_time
        
        # Second call (cache hit)
        start_time = time.perf_counter()
        result2 = expensive_physics_computation(100.0, 1000)
        second_call_time = time.perf_counter() - start_time
        
        # Should be much faster due to caching
        speedup = first_call_time / second_call_time
        print(f"✓ Cache speedup: {speedup:.1f}x ({first_call_time:.4f}s -> {second_call_time:.4f}s)")
        
        # Results should be identical for cached computation
        assert abs(result1 - result2) < 1e-6, "Cached results should be identical"
        
        # Test cache statistics
        cache_stats = cache_optimizer.get_cache_stats()
        assert cache_stats['cache_hits'] > 0, "Should have cache hits"
        assert cache_stats['hit_rate'] > 0, "Should have positive hit rate"
        
        print(f"✓ Cache statistics: {cache_stats['hit_rate']:.1%} hit rate, {cache_stats['cache_size']} items")
        
        # Test cache persistence
        cache_optimizer.save_cache()
        assert cache_file.exists(), "Cache file should be saved"
        print("✓ Cache persistence test completed")
    
    print("📊 Caching optimization tests completed")


def test_batch_optimization():
    """Test batch size optimization."""
    print("🧪 Testing batch optimization...")
    
    batch_optimizer = BatchOptimizer(target_memory_mb=2000)
    
    # Create test model and input
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=64)
    sample_input = torch.randn(1, 50, 50)
    
    # Find optimal batch size
    optimal_batch_size = batch_optimizer.find_optimal_batch_size(
        model, sample_input, max_batch_size=64
    )
    
    print(f"✓ Optimal batch size found: {optimal_batch_size}")
    assert optimal_batch_size > 0, "Should find positive batch size"
    
    # Test adaptive batch processing
    test_inputs = [torch.randn(1, 50, 50) for _ in range(100)]
    
    start_time = time.perf_counter()
    results = batch_optimizer.adaptive_batch_processing(model, test_inputs)
    processing_time = time.perf_counter() - start_time
    
    assert len(results) == len(test_inputs), "Should process all inputs"
    print(f"✓ Adaptive batch processing: {len(results)} inputs in {processing_time:.3f}s")
    
    print("📊 Batch optimization tests completed")


def test_resource_monitoring():
    """Test resource monitoring system."""
    print("🧪 Testing resource monitoring...")
    
    monitor = ResourceMonitor(monitoring_interval=0.1, history_size=20)
    monitor.start_monitoring()
    
    # Let it collect some metrics
    time.sleep(0.5)
    
    current_metrics = monitor.get_current_metrics()
    assert current_metrics is not None, "Should collect current metrics"
    
    average_metrics = monitor.get_average_metrics(window_size=3)
    assert average_metrics is not None, "Should compute average metrics"
    
    print(f"✓ Current CPU usage: {current_metrics.cpu_percent:.1f}%")
    print(f"✓ Current memory usage: {current_metrics.memory_percent:.1f}%")
    print(f"✓ Average CPU usage: {average_metrics.cpu_percent:.1f}%")
    
    # Verify metrics are reasonable
    assert 0 <= current_metrics.cpu_percent <= 100, "CPU percentage should be valid"
    assert 0 <= current_metrics.memory_percent <= 100, "Memory percentage should be valid"
    
    monitor.stop_monitoring()
    print("📊 Resource monitoring tests completed")


def test_auto_scaling():
    """Test auto-scaling logic."""
    print("🧪 Testing auto-scaling...")
    
    auto_scaler = AutoScaler(
        min_workers=1,
        max_workers=4,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        cooldown_period=0.1  # Short cooldown for testing
    )
    
    auto_scaler.resource_monitor.start_monitoring()
    time.sleep(0.2)  # Let monitor collect data
    
    # Test scaling decisions
    initial_workers = auto_scaler.current_workers
    print(f"✓ Initial workers: {initial_workers}")
    
    # Simulate high load (many pending tasks)
    decision = auto_scaler.make_scaling_decision(pending_task_count=20)
    print(f"✓ High load scaling decision: {decision} workers")
    
    # Wait for cooldown and test low load
    time.sleep(0.2)
    decision = auto_scaler.make_scaling_decision(pending_task_count=0)
    print(f"✓ Low load scaling decision: {decision} workers")
    
    auto_scaler.resource_monitor.stop_monitoring()
    print("📊 Auto-scaling tests completed")


def test_distributed_coordination():
    """Test distributed computing coordination."""
    print("🧪 Testing distributed coordination...")
    
    coordinator = DistributedCoordinator(auto_scale=True, max_workers=2)
    coordinator.start()
    
    # Create test model and data
    class TestPhysicsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return torch.relu(self.linear(x))
    
    model = TestPhysicsModel()
    test_batches = [torch.randn(8, 10) for _ in range(12)]
    
    # Submit computation
    start_time = time.perf_counter()
    future = coordinator.submit_computation(model, test_batches)
    results = future.result(timeout=30)
    computation_time = time.perf_counter() - start_time
    
    assert len(results) == len(test_batches), "Should process all batches"
    print(f"✓ Distributed computation: {len(results)} batches in {computation_time:.3f}s")
    
    # Check coordinator status
    status = coordinator.get_status()
    print(f"✓ Coordinator status: {status['num_workers']} workers")
    
    # Submit multiple computations concurrently
    futures = []
    for i in range(3):
        future = coordinator.submit_computation(model, test_batches[:4])
        futures.append(future)
    
    # Wait for all to complete
    concurrent_results = []
    for future in as_completed(futures):
        result = future.result(timeout=30)
        concurrent_results.append(len(result))
    
    print(f"✓ Concurrent computations: {len(concurrent_results)} completed")
    
    coordinator.stop()
    print("📊 Distributed coordination tests completed")


def test_end_to_end_scaling():
    """Test end-to-end scaling with physics workflow."""
    print("🧪 Testing end-to-end scaling with physics workflow...")
    
    # Generate synthetic physics data
    events = generate_background_events(n_events=500, seed=42)
    print(f"✓ Generated {len(events['event_id'])} physics events")
    
    # Create physics model
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=128)
    
    # Create synthetic calorimeter data
    calorimeter_data = []
    for i in range(50):  # Process subset for testing
        data = torch.randn(1, 50, 50)  # Simulate calorimeter shower
        calorimeter_data.append(data)
    
    # Setup distributed coordinator with optimization
    coordinator = DistributedCoordinator(auto_scale=True, max_workers=2)
    coordinator.start()
    
    # Setup performance monitoring
    with PerformanceProfiler(enabled=True) as profiler:
        # Setup GPU optimization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_optimizer = GPUOptimizer(device)
        optimized_model = gpu_optimizer.optimize_model_for_inference(model.to(device))
        
        # Process data in optimized batches
        batch_optimizer = BatchOptimizer(target_memory_mb=1000)
        optimal_batch_size = batch_optimizer.find_optimal_batch_size(
            optimized_model, calorimeter_data[0]
        )
        
        print(f"✓ Using optimal batch size: {optimal_batch_size}")
        
        # Create batches for distributed processing
        batches = []
        for i in range(0, len(calorimeter_data), optimal_batch_size):
            batch = torch.stack(calorimeter_data[i:i + optimal_batch_size])
            batches.append(batch)
        
        # Process with distributed coordination
        start_time = time.perf_counter()
        future = coordinator.submit_computation(optimized_model, batches)
        results = future.result(timeout=60)
        total_time = time.perf_counter() - start_time
        
        print(f"✓ End-to-end processing: {len(calorimeter_data)} events in {total_time:.3f}s")
        print(f"  Throughput: {len(calorimeter_data) / total_time:.1f} events/second")
        
        # Verify results
        assert len(results) == len(batches), "Should process all batches"
        total_processed_events = sum(r.size(0) for r in results)
        assert total_processed_events == len(calorimeter_data), "Should process all events"
        
        # Get performance report
        performance_report = profiler.get_performance_report()
        print("✓ Performance profiling completed")
        
        # Get resource usage
        status = coordinator.get_status()
        if status['resource_metrics']:
            metrics = status['resource_metrics']
            print(f"  Final CPU usage: {metrics['cpu_percent']:.1f}%")
            print(f"  Final memory usage: {metrics['memory_percent']:.1f}%")
    
    coordinator.stop()
    print("✅ End-to-end scaling test completed successfully!")


def main():
    """Run all Generation 3 performance tests."""
    print("⚡ GENERATION 3: PERFORMANCE SCALING TESTING")
    print("=" * 55)
    
    test_functions = [
        test_performance_profiler,
        test_gpu_optimization,
        test_parallel_processing,
        test_caching_optimization,
        test_batch_optimization,
        test_resource_monitoring,
        test_auto_scaling,
        test_distributed_coordination,
        test_end_to_end_scaling,
    ]
    
    passed = 0
    failed = 0
    total_start_time = time.perf_counter()
    
    for test_func in test_functions:
        try:
            print(f"\n{test_func.__name__.replace('_', ' ').title()}")
            print("-" * 45)
            
            test_start_time = time.perf_counter()
            test_func()
            test_time = time.perf_counter() - test_start_time
            
            print(f"⏱️  Test completed in {test_time:.3f}s")
            passed += 1
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            traceback.print_exc()
            failed += 1
    
    total_time = time.perf_counter() - total_start_time
    
    print("\n" + "=" * 55)
    print(f"⚡ GENERATION 3 PERFORMANCE TEST SUMMARY")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🎯 Success Rate: {passed/(passed+failed)*100:.1f}%")
    print(f"⏱️  Total Time: {total_time:.3f}s")
    
    if failed == 0:
        print("🚀 ALL PERFORMANCE SCALING TESTS PASSED!")
        print("🎉 Framework is ready for production-scale physics simulations!")
        return True
    else:
        print("⚠️  Some performance tests failed")
        return False


if __name__ == "__main__":
    main()