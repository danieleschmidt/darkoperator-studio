"""Performance and benchmarking tests."""

import pytest
import torch
import time
import numpy as np
from unittest.mock import patch

from darkoperator.operators import CalorimeterOperator
from darkoperator.optimization import BatchProcessor, ModelCache, ParallelProcessor
from darkoperator.monitoring import PerformanceMonitor


class TestOperatorPerformance:
    """Test neural operator performance."""
    
    @pytest.fixture
    def operator(self):
        """Create test operator."""
        return CalorimeterOperator(
            modes=16,  # Smaller for faster testing
            width=32,
            output_shape=(20, 20, 10)
        )
    
    def test_inference_speed(self, operator):
        """Test inference speed requirements."""
        batch_size = 32
        n_particles = 6
        
        # Create test batch
        test_batch = torch.randn(batch_size, n_particles, 4)
        test_batch[:, :, 0] = torch.abs(test_batch[:, :, 0]) + 5
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = operator(test_batch)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                output = operator(test_batch)
                end = time.time()
                times.append(end - start)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        # Performance requirements
        assert avg_time < 1.0  # Should process batch in under 1 second
        assert throughput > 10  # At least 10 events per second
        
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.1f} events/sec")
    
    def test_memory_efficiency(self, operator):
        """Test memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        operator = operator.cuda()
        
        # Measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Large batch test
        batch_size = 64
        test_batch = torch.randn(batch_size, 8, 4).cuda()
        test_batch[:, :, 0] = torch.abs(test_batch[:, :, 0]) + 10
        
        with torch.no_grad():
            output = operator(test_batch)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_usage = (peak_memory - initial_memory) / (1024**2)  # MB
        
        # Should use reasonable amount of memory
        assert memory_usage < 1000  # Less than 1GB for this test
        
        print(f"Memory usage: {memory_usage:.1f} MB")
    
    def test_batch_scaling(self, operator):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 8, 16, 32, 64]
        throughputs = []
        
        for batch_size in batch_sizes:
            test_batch = torch.randn(batch_size, 5, 4)
            test_batch[:, :, 0] = torch.abs(test_batch[:, :, 0]) + 5
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = operator(test_batch)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start = time.time()
                    _ = operator(test_batch)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            throughputs.append(throughput)
        
        # Throughput should generally increase with batch size
        assert throughputs[-1] > throughputs[0]  # Larger batches should be more efficient
        
        print(f"Throughput scaling: {dict(zip(batch_sizes, throughputs))}")


class TestOptimizationComponents:
    """Test optimization components performance."""
    
    def test_model_cache_performance(self):
        """Test model cache hit/miss performance."""
        cache = ModelCache(max_memory_gb=1.0)
        
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )
        
        config = {"test": True, "version": 1}
        
        # Test cache miss (first access)
        start = time.time()
        cached_model = cache.get("test_model", config)
        miss_time = time.time() - start
        assert cached_model is None
        
        # Cache the model
        cache.put("test_model", config, model)
        
        # Test cache hit
        start = time.time()
        cached_model = cache.get("test_model", config)
        hit_time = time.time() - start
        
        assert cached_model is not None
        
        # Cache hit should be much faster than miss
        assert hit_time < miss_time
        assert hit_time < 0.1  # Should be very fast
    
    def test_batch_processor_efficiency(self):
        """Test batch processor efficiency."""
        # Simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        
        processor = BatchProcessor(
            model=model,
            batch_size=32,
            mixed_precision=False  # Disable for consistent testing
        )
        
        # Create large dataset
        n_events = 1000
        dataset = [torch.randn(4) for _ in range(n_events)]
        
        # Process dataset
        start_time = time.time()
        results = processor.process_dataset(dataset)
        processing_time = time.time() - start_time
        
        # Check results
        assert len(results) > 0
        total_processed = sum(len(batch) for batch in results)
        assert total_processed == n_events
        
        # Performance requirements
        throughput = n_events / processing_time
        assert throughput > 100  # At least 100 events/sec
        
        print(f"Batch processing throughput: {throughput:.1f} events/sec")
    
    def test_parallel_processor_speedup(self):
        """Test parallel processing speedup."""
        def slow_function(x):
            """Simulate CPU-intensive computation."""
            time.sleep(0.01)  # 10ms per item
            return x ** 2
        
        data = list(range(100))
        
        # Sequential processing
        start = time.time()
        sequential_results = [slow_function(x) for x in data]
        sequential_time = time.time() - start
        
        # Parallel processing
        processor = ParallelProcessor(n_workers=4)
        start = time.time()
        parallel_results = processor.map(slow_function, data)
        parallel_time = time.time() - start
        
        # Check correctness
        assert parallel_results == sequential_results
        
        # Check speedup (should be faster with multiple cores)
        speedup = sequential_time / parallel_time
        assert speedup > 1.5  # At least 1.5x speedup
        
        print(f"Parallel processing speedup: {speedup:.2f}x")


class TestMonitoringPerformance:
    """Test performance monitoring overhead."""
    
    def test_monitoring_overhead(self):
        """Test that performance monitoring has minimal overhead."""
        monitor = PerformanceMonitor()
        
        def compute_intensive_task():
            """Simulate computation."""
            x = torch.randn(1000, 1000)
            return torch.mm(x, x.T)
        
        # Benchmark without monitoring
        times_without = []
        for _ in range(10):
            start = time.time()
            result = compute_intensive_task()
            times_without.append(time.time() - start)
        
        avg_time_without = np.mean(times_without)
        
        # Benchmark with monitoring
        times_with = []
        for _ in range(10):
            with monitor.time_inference("test_model"):
                start = time.time()
                result = compute_intensive_task()
                times_with.append(time.time() - start)
        
        avg_time_with = np.mean(times_with)
        
        # Monitoring overhead should be minimal
        overhead = (avg_time_with - avg_time_without) / avg_time_without
        assert overhead < 0.05  # Less than 5% overhead
        
        print(f"Monitoring overhead: {overhead*100:.2f}%")
    
    def test_memory_monitoring_accuracy(self):
        """Test accuracy of memory monitoring."""
        monitor = PerformanceMonitor()
        
        # Allocate known amount of memory
        large_tensor = torch.randn(1000, 1000)  # ~4MB
        
        with monitor.monitor_physics_computation("memory_test"):
            # Allocate more memory
            larger_tensor = torch.randn(2000, 2000)  # ~16MB
        
        # Check if monitoring captured the allocation
        stats = monitor.get_statistics("memory_test_memory_delta_gb")
        
        if torch.cuda.is_available() and stats:
            # Should detect some memory increase
            assert stats.get('latest', 0) > 0


class TestScalabilityLimits:
    """Test system scalability limits."""
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
    def test_batch_size_scaling(self, batch_size):
        """Test how system handles different batch sizes."""
        operator = CalorimeterOperator(
            modes=8,  # Smaller for testing
            width=16,
            output_shape=(10, 10, 5)
        )
        
        # Create test batch
        test_batch = torch.randn(batch_size, 4, 4)
        test_batch[:, :, 0] = torch.abs(test_batch[:, :, 0]) + 5
        
        # Measure processing time
        start = time.time()
        with torch.no_grad():
            output = operator(test_batch)
        processing_time = time.time() - start
        
        # Check output validity
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()
        
        # Time per event should be reasonable
        time_per_event = processing_time / batch_size
        assert time_per_event < 1.0  # Less than 1 second per event
    
    def test_memory_scaling(self):
        """Test memory usage scaling with problem size."""
        sizes = [(10, 10, 5), (20, 20, 10), (30, 30, 15)]
        memory_usage = []
        
        for output_shape in sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            operator = CalorimeterOperator(
                modes=8,
                width=16,
                output_shape=output_shape
            )
            
            if torch.cuda.is_available():
                operator = operator.cuda()
                current_memory = torch.cuda.memory_allocated()
                memory_usage.append(current_memory - initial_memory)
        
        if torch.cuda.is_available() and len(memory_usage) > 1:
            # Memory usage should scale reasonably with problem size
            scaling_factor = memory_usage[-1] / memory_usage[0]
            assert scaling_factor < 10  # Shouldn't scale too aggressively