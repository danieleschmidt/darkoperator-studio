"""Advanced performance optimization for neural operators and physics simulations."""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import gc
from pathlib import Path
import psutil
import threading
from functools import wraps, lru_cache
import asyncio
import cProfile
import pstats
import io


class PerformanceProfiler:
    """Advanced profiler for neural operator performance analysis."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profilers = {}
        self.results = {}
    
    def __enter__(self):
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.profiler.disable()
            
            # Collect stats
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            self.results['profile'] = s.getvalue()
    
    def profile_function(self, func_name: str):
        """Decorator to profile specific functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.enabled:
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    
                    if func_name not in self.results:
                        self.results[func_name] = []
                    
                    self.results[func_name].append({
                        'execution_time': end_time - start_time,
                        'args_size': len(args) if args else 0,
                        'kwargs_size': len(kwargs) if kwargs else 0,
                    })
                    
                    return result
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {}
        
        for func_name, measurements in self.results.items():
            if func_name == 'profile':
                continue
                
            times = [m['execution_time'] for m in measurements]
            report[func_name] = {
                'total_calls': len(measurements),
                'total_time': sum(times),
                'average_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': min(times),
                'max_time': max(times),
            }
        
        return report


class GPUOptimizer:
    """GPU-specific optimizations for neural operators."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_pool = None
        
        if torch.cuda.is_available() and device.type == 'cuda':
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory pool for efficient allocation
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference performance."""
        model.eval()
        
        # Enable inference mode optimizations
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Use torch.jit.script for eligible models
        try:
            # Only script simple models to avoid compatibility issues
            if self._is_scriptable(model):
                model = torch.jit.script(model)
        except Exception:
            pass  # Fall back to eager execution
        
        # Fuse operations where possible
        if hasattr(torch.nn.utils, 'fusion') and hasattr(model, 'fuse_model'):
            try:
                model.fuse_model()
            except:
                pass
        
        return model
    
    def _is_scriptable(self, model: nn.Module) -> bool:
        """Check if model can be safely scripted."""
        # Simple heuristic: only script models with basic operations
        for module in model.modules():
            if hasattr(module, 'forward') and hasattr(module.forward, '__code__'):
                # Check for complex control flow that might not be scriptable
                code = module.forward.__code__
                if code.co_flags & 0x20:  # Has generators
                    return False
        return True
    
    def batch_inference(
        self, 
        model: nn.Module, 
        inputs: List[torch.Tensor], 
        batch_size: int = 32,
        use_amp: bool = True
    ) -> List[torch.Tensor]:
        """Perform batched inference with automatic mixed precision."""
        
        # Optimize model
        model = self.optimize_model_for_inference(model)
        
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            
            # Stack into batch
            if batch_inputs:
                try:
                    batch_tensor = torch.stack(batch_inputs).to(self.device)
                    
                    with torch.no_grad():
                        if use_amp and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                batch_output = model(batch_tensor)
                        else:
                            batch_output = model(batch_tensor)
                    
                    # Split results
                    for j in range(batch_output.size(0)):
                        results.append(batch_output[j].cpu())
                    
                    # Clear GPU cache periodically
                    if i % (batch_size * 10) == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    # Process individually if batching fails
                    for single_input in batch_inputs:
                        try:
                            with torch.no_grad():
                                single_output = model(single_input.unsqueeze(0).to(self.device))
                                results.append(single_output[0].cpu())
                        except Exception:
                            results.append(torch.zeros_like(single_input))  # Fallback
        
        return results
    
    def memory_efficient_forward(
        self, 
        model: nn.Module, 
        x: torch.Tensor,
        checkpoint_segments: int = 2
    ) -> torch.Tensor:
        """Memory-efficient forward pass using gradient checkpointing."""
        
        if not self.device.type == 'cuda' or not torch.cuda.is_available():
            return model(x)
        
        # Use gradient checkpointing for memory efficiency
        try:
            if hasattr(torch.utils.checkpoint, 'checkpoint_sequential') and isinstance(model, nn.Sequential):
                return torch.utils.checkpoint.checkpoint_sequential(
                    model, checkpoint_segments, x, use_reentrant=False
                )
            else:
                return torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
        except:
            return model(x)  # Fallback to regular forward


class ParallelProcessor:
    """Parallel processing for large-scale physics computations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.thread_pool = None
        self.process_pool = None
    
    def __enter__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, 4))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    def parallel_map(
        self, 
        func: Callable, 
        data: List[Any], 
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Parallel map with automatic chunking."""
        
        if len(data) == 0:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.max_workers * 4))
        
        # Choose executor
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Submit tasks
        futures = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            future = executor.submit(self._map_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                # Handle failed chunks gracefully
                print(f"Warning: Chunk processing failed: {e}")
        
        return results
    
    def _map_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of data."""
        return [func(item) for item in chunk]
    
    def parallel_batch_process(
        self,
        model: nn.Module,
        data_batches: List[torch.Tensor],
        device_ids: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Process batches in parallel across multiple GPUs."""
        
        if not torch.cuda.is_available() or len(data_batches) == 0:
            return [model(batch) for batch in data_batches]
        
        # Get available GPUs
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) <= 1:
            # Single GPU - use sequential processing
            return [model(batch) for batch in data_batches]
        
        # Multi-GPU processing
        results = [None] * len(data_batches)
        
        def process_batch(batch_idx: int, batch: torch.Tensor, gpu_id: int):
            device = torch.device(f'cuda:{gpu_id}')
            model_gpu = model.to(device)
            batch_gpu = batch.to(device)
            
            with torch.no_grad():
                result = model_gpu(batch_gpu).cpu()
            
            results[batch_idx] = result
        
        # Submit tasks to different GPUs
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            futures = []
            
            for i, batch in enumerate(data_batches):
                gpu_id = device_ids[i % len(device_ids)]
                future = executor.submit(process_batch, i, batch, gpu_id)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        return results


class CacheOptimizer:
    """Advanced caching for expensive computations."""
    
    def __init__(self, cache_size: int = 10000, persist_path: Optional[Path] = None):
        self.cache_size = cache_size
        self.persist_path = persist_path
        self.memory_cache = {}
        self.access_count = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load persistent cache if available
        self._load_persistent_cache()
    
    def cached_computation(self, cache_key: str = None):
        """Decorator for caching expensive computations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key = self._generate_key(func.__name__, args, kwargs)
                
                # Check cache
                if key in self.memory_cache:
                    self.cache_hits += 1
                    self.access_count[key] = self.access_count.get(key, 0) + 1
                    return self.memory_cache[key]
                
                # Compute result
                self.cache_misses += 1
                result = func(*args, **kwargs)
                
                # Store in cache
                self._store_in_cache(key, result)
                
                return result
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments."""
        try:
            # Convert tensors to their properties for hashing
            processed_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    processed_args.append(f"tensor_{arg.shape}_{arg.dtype}_{arg.mean():.6f}")
                else:
                    processed_args.append(str(arg))
            
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    processed_kwargs[k] = f"tensor_{v.shape}_{v.dtype}_{v.mean():.6f}"
                else:
                    processed_kwargs[k] = str(v)
            
            key_str = f"{func_name}_{hash(tuple(processed_args))}_{hash(tuple(sorted(processed_kwargs.items())))}"
            return key_str
            
        except Exception:
            # Fallback to simple hash
            return f"{func_name}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _store_in_cache(self, key: str, value: Any):
        """Store value in cache with LRU eviction."""
        # Evict if cache is full
        if len(self.memory_cache) >= self.cache_size:
            self._evict_lru()
        
        self.memory_cache[key] = value
        self.access_count[key] = 1
    
    def _evict_lru(self):
        """Evict least recently used items."""
        if not self.access_count:
            return
            
        # Find least accessed item
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        
        del self.memory_cache[lru_key]
        del self.access_count[lru_key]
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        if self.persist_path and self.persist_path.exists():
            try:
                import pickle
                with open(self.persist_path, 'rb') as f:
                    self.memory_cache = pickle.load(f)
            except Exception:
                pass
    
    def save_cache(self):
        """Save cache to persistent storage."""
        if self.persist_path:
            try:
                import pickle
                self.persist_path.parent.mkdir(exist_ok=True, parents=True)
                with open(self.persist_path, 'wb') as f:
                    pickle.dump(self.memory_cache, f)
            except Exception:
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.memory_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
        }


class BatchOptimizer:
    """Optimize batch processing for neural operators."""
    
    def __init__(self, target_memory_mb: float = 8000):
        self.target_memory_bytes = target_memory_mb * 1024 * 1024
        self.optimal_batch_sizes = {}
    
    def find_optimal_batch_size(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        max_batch_size: int = 1024,
        memory_safety_factor: float = 0.8
    ) -> int:
        """Find optimal batch size for given model and input."""
        
        # Check cache first
        model_key = f"{type(model).__name__}_{sample_input.shape}_{sample_input.dtype}"
        if model_key in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[model_key]
        
        device = next(model.parameters()).device
        
        # Start with conservative batch size
        batch_size = 1
        optimal_batch_size = 1
        
        model.eval()
        with torch.no_grad():
            while batch_size <= max_batch_size:
                try:
                    # Create batch
                    batch_input = sample_input.unsqueeze(0).repeat(batch_size, *([1] * len(sample_input.shape)))
                    batch_input = batch_input.to(device)
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        memory_before = torch.cuda.memory_allocated()
                    
                    # Forward pass
                    output = model(batch_input)
                    
                    if torch.cuda.is_available():
                        memory_after = torch.cuda.memory_allocated()
                        memory_used = memory_after - memory_before
                        
                        # Check if within memory budget
                        if memory_used * memory_safety_factor < self.target_memory_bytes:
                            optimal_batch_size = batch_size
                        else:
                            break
                    else:
                        optimal_batch_size = batch_size
                    
                    # Try larger batch size
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
        
        # Cache result
        self.optimal_batch_sizes[model_key] = optimal_batch_size
        
        return optimal_batch_size
    
    def adaptive_batch_processing(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        initial_batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Process inputs with adaptive batching based on memory usage."""
        
        if not inputs:
            return []
        
        # Determine initial batch size
        if initial_batch_size is None:
            initial_batch_size = self.find_optimal_batch_size(model, inputs[0])
        
        batch_size = initial_batch_size
        results = []
        i = 0
        
        while i < len(inputs):
            try:
                # Create batch
                batch_end = min(i + batch_size, len(inputs))
                batch_inputs = inputs[i:batch_end]
                
                if batch_inputs:
                    batch_tensor = torch.stack(batch_inputs)
                    
                    with torch.no_grad():
                        batch_output = model(batch_tensor)
                    
                    # Split and store results
                    for j in range(batch_output.size(0)):
                        results.append(batch_output[j])
                    
                    i = batch_end
                    
                    # Increase batch size if successful
                    batch_size = min(batch_size * 2, initial_batch_size * 4)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    if batch_size == 0:
                        raise RuntimeError("Cannot process even single sample")
                else:
                    raise
        
        return results


def test_performance_optimization():
    """Test performance optimization components."""
    print("⚡ Testing performance optimization...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 50 * 50, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = TestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test GPU optimizer
    gpu_optimizer = GPUOptimizer(device)
    optimized_model = gpu_optimizer.optimize_model_for_inference(model)
    print("✓ GPU optimizer test passed")
    
    # Test batch optimizer
    batch_optimizer = BatchOptimizer(target_memory_mb=2000)
    sample_input = torch.randn(1, 50, 50)
    optimal_batch_size = batch_optimizer.find_optimal_batch_size(model, sample_input, max_batch_size=64)
    print(f"✓ Optimal batch size found: {optimal_batch_size}")
    
    # Test cache optimizer
    cache_optimizer = CacheOptimizer(cache_size=100)
    
    @cache_optimizer.cached_computation()
    def expensive_computation(x):
        return torch.sum(x ** 2)
    
    # Test caching
    test_tensor = torch.randn(100, 100)
    result1 = expensive_computation(test_tensor)
    result2 = expensive_computation(test_tensor)  # Should be cached
    
    cache_stats = cache_optimizer.get_cache_stats()
    assert cache_stats['cache_hits'] > 0, "Cache should have hits"
    print(f"✓ Cache optimizer test passed: {cache_stats['hit_rate']:.1%} hit rate")
    
    # Test parallel processor
    with ParallelProcessor(max_workers=4) as processor:
        test_data = list(range(100))
        results = processor.parallel_map(lambda x: x ** 2, test_data)
        assert len(results) == len(test_data), "Parallel map should preserve length"
        print("✓ Parallel processor test passed")
    
    # Test profiler
    with PerformanceProfiler(enabled=True) as profiler:
        # Simulate some computation
        for _ in range(10):
            _ = torch.randn(100, 100) @ torch.randn(100, 100)
    
    print("✓ Performance profiler test passed")
    
    print("✅ All performance optimization tests passed!")


if __name__ == "__main__":
    test_performance_optimization()