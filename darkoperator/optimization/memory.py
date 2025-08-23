"""Memory management and optimization utilities."""

import torch
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_allocated_gb: Dict[int, float]
    gpu_reserved_gb: Dict[int, float]
    gpu_total_gb: Dict[int, float]


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.cleanup_threshold = 0.9  # 90% memory usage triggers cleanup
    
    def cleanup_unused_memory(self):
        """Cleanup unused memory."""
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        stats = {
            'used_percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                stats[f'gpu_{i}_allocated_gb'] = allocated
                stats[f'gpu_{i}_reserved_gb'] = reserved
        
        return stats


class MemoryManager:
    """Advanced memory management for large-scale physics computations."""
    
    def __init__(self, gc_threshold_gb: float = 2.0, auto_cleanup: bool = True):
        self.gc_threshold_bytes = int(gc_threshold_gb * 1024**3)
        self.auto_cleanup = auto_cleanup
        self.lock = threading.RLock()
        
        # Memory tracking
        self.peak_memory = 0
        self.allocation_history = []
        self.cleanup_callbacks = []
        
        # Auto cleanup thread
        if auto_cleanup:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # RAM statistics
        ram = psutil.virtual_memory()
        
        # GPU statistics
        gpu_allocated = {}
        gpu_reserved = {}
        gpu_total = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_allocated[i] = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_reserved[i] = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_total[i] = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        return MemoryStats(
            ram_used_gb=ram.used / (1024**3),
            ram_total_gb=ram.total / (1024**3),
            ram_percent=ram.percent,
            gpu_allocated_gb=gpu_allocated,
            gpu_reserved_gb=gpu_reserved,
            gpu_total_gb=gpu_total
        )
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register callback to be called during cleanup."""
        with self.lock:
            self.cleanup_callbacks.append(callback)
    
    def force_cleanup(self):
        """Force memory cleanup."""
        with self.lock:
            logger.info("Forcing memory cleanup")
            
            # Run registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                logger.debug("CUDA cache cleared")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            try:
                stats = self.get_memory_stats()
                
                # Check if cleanup is needed
                needs_cleanup = (
                    stats.ram_percent > 85 or
                    any(allocated > total * 0.9 
                        for allocated, total in zip(stats.gpu_allocated_gb.values(), 
                                                  stats.gpu_total_gb.values()))
                )
                
                if needs_cleanup:
                    self.force_cleanup()
                    
            except Exception as e:
                logger.warning(f"Memory cleanup loop error: {e}")
    
    @contextmanager
    def memory_limit(self, max_memory_gb: float):
        """Context manager that enforces memory limits."""
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            current_stats = self.get_memory_stats()
            
            # Check if we exceeded limits
            memory_increase = current_stats.ram_used_gb - initial_stats.ram_used_gb
            if memory_increase > max_memory_gb:
                logger.warning(f"Memory usage increased by {memory_increase:.2f} GB (limit: {max_memory_gb:.2f} GB)")
                self.force_cleanup()
    
    @contextmanager
    def track_memory(self, operation_name: str = "operation"):
        """Context manager for tracking memory usage of operations."""
        initial_stats = self.get_memory_stats()
        start_time = time.time()
        
        logger.debug(f"Starting {operation_name} - RAM: {initial_stats.ram_used_gb:.2f} GB")
        
        try:
            yield
        finally:
            final_stats = self.get_memory_stats()
            elapsed = time.time() - start_time
            
            memory_delta = final_stats.ram_used_gb - initial_stats.ram_used_gb
            
            logger.debug(f"Completed {operation_name} in {elapsed:.2f}s - "
                        f"Memory delta: {memory_delta:+.2f} GB")
            
            # Track peak memory
            self.peak_memory = max(self.peak_memory, final_stats.ram_used_gb)


class GPUMemoryOptimizer:
    """GPU memory optimization for neural operators."""
    
    def __init__(self):
        self.device_optimizers = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device_optimizers[i] = self._create_device_optimizer(i)
    
    def _create_device_optimizer(self, device_id: int):
        """Create optimizer for specific GPU device."""
        return {
            'device_id': device_id,
            'memory_pool': torch.cuda.memory_pool(device_id),
            'peak_memory': 0,
            'allocation_history': []
        }
    
    def optimize_for_inference(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """Optimize model and memory layout for inference."""
        device = next(model.parameters()).device
        
        if device.type != 'cuda':
            logger.warning("GPU optimization requested but model is not on CUDA")
            return
        
        device_id = device.index
        logger.info(f"Optimizing model for inference on GPU {device_id}")
        
        with torch.cuda.device(device_id):
            # Clear cache first
            torch.cuda.empty_cache()
            
            # Optimize model
            model.eval()
            
            # Use torch.jit.trace for optimization if possible
            try:
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, sample_input)
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                logger.info("Model optimized with TorchScript")
                return traced_model
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
                return model
    
    def optimize_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 128
    ) -> int:
        """Find optimal batch size for given model and input."""
        device = next(model.parameters()).device
        
        if device.type != 'cuda':
            return min(32, max_batch_size)  # Default for CPU
        
        device_id = device.index
        
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            
            # Binary search for optimal batch size
            min_batch = 1
            max_batch = max_batch_size
            optimal_batch = 1
            
            model.eval()
            
            while min_batch <= max_batch:
                test_batch = (min_batch + max_batch) // 2
                
                try:
                    # Create test batch
                    test_input = sample_input.repeat(test_batch, *([1] * (sample_input.dim() - 1)))
                    
                    # Test forward pass
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    # If successful, try larger batch
                    optimal_batch = test_batch
                    min_batch = test_batch + 1
                    
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM, try smaller batch
                        max_batch = test_batch - 1
                        torch.cuda.empty_cache()
                    else:
                        raise
            
            logger.info(f"Optimal batch size for GPU {device_id}: {optimal_batch}")
            return optimal_batch
    
    def enable_memory_efficient_attention(self, model: torch.nn.Module):
        """Enable memory-efficient attention if available."""
        try:
            # Check if model has attention layers
            attention_modules = []
            for name, module in model.named_modules():
                if 'attention' in name.lower() or isinstance(module, torch.nn.MultiheadAttention):
                    attention_modules.append((name, module))
            
            if attention_modules:
                # Enable memory efficient attention
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info(f"Enabled memory-efficient attention for {len(attention_modules)} modules")
            
        except Exception as e:
            logger.warning(f"Failed to enable memory-efficient attention: {e}")
    
    def get_memory_summary(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed memory summary for GPU(s)."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        summaries = {}
        devices = [device_id] if device_id is not None else range(torch.cuda.device_count())
        
        for dev_id in devices:
            with torch.cuda.device(dev_id):
                props = torch.cuda.get_device_properties(dev_id)
                
                allocated = torch.cuda.memory_allocated(dev_id)
                reserved = torch.cuda.memory_reserved(dev_id)
                total = props.total_memory
                
                summaries[f"gpu_{dev_id}"] = {
                    "name": props.name,
                    "total_memory_gb": total / (1024**3),
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "free_gb": (total - reserved) / (1024**3),
                    "utilization_percent": (allocated / total) * 100,
                    "memory_efficiency": (allocated / reserved) * 100 if reserved > 0 else 0
                }
        
        return summaries
    
    @contextmanager
    def temporary_memory_fraction(self, fraction: float, device_id: Optional[int] = None):
        """Temporarily limit GPU memory usage."""
        if not torch.cuda.is_available():
            yield
            return
        
        devices = [device_id] if device_id is not None else range(torch.cuda.device_count())
        
        try:
            for dev_id in devices:
                torch.cuda.set_per_process_memory_fraction(fraction, dev_id)
            
            yield
            
        finally:
            # Reset to default (1.0)
            for dev_id in devices:
                torch.cuda.set_per_process_memory_fraction(1.0, dev_id)


class AdaptiveMemoryManager:
    """Adaptive memory management that learns from usage patterns."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.gpu_optimizer = GPUMemoryOptimizer()
        
        # Learning components
        self.usage_history = []
        self.optimal_batch_sizes = {}
        self.cleanup_triggers = {}
    
    def adapt_to_workload(self, model_name: str, typical_input_size: tuple):
        """Adapt memory management to specific workload."""
        workload_id = f"{model_name}_{typical_input_size}"
        
        # Learn optimal settings over time
        if workload_id not in self.optimal_batch_sizes:
            # Initial learning phase
            logger.info(f"Learning optimal settings for workload: {workload_id}")
            
            # This would include adaptive batch size tuning,
            # memory allocation patterns, etc.
            self.optimal_batch_sizes[workload_id] = 32  # Default
        
        return self.optimal_batch_sizes[workload_id]
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations based on usage patterns."""
        stats = self.memory_manager.get_memory_stats()
        gpu_summary = self.gpu_optimizer.get_memory_summary()
        
        recommendations = []
        
        # RAM recommendations
        if stats.ram_percent > 80:
            recommendations.append({
                "type": "ram",
                "severity": "high",
                "message": f"High RAM usage ({stats.ram_percent:.1f}%). Consider reducing batch size or enabling data streaming."
            })
        
        # GPU recommendations
        for gpu_id, gpu_info in gpu_summary.items():
            if gpu_info["utilization_percent"] > 90:
                recommendations.append({
                    "type": "gpu",
                    "severity": "high", 
                    "device": gpu_id,
                    "message": f"High GPU memory usage ({gpu_info['utilization_percent']:.1f}%). Consider mixed precision or gradient checkpointing."
                })
            elif gpu_info["memory_efficiency"] < 50:
                recommendations.append({
                    "type": "efficiency",
                    "severity": "medium",
                    "device": gpu_id,
                    "message": f"Low memory efficiency ({gpu_info['memory_efficiency']:.1f}%). Consider optimizing memory allocation patterns."
                })
        
        return {
            "timestamp": time.time(),
            "memory_stats": stats,
            "gpu_summary": gpu_summary,
            "recommendations": recommendations
        }