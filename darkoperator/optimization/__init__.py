"""Performance optimization utilities."""

# Legacy imports with graceful fallbacks
try:
    from .performance_optimizer import (
        PerformanceProfiler, GPUOptimizer, ParallelProcessor, 
        CacheOptimizer, BatchOptimizer
    )
    LEGACY_OPTIMIZERS_AVAILABLE = True
except ImportError:
    # Fallback classes
    class PerformanceProfiler:
        pass
    class GPUOptimizer:
        pass
    class ParallelProcessor:
        pass
    class CacheOptimizer:
        pass
    class BatchOptimizer:
        pass
    LEGACY_OPTIMIZERS_AVAILABLE = False

# New Generation 3 optimizers (lightweight, no heavy dependencies)
try:
    from .adaptive_performance_optimizer import (
        AdaptivePerformanceOptimizer, AdaptiveMemoryManager, 
        DynamicBatchOptimizer, WorkloadPredictor, get_performance_optimizer,
        optimize_for_physics_workload
    )
    ADAPTIVE_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ADAPTIVE_OPTIMIZERS_AVAILABLE = False

__all__ = [
    "PerformanceProfiler",
    "GPUOptimizer", 
    "ParallelProcessor", 
    "CacheOptimizer",
    "BatchOptimizer"
]

if ADAPTIVE_OPTIMIZERS_AVAILABLE:
    __all__.extend([
        "AdaptivePerformanceOptimizer", "AdaptiveMemoryManager", 
        "DynamicBatchOptimizer", "WorkloadPredictor", "get_performance_optimizer",
        "optimize_for_physics_workload"
    ])