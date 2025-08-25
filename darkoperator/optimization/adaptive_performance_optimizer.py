"""
Adaptive Performance Optimizer for DarkOperator Studio

Implements intelligent performance optimization with adaptive algorithms,
memory management, and dynamic scaling based on workload patterns.
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import warnings
import weakref

# Graceful imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class OptimizationProfile:
    """Performance optimization profile for specific workloads."""
    workload_type: str
    batch_size: int
    memory_usage_mb: float
    execution_time_ms: float
    throughput_ops_per_sec: float
    gpu_utilization: float
    optimization_level: str
    custom_parameters: Dict[str, Any]
    timestamp: float


@dataclass
class PerformanceTarget:
    """Performance targets for optimization."""
    max_latency_ms: float
    min_throughput_ops: float
    max_memory_mb: float
    target_gpu_utilization: float
    energy_efficiency_target: float


class AdaptiveMemoryManager:
    """Intelligent memory management with adaptive caching."""
    
    def __init__(self, max_cache_size_mb: float = 1024.0):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.memory_usage = 0.0
        self.lock = threading.RLock()
        
        # Adaptive parameters
        self.cache_hit_rate = 0.0
        self.eviction_policy = "lru_smart"  # LRU with smart eviction
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self._update_hit_rate(True)
                return self.cache[key]
            else:
                self._update_hit_rate(False)
                return None
                
    def put(self, key: str, value: Any, estimated_size_mb: float = 1.0):
        """Put item in cache with intelligent eviction."""
        with self.lock:
            # Check if we need to evict items
            while (self.memory_usage + estimated_size_mb > self.max_cache_size_mb 
                   and len(self.cache) > 0):
                self._evict_item()
            
            # Store the item
            if key in self.cache:
                # Update existing item
                self.cache[key] = value
            else:
                self.cache[key] = value
                self.memory_usage += estimated_size_mb
                
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
    def _evict_item(self):
        """Evict item using adaptive policy."""
        if not self.cache:
            return
            
        if self.eviction_policy == "lru":
            # Simple LRU
            oldest_key = min(self.access_times, key=self.access_times.get)
        elif self.eviction_policy == "lru_smart":
            # Smart LRU considering access frequency
            scores = {}
            current_time = time.time()
            
            for key in self.cache:
                recency = current_time - self.access_times.get(key, current_time)
                frequency = self.access_counts.get(key, 1)
                # Lower score = better candidate for eviction
                scores[key] = recency / max(frequency, 1)
                
            oldest_key = max(scores, key=scores.get)
        else:
            # Fallback to simple LRU
            oldest_key = min(self.access_times, key=self.access_times.get)
            
        # Remove the selected item
        if oldest_key in self.cache:
            del self.cache[oldest_key]
            self.access_times.pop(oldest_key, None)
            # Estimate memory reduction (simplified)
            self.memory_usage = max(0, self.memory_usage - 1.0)
            
    def _update_hit_rate(self, was_hit: bool):
        """Update cache hit rate with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        hit_value = 1.0 if was_hit else 0.0
        self.cache_hit_rate = alpha * hit_value + (1 - alpha) * self.cache_hit_rate
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "memory_usage_mb": self.memory_usage,
            "max_memory_mb": self.max_cache_size_mb,
            "hit_rate": self.cache_hit_rate,
            "total_accesses": sum(self.access_counts.values()),
            "eviction_policy": self.eviction_policy
        }
    
    def optimize_policy(self):
        """Adaptively optimize eviction policy based on hit rate."""
        if self.cache_hit_rate < 0.3:
            self.eviction_policy = "lru_smart"  # Need better intelligence
        elif self.cache_hit_rate > 0.8:
            self.eviction_policy = "lru"  # Simple policy is working well


class DynamicBatchOptimizer:
    """Dynamic batch size optimization based on system performance."""
    
    def __init__(self, initial_batch_size: int = 32):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = 1
        self.max_batch_size = 1024
        
        # Performance history
        self.performance_history = deque(maxlen=100)
        self.optimization_step = 0
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.performance_target = None
        
    def record_performance(self, batch_size: int, execution_time: float, 
                          memory_usage: float, throughput: float):
        """Record performance metrics for a batch size."""
        performance = {
            'batch_size': batch_size,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'efficiency': throughput / max(execution_time, 0.001),  # ops/ms
            'timestamp': time.time()
        }
        self.performance_history.append(performance)
        
    def optimize_batch_size(self) -> int:
        """Optimize batch size based on recent performance."""
        if len(self.performance_history) < 3:
            return self.current_batch_size
            
        recent_performances = list(self.performance_history)[-5:]
        
        # Analyze performance trends
        avg_efficiency = sum(p['efficiency'] for p in recent_performances) / len(recent_performances)
        avg_memory = sum(p['memory_usage'] for p in recent_performances) / len(recent_performances)
        
        # Decision logic
        if avg_efficiency < 100:  # Low efficiency
            if avg_memory > 800:  # High memory usage
                # Decrease batch size
                new_batch_size = max(self.min_batch_size, 
                                   int(self.current_batch_size * 0.8))
            else:
                # Try increasing batch size for better efficiency
                new_batch_size = min(self.max_batch_size,
                                   int(self.current_batch_size * 1.2))
        else:
            # Good efficiency, try gradual increase
            new_batch_size = min(self.max_batch_size,
                               self.current_batch_size + 1)
                               
        self.current_batch_size = new_batch_size
        self.optimization_step += 1
        
        return self.current_batch_size
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get batch optimization statistics."""
        if not self.performance_history:
            return {'current_batch_size': self.current_batch_size}
            
        recent = list(self.performance_history)[-10:]
        
        return {
            'current_batch_size': self.current_batch_size,
            'optimization_steps': self.optimization_step,
            'recent_avg_efficiency': sum(p['efficiency'] for p in recent) / len(recent),
            'recent_avg_memory': sum(p['memory_usage'] for p in recent) / len(recent),
            'total_measurements': len(self.performance_history)
        }


class WorkloadPredictor:
    """Predict workload patterns for proactive optimization."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.workload_history = deque(maxlen=history_size)
        self.pattern_cache = {}
        
    def record_workload(self, workload_type: str, size: int, complexity: float):
        """Record workload characteristics."""
        workload = {
            'type': workload_type,
            'size': size,
            'complexity': complexity,
            'timestamp': time.time(),
            'hour_of_day': time.localtime().tm_hour,
            'day_of_week': time.localtime().tm_wday
        }
        self.workload_history.append(workload)
        
    def predict_next_workload(self) -> Dict[str, Any]:
        """Predict characteristics of next workload."""
        if len(self.workload_history) < 10:
            return {'predicted_size': 100, 'predicted_complexity': 1.0, 'confidence': 0.1, 'time_context': 'insufficient_data'}
            
        recent_workloads = list(self.workload_history)[-20:]
        
        # Simple pattern detection
        avg_size = sum(w['size'] for w in recent_workloads) / len(recent_workloads)
        avg_complexity = sum(w['complexity'] for w in recent_workloads) / len(recent_workloads)
        
        # Time-based adjustments
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:  # Business hours
            size_multiplier = 1.5
        elif 18 <= current_hour <= 22:  # Evening
            size_multiplier = 1.2  
        else:  # Night/early morning
            size_multiplier = 0.8
            
        predicted_size = int(avg_size * size_multiplier)
        predicted_complexity = avg_complexity
        
        return {
            'predicted_size': predicted_size,
            'predicted_complexity': predicted_complexity,
            'confidence': min(len(recent_workloads) / 20.0, 1.0),
            'time_context': 'business_hours' if 9 <= current_hour <= 17 else 'off_hours'
        }


class AdaptivePerformanceOptimizer:
    """Main adaptive performance optimizer coordinating all components."""
    
    def __init__(self, 
                 memory_manager: Optional[AdaptiveMemoryManager] = None,
                 batch_optimizer: Optional[DynamicBatchOptimizer] = None,
                 workload_predictor: Optional[WorkloadPredictor] = None):
        
        self.memory_manager = memory_manager or AdaptiveMemoryManager()
        self.batch_optimizer = batch_optimizer or DynamicBatchOptimizer()
        self.workload_predictor = workload_predictor or WorkloadPredictor()
        
        # Optimization profiles for different workload types
        self.optimization_profiles = {}
        self.active_optimizations = {}
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.optimization_effectiveness = deque(maxlen=100)
        
        # Global optimization state
        self.optimization_enabled = True
        self.learning_mode = True
        
    def optimize_for_workload(self, workload_type: str, workload_size: int, 
                            complexity: float = 1.0) -> Dict[str, Any]:
        """Optimize system for specific workload characteristics."""
        
        # Record workload for learning
        self.workload_predictor.record_workload(workload_type, workload_size, complexity)
        
        # Get existing profile or create new one
        profile_key = f"{workload_type}_{self._size_bucket(workload_size)}"
        
        if profile_key not in self.optimization_profiles:
            self.optimization_profiles[profile_key] = self._create_default_profile(
                workload_type, workload_size, complexity
            )
            
        profile = self.optimization_profiles[profile_key]
        
        # Adaptive optimizations
        optimizations = {}
        
        # 1. Memory optimization
        if complexity > 2.0:  # High complexity workload
            self.memory_manager.max_cache_size_mb *= 1.5
            optimizations['memory_boost'] = True
        
        # 2. Batch size optimization
        if workload_size > 1000:  # Large workload
            optimal_batch_size = self.batch_optimizer.optimize_batch_size()
            optimizations['optimal_batch_size'] = optimal_batch_size
        
        # 3. Caching strategy
        cache_key = self._generate_cache_key(workload_type, workload_size)
        cached_result = self.memory_manager.get(cache_key)
        if cached_result:
            optimizations['cache_hit'] = True
            return {'optimizations': optimizations, 'cached_result': cached_result}
        
        optimizations['cache_hit'] = False
        
        return {
            'optimizations': optimizations,
            'profile': profile,
            'predicted_performance': self._estimate_performance(profile, workload_size)
        }
        
    def record_execution_performance(self, workload_type: str, workload_size: int,
                                   execution_time: float, memory_used: float,
                                   result_quality: float = 1.0):
        """Record actual execution performance for learning."""
        
        performance_record = {
            'workload_type': workload_type,
            'workload_size': workload_size,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'result_quality': result_quality,
            'throughput': workload_size / max(execution_time, 0.001),
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance_record)
        
        # Update batch optimizer
        self.batch_optimizer.record_performance(
            self.batch_optimizer.current_batch_size,
            execution_time * 1000,  # Convert to ms
            memory_used,
            performance_record['throughput']
        )
        
        # Learn from performance
        if self.learning_mode:
            self._update_optimization_effectiveness(performance_record)
            
    def _create_default_profile(self, workload_type: str, workload_size: int, 
                               complexity: float) -> OptimizationProfile:
        """Create default optimization profile for workload."""
        
        # Estimate parameters based on workload characteristics
        if workload_type == "neural_operator_inference":
            base_batch_size = 32
            memory_estimate = workload_size * 0.1  # MB per item
        elif workload_type == "anomaly_detection":
            base_batch_size = 64
            memory_estimate = workload_size * 0.05
        else:
            base_batch_size = 16
            memory_estimate = workload_size * 0.2
            
        return OptimizationProfile(
            workload_type=workload_type,
            batch_size=base_batch_size,
            memory_usage_mb=memory_estimate,
            execution_time_ms=workload_size * complexity * 0.1,
            throughput_ops_per_sec=1000.0 / complexity,
            gpu_utilization=0.7,
            optimization_level="adaptive",
            custom_parameters={
                'complexity_factor': complexity,
                'size_category': self._size_bucket(workload_size)
            },
            timestamp=time.time()
        )
        
    def _size_bucket(self, size: int) -> str:
        """Categorize workload size into buckets."""
        if size < 100:
            return "small"
        elif size < 1000:
            return "medium"
        elif size < 10000:
            return "large"
        else:
            return "xlarge"
            
    def _generate_cache_key(self, workload_type: str, workload_size: int) -> str:
        """Generate cache key for workload."""
        size_bucket = self._size_bucket(workload_size)
        return f"{workload_type}_{size_bucket}"
        
    def _estimate_performance(self, profile: OptimizationProfile, 
                            workload_size: int) -> Dict[str, float]:
        """Estimate performance based on profile and workload size."""
        scale_factor = workload_size / 1000.0  # Normalize
        
        return {
            'estimated_time_ms': profile.execution_time_ms * scale_factor,
            'estimated_memory_mb': profile.memory_usage_mb * scale_factor,
            'estimated_throughput': profile.throughput_ops_per_sec / scale_factor
        }
        
    def _update_optimization_effectiveness(self, performance_record: Dict[str, Any]):
        """Update optimization effectiveness based on actual performance."""
        # Compare actual vs predicted performance
        effectiveness = {
            'timestamp': time.time(),
            'workload_type': performance_record['workload_type'],
            'actual_throughput': performance_record['throughput'],
            'quality_score': performance_record['result_quality']
        }
        
        self.optimization_effectiveness.append(effectiveness)
        
        # Adaptive learning: adjust optimization parameters
        if len(self.optimization_effectiveness) >= 10:
            recent_effectiveness = list(self.optimization_effectiveness)[-10:]
            avg_quality = sum(e['quality_score'] for e in recent_effectiveness) / 10
            
            if avg_quality < 0.8:  # Poor quality, be more conservative
                self.memory_manager.eviction_policy = "lru_smart"
                self.batch_optimizer.learning_rate *= 0.9
            elif avg_quality > 0.95:  # Excellent quality, be more aggressive
                self.batch_optimizer.learning_rate *= 1.1
                
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        # Memory stats
        memory_stats = self.memory_manager.get_stats()
        
        # Batch optimization stats
        batch_stats = self.batch_optimizer.get_optimization_stats()
        
        # Performance trends
        if self.performance_history:
            recent_perf = list(self.performance_history)[-20:]
            avg_throughput = sum(p['throughput'] for p in recent_perf) / len(recent_perf)
            avg_quality = sum(p['result_quality'] for p in recent_perf) / len(recent_perf)
        else:
            avg_throughput = 0.0
            avg_quality = 1.0
            
        # Workload prediction
        next_workload = self.workload_predictor.predict_next_workload()
        
        return {
            'timestamp': time.time(),
            'optimization_enabled': self.optimization_enabled,
            'learning_mode': self.learning_mode,
            'memory_management': memory_stats,
            'batch_optimization': batch_stats,
            'performance_trends': {
                'avg_throughput_recent': avg_throughput,
                'avg_quality_recent': avg_quality,
                'total_workloads_processed': len(self.performance_history)
            },
            'workload_prediction': next_workload,
            'optimization_profiles': len(self.optimization_profiles),
            'system_recommendations': self._generate_recommendations()
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        memory_stats = self.memory_manager.get_stats()
        if memory_stats['hit_rate'] < 0.3:
            recommendations.append("Consider increasing cache size for better hit rate")
        elif memory_stats['hit_rate'] > 0.9:
            recommendations.append("Cache performing excellently")
            
        # Batch size recommendations
        batch_stats = self.batch_optimizer.get_optimization_stats()
        if batch_stats.get('recent_avg_efficiency', 0) < 50:
            recommendations.append("Batch size optimization may improve efficiency")
            
        # Performance recommendations
        if len(self.performance_history) > 50:
            recent_quality = [p['result_quality'] for p in list(self.performance_history)[-20:]]
            if sum(recent_quality) / len(recent_quality) < 0.8:
                recommendations.append("Consider more conservative optimization parameters")
                
        if not recommendations:
            recommendations.append("System optimization performing well")
            
        return recommendations


# Global optimizer instance
_global_optimizer = None

def get_performance_optimizer() -> AdaptivePerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptivePerformanceOptimizer()
    return _global_optimizer


def optimize_for_physics_workload(workload_type: str, data_size: int, complexity: float = 1.0):
    """Decorator for optimizing physics computations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            # Pre-execution optimization
            start_time = time.time()
            optimization_result = optimizer.optimize_for_workload(workload_type, data_size, complexity)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-execution performance recording
            execution_time = time.time() - start_time
            # Estimate memory usage (simplified)
            memory_estimate = data_size * 0.1 if data_size else 10.0
            
            optimizer.record_execution_performance(
                workload_type, data_size, execution_time, memory_estimate
            )
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    print("⚡ DarkOperator Adaptive Performance Optimizer")
    
    # Create optimizer
    optimizer = AdaptivePerformanceOptimizer()
    
    # Simulate various workloads
    workload_types = ["neural_operator_inference", "anomaly_detection", "physics_simulation"]
    
    for i in range(10):
        workload_type = workload_types[i % len(workload_types)]
        workload_size = 100 * (i + 1)
        complexity = 1.0 + (i % 3) * 0.5
        
        # Optimize for workload
        opt_result = optimizer.optimize_for_workload(workload_type, workload_size, complexity)
        
        # Simulate execution
        execution_time = 0.1 + (complexity * workload_size / 1000)
        memory_used = workload_size * 0.05
        quality = 0.95 + 0.05 * (i % 2)
        
        optimizer.record_execution_performance(
            workload_type, workload_size, execution_time, memory_used, quality
        )
        
        print(f"Processed {workload_type} workload (size={workload_size}, complexity={complexity:.1f})")
    
    # Generate report
    report = optimizer.get_optimization_report()
    print(f"\n📊 Optimization Report:")
    print(f"Average throughput: {report['performance_trends']['avg_throughput_recent']:.2f} ops/sec")
    print(f"Cache hit rate: {report['memory_management']['hit_rate']:.2%}")
    print(f"Current batch size: {report['batch_optimization']['current_batch_size']}")
    print(f"Recommendations: {', '.join(report['system_recommendations'])}")
    
    print("✅ Performance optimization test completed!")