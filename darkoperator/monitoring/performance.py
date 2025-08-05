"""Performance monitoring for neural operators and anomaly detection."""

import time
import torch
import psutil
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading
import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time performance monitoring for physics computations."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Monitoring thresholds
        self.thresholds = {
            'inference_time_ms': 1000,  # Max inference time
            'memory_usage_gb': 8.0,     # Max memory usage
            'gpu_utilization': 0.95,    # Max GPU utilization
            'error_rate': 0.05          # Max error rate
        }
    
    def record_metric(self, name: str, value: float, unit: str, **metadata):
        """Record a performance metric."""
        metric = PerformanceMetric(name, value, unit, metadata=metadata)
        
        with self.lock:
            self.metrics[name].append(metric)
            
            # Keep buffer size manageable
            if len(self.metrics[name]) > self.buffer_size:
                self.metrics[name] = self.metrics[name][-self.buffer_size:]
    
    def time_inference(self, model_name: str = "unknown"):
        """Context manager for timing inference operations."""
        class InferenceTimer:
            def __init__(self, monitor, model_name):
                self.monitor = monitor
                self.model_name = model_name
                self.start_time = None
                self.cuda_available = torch.cuda.is_available()
                
                if self.cuda_available:
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
            
            def __enter__(self):
                if self.cuda_available:
                    self.start_event.record()
                else:
                    self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.cuda_available:
                    self.end_event.record()
                    torch.cuda.synchronize()
                    elapsed_ms = self.start_event.elapsed_time(self.end_event)
                else:
                    elapsed_ms = (time.time() - self.start_time) * 1000
                
                self.monitor.record_metric(
                    "inference_time_ms",
                    elapsed_ms,
                    "ms",
                    model=self.model_name,
                    success=exc_type is None
                )
                
                # Check threshold
                if elapsed_ms > self.monitor.thresholds['inference_time_ms']:
                    logger.warning(f"Slow inference: {elapsed_ms:.2f}ms for {self.model_name}")
        
        return InferenceTimer(self, model_name)
    
    def monitor_memory(self):
        """Record current memory usage."""
        # System RAM
        ram = psutil.virtual_memory()
        self.record_metric("ram_usage_gb", ram.used / (1024**3), "GB", percentage=ram.percent)
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                self.record_metric(f"gpu_{i}_allocated_gb", allocated, "GB", 
                                 device=i, percentage=allocated/total*100)
                self.record_metric(f"gpu_{i}_reserved_gb", reserved, "GB",
                                 device=i, percentage=reserved/total*100)
                
                # Check thresholds
                if allocated > self.thresholds['memory_usage_gb']:
                    logger.warning(f"High GPU {i} memory usage: {allocated:.2f} GB")
    
    def monitor_physics_computation(self, operation_name: str, **kwargs):
        """Context manager for monitoring physics computations."""
        class PhysicsMonitor:
            def __init__(self, monitor, operation_name, **metadata):
                self.monitor = monitor
                self.operation_name = operation_name
                self.metadata = metadata
                self.start_time = None
                self.start_memory = None
            
            def __enter__(self):
                self.start_time = time.time()
                if torch.cuda.is_available():
                    self.start_memory = torch.cuda.memory_allocated()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                
                # Record timing
                self.monitor.record_metric(
                    f"{self.operation_name}_time_s",
                    elapsed,
                    "s",
                    success=exc_type is None,
                    **self.metadata
                )
                
                # Record memory delta
                if torch.cuda.is_available() and self.start_memory is not None:
                    memory_delta = (torch.cuda.memory_allocated() - self.start_memory) / (1024**3)
                    self.monitor.record_metric(
                        f"{self.operation_name}_memory_delta_gb",
                        memory_delta,
                        "GB",
                        **self.metadata
                    )
        
        return PhysicsMonitor(self, operation_name, **kwargs)
    
    def get_statistics(self, metric_name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric over time window."""
        with self.lock:
            if metric_name not in self.metrics:
                return {}
            
            metrics = self.metrics[metric_name]
            
            # Filter by time window if specified
            if window_seconds is not None:
                cutoff_time = time.time() - window_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                'latest': values[-1],
                'trend': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'metrics_collected': {name: len(metrics) for name, metrics in self.metrics.items()},
            'recent_performance': {}
        }
        
        # Recent performance statistics (last 5 minutes)
        window = 300  # 5 minutes
        for metric_name in self.metrics.keys():
            stats = self.get_statistics(metric_name, window)
            if stats:
                report['recent_performance'][metric_name] = stats
        
        # System health indicators
        report['system_health'] = {
            'ram_usage_percent': psutil.virtual_memory().percent,
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            report['gpu_health'] = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                report['gpu_health'][f'gpu_{i}_memory_percent'] = (allocated / total) * 100
        
        # Alert conditions
        alerts = []
        for metric, threshold in self.thresholds.items():
            recent_stats = report['recent_performance'].get(metric, {})
            if recent_stats.get('max', 0) > threshold:
                alerts.append(f"{metric} exceeded threshold: {recent_stats['max']:.2f} > {threshold}")
        
        report['active_alerts'] = alerts
        
        return report
    
    def export_metrics(self, filepath: str = None) -> Dict[str, Any]:
        """Export all collected metrics."""
        if filepath is None:
            filepath = f"performance_metrics_{int(time.time())}.json"
        
        export_data = {
            'collection_start': self.start_time,
            'export_time': time.time(),
            'metrics': {}
        }
        
        with self.lock:
            for name, metric_list in self.metrics.items():
                export_data['metrics'][name] = [
                    {
                        'timestamp': m.timestamp,
                        'value': m.value,
                        'unit': m.unit,
                        'metadata': m.metadata
                    }
                    for m in metric_list
                ]
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance metrics exported to: {filepath}")
        return export_data
    
    def reset_metrics(self):
        """Clear all collected metrics."""
        with self.lock:
            self.metrics.clear()
            self.start_time = time.time()
        logger.info("Performance metrics reset")