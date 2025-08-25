"""Advanced monitoring and observability for DarkOperator Studio."""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import warnings


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    unit: str = ""
    

@dataclass  
class PerformanceProfile:
    """Performance profiling data."""
    operation: str
    duration_ms: float
    memory_delta_mb: float
    gpu_utilization: float
    custom_metrics: Dict[str, float]
    timestamp: float
    

class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=max_points))
        self.aggregated_metrics = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Built-in metric types
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     unit: str = ""):
        """Record a metric point."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics_buffer[name].append(point)
            
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
        self.record_metric(f"{name}_total", self.counters[name], tags, "count")
        
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value
        self.record_metric(name, value, tags, "gauge")
        
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        with self.lock:
            self.histograms[name].append(value)
            
            # Keep only recent values for memory efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]
                
        # Calculate percentiles
        if self.histograms[name]:
            try:
                import numpy as np
                values = np.array(self.histograms[name])
                self.record_metric(f"{name}_p50", float(np.percentile(values, 50)), tags, "ms")
                self.record_metric(f"{name}_p95", float(np.percentile(values, 95)), tags, "ms")
                self.record_metric(f"{name}_p99", float(np.percentile(values, 99)), tags, "ms")
            except ImportError:
                # Fallback without numpy
                sorted_values = sorted(self.histograms[name])
                n = len(sorted_values)
                self.record_metric(f"{name}_median", sorted_values[n//2], tags, "ms")
                
    def get_metric_summary(self, metric_name: str, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            points = list(self.metrics_buffer[metric_name])
            
        if not points:
            return {'count': 0}
            
        # Filter by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            points = [p for p in points if p.timestamp >= cutoff_time]
            
        values = [p.value for p in points]
        
        if not values:
            return {'count': 0}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else 0,
            'first_timestamp': points[0].timestamp,
            'last_timestamp': points[-1].timestamp
        }
        
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in various formats."""
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name}_total counter")
            lines.append(f"{name}_total {value}")
            
        # Export gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
            
        return "\n".join(lines)
        
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        data = {
            'timestamp': time.time(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'recent_metrics': {}
        }
        
        # Add recent metric summaries
        for metric_name in self.metrics_buffer.keys():
            data['recent_metrics'][metric_name] = self.get_metric_summary(metric_name, time_window=300)
            
        return json.dumps(data, indent=2)


class PerformanceProfiler:
    """Performance profiler for DarkOperator operations."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.active_profiles = {}
        self.completed_profiles = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def start_profile(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        profile_data = {
            'operation': operation_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_gpu_util': self._get_gpu_utilization()
        }
        
        with self.lock:
            self.active_profiles[profile_id] = profile_data
            
        return profile_id
        
    def end_profile(self, profile_id: str, custom_metrics: Optional[Dict[str, float]] = None) -> PerformanceProfile:
        """End profiling and return results."""
        with self.lock:
            if profile_id not in self.active_profiles:
                raise ValueError(f"Profile {profile_id} not found")
                
            start_data = self.active_profiles.pop(profile_id)
            
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_gpu_util = self._get_gpu_utilization()
        
        profile = PerformanceProfile(
            operation=start_data['operation'],
            duration_ms=(end_time - start_data['start_time']) * 1000,
            memory_delta_mb=end_memory - start_data['start_memory'],
            gpu_utilization=max(end_gpu_util, start_data['start_gpu_util']),
            custom_metrics=custom_metrics or {},
            timestamp=end_time
        )
        
        self.completed_profiles.append(profile)
        
        # Record metrics
        self.metrics_collector.record_histogram(
            f"operation_duration_{start_data['operation']}", 
            profile.duration_ms,
            tags={'operation': start_data['operation']}
        )
        
        self.metrics_collector.record_metric(
            f"memory_delta_{start_data['operation']}",
            profile.memory_delta_mb,
            tags={'operation': start_data['operation']},
            unit="MB"
        )
        
        return profile
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
            
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0.0
        except ImportError:
            return 0.0
            
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                profile_id = self.start_profile(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profile = self.end_profile(profile_id)
                    if hasattr(result, 'shape'):
                        # Add tensor shape info if available
                        profile.custom_metrics['output_size'] = float(result.numel() if hasattr(result, 'numel') else 1)
                        
            return wrapper
        return decorator
        
    def get_performance_report(self, operation: Optional[str] = None, 
                              time_window: float = 3600) -> Dict[str, Any]:
        """Generate performance report."""
        cutoff_time = time.time() - time_window
        
        # Filter profiles
        relevant_profiles = [
            p for p in self.completed_profiles 
            if p.timestamp >= cutoff_time and (operation is None or p.operation == operation)
        ]
        
        if not relevant_profiles:
            return {'message': 'No performance data available'}
            
        # Calculate statistics
        durations = [p.duration_ms for p in relevant_profiles]
        memory_deltas = [p.memory_delta_mb for p in relevant_profiles]
        
        try:
            import numpy as np
            duration_stats = {
                'mean': float(np.mean(durations)),
                'median': float(np.median(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'p95': float(np.percentile(durations, 95)),
                'p99': float(np.percentile(durations, 99))
            }
            
            memory_stats = {
                'mean': float(np.mean(memory_deltas)),
                'median': float(np.median(memory_deltas)),
                'max': float(np.max(memory_deltas)),
                'min': float(np.min(memory_deltas))
            }
        except ImportError:
            # Fallback without numpy
            duration_stats = {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'median': sorted(durations)[len(durations)//2]
            }
            
            memory_stats = {
                'mean': sum(memory_deltas) / len(memory_deltas),
                'min': min(memory_deltas),
                'max': max(memory_deltas)
            }
        
        return {
            'operation': operation or 'all',
            'time_window_hours': time_window / 3600,
            'sample_count': len(relevant_profiles),
            'duration_ms': duration_stats,
            'memory_delta_mb': memory_stats,
            'operations_per_second': len(relevant_profiles) / time_window if time_window > 0 else 0,
            'timestamp': time.time()
        }


class AlertManager:
    """Alert management for monitoring system."""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: str = "warning", message: str = ""):
        """Add an alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'enabled': True
        }
        self.alert_rules.append(rule)
        
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule in self.alert_rules:
            if not rule['enabled']:
                continue
                
            try:
                if rule['condition'](metrics):
                    self._trigger_alert(rule, metrics)
                else:
                    self._resolve_alert(rule['name'])
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule['name']}: {e}")
                
    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_key = rule['name']
        
        if alert_key not in self.active_alerts:
            alert = {
                'name': rule['name'],
                'severity': rule['severity'],
                'message': rule['message'],
                'triggered_at': time.time(),
                'trigger_count': 1,
                'metrics_snapshot': metrics.copy()
            }
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert.copy())
            self.logger.warning(f"ALERT TRIGGERED: {rule['name']} - {rule['message']}")
            
        else:
            # Update existing alert
            self.active_alerts[alert_key]['trigger_count'] += 1
            
    def _resolve_alert(self, alert_name: str):
        """Resolve an active alert."""
        if alert_name in self.active_alerts:
            resolved_alert = self.active_alerts.pop(alert_name)
            resolved_alert['resolved_at'] = time.time()
            self.alert_history.append(resolved_alert)
            self.logger.info(f"ALERT RESOLVED: {alert_name}")
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary."""
        return {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': sum(1 for rule in self.alert_rules if rule['enabled']),
            'alert_history_size': len(self.alert_history),
            'active_alert_names': list(self.active_alerts.keys())
        }


def create_default_alerts() -> AlertManager:
    """Create default alert rules for DarkOperator Studio."""
    alert_manager = AlertManager()
    
    # High error rate alert
    alert_manager.add_alert_rule(
        name="high_error_rate",
        condition=lambda m: m.get('error_rate', 0) > 0.05,  # >5% error rate
        severity="critical",
        message="Error rate exceeded 5% threshold"
    )
    
    # High memory usage alert
    alert_manager.add_alert_rule(
        name="high_memory_usage",
        condition=lambda m: m.get('memory_usage_percent', 0) > 90,
        severity="warning", 
        message="Memory usage exceeded 90%"
    )
    
    # Physics accuracy degradation
    alert_manager.add_alert_rule(
        name="physics_accuracy_degraded",
        condition=lambda m: m.get('physics_accuracy', 1.0) < 0.95,
        severity="warning",
        message="Physics accuracy below 95% threshold"
    )
    
    # Operation timeout alert
    alert_manager.add_alert_rule(
        name="operation_timeouts",
        condition=lambda m: m.get('timeout_count', 0) > 10,
        severity="critical",
        message="Multiple operation timeouts detected"
    )
    
    return alert_manager


# Global instances
_global_metrics_collector = None
_global_profiler = None
_global_alert_manager = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector

def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_metrics_collector())
    return _global_profiler

def get_alert_manager() -> AlertManager:
    """Get or create global alert manager."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = create_default_alerts()
    return _global_alert_manager


if __name__ == "__main__":
    # Example usage and demonstration
    print("📊 DarkOperator Advanced Monitoring System")
    
    # Test metrics collection
    collector = get_metrics_collector()
    collector.increment_counter("operations_processed", 1.0)
    collector.set_gauge("memory_usage_percent", 67.5)
    collector.record_histogram("inference_latency_ms", 15.2)
    
    print("\n📈 Metrics Summary:")
    print(f"Operations processed: {collector.counters['operations_processed']}")
    print(f"Memory usage: {collector.gauges['memory_usage_percent']}%")
    
    # Test profiling
    profiler = get_profiler()
    
    @profiler.profile_operation("test_operation")
    def sample_operation():
        time.sleep(0.1)  # Simulate work
        return "completed"
    
    result = sample_operation()
    
    # Generate performance report
    report = profiler.get_performance_report(time_window=60)
    print(f"\n⚡ Performance Report: {report}")
    
    # Test alerting
    alert_mgr = get_alert_manager()
    test_metrics = {"error_rate": 0.08, "memory_usage_percent": 75}
    alert_mgr.check_alerts(test_metrics)
    
    print(f"\n🚨 Alert Summary: {alert_mgr.get_alert_summary()}")
    
    print("✅ Monitoring system test completed!")