"""Monitoring and health check utilities."""

try:
    from .performance import PerformanceMonitor
    BASIC_PERFORMANCE_AVAILABLE = True
except ImportError:
    class PerformanceMonitor:
        pass
    BASIC_PERFORMANCE_AVAILABLE = False

try:
    from .metrics import MetricsCollector as BasicMetricsCollector
    BASIC_METRICS_AVAILABLE = True
except ImportError:
    class BasicMetricsCollector:
        pass
    BASIC_METRICS_AVAILABLE = False

# Advanced monitoring components
try:
    from .advanced_monitoring import (
        MetricsCollector, PerformanceProfiler, AlertManager,
        get_metrics_collector, get_profiler, get_alert_manager
    )
    ADVANCED_MONITORING_AVAILABLE = True
except ImportError:
    ADVANCED_MONITORING_AVAILABLE = False
    # Fallback classes
    class MetricsCollector:
        pass
    class PerformanceProfiler:
        pass
    class AlertManager:
        pass

__all__ = ["PerformanceMonitor", "MetricsCollector"]

if ADVANCED_MONITORING_AVAILABLE:
    __all__.extend([
        "PerformanceProfiler", "AlertManager",
        "get_metrics_collector", "get_profiler", "get_alert_manager"
    ])