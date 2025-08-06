"""
Comprehensive monitoring and metrics system for quantum task planning.

Provides real-time monitoring, performance analytics, and quantum-specific
metrics for the task planning system with physics-informed insights.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import queue
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class QuantumMetric:
    """Single quantum metric measurement."""
    
    name: str
    value: float
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'unit': self.unit,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class MetricsSummary:
    """Statistical summary of metrics over time period."""
    
    metric_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    percentiles: Dict[str, float]  # p50, p95, p99
    time_range: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            'metric_name': self.metric_name,
            'count': self.count,
            'statistics': {
                'mean': self.mean,
                'std': self.std,
                'min': self.min_value,
                'max': self.max_value,
                'percentiles': self.percentiles
            },
            'time_range': {
                'start': self.time_range[0],
                'end': self.time_range[1],
                'duration': self.time_range[1] - self.time_range[0]
            }
        }


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors."""
    
    @abstractmethod
    def collect(self) -> List[QuantumMetric]:
        """Collect current metrics."""
        pass
    
    @abstractmethod
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector metadata."""
        pass


class SystemMetricsCollector(MetricsCollector):
    """Collects system-level metrics (CPU, memory, GPU)."""
    
    def __init__(self):
        self.last_collection_time = 0
        self.collection_interval = 1.0  # seconds
        
    def collect(self) -> List[QuantumMetric]:
        """Collect system metrics."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_collection_time < self.collection_interval:
            return []
        
        metrics = []
        
        try:
            # CPU metrics
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(QuantumMetric(
                name='system.cpu.usage_percent',
                value=cpu_percent,
                timestamp=current_time,
                unit='percent',
                tags={'source': 'system'}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(QuantumMetric(
                name='system.memory.usage_percent',
                value=memory.percent,
                timestamp=current_time,
                unit='percent',
                tags={'source': 'system'}
            ))
            
            metrics.append(QuantumMetric(
                name='system.memory.available_gb',
                value=memory.available / (1024**3),
                timestamp=current_time,
                unit='GB',
                tags={'source': 'system'}
            ))
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(QuantumMetric(
                    name='system.disk.read_bytes_per_sec',
                    value=disk_io.read_bytes,
                    timestamp=current_time,
                    unit='bytes/sec',
                    tags={'source': 'system'}
                ))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(QuantumMetric(
                    name='system.network.bytes_sent_per_sec',
                    value=net_io.bytes_sent,
                    timestamp=current_time,
                    unit='bytes/sec',
                    tags={'source': 'system'}
                ))
        
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        # GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    gpu_memory_used = torch.cuda.memory_allocated(device_id) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                    
                    metrics.append(QuantumMetric(
                        name='system.gpu.memory_used_gb',
                        value=gpu_memory_used,
                        timestamp=current_time,
                        unit='GB',
                        tags={'source': 'gpu', 'device_id': str(device_id)}
                    ))
                    
                    metrics.append(QuantumMetric(
                        name='system.gpu.memory_usage_percent',
                        value=(gpu_memory_used / gpu_memory_total) * 100,
                        timestamp=current_time,
                        unit='percent',
                        tags={'source': 'gpu', 'device_id': str(device_id)}
                    ))
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        self.last_collection_time = current_time
        return metrics
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information."""
        return {
            'name': 'SystemMetricsCollector',
            'collection_interval': self.collection_interval,
            'metrics_provided': [
                'system.cpu.usage_percent',
                'system.memory.usage_percent', 
                'system.memory.available_gb',
                'system.disk.read_bytes_per_sec',
                'system.network.bytes_sent_per_sec',
                'system.gpu.memory_used_gb',
                'system.gpu.memory_usage_percent'
            ]
        }


class QuantumPlanningMetricsCollector(MetricsCollector):
    """Collects quantum planning specific metrics."""
    
    def __init__(self):
        self.task_metrics = defaultdict(list)
        self.quantum_state_metrics = defaultdict(float)
        self.optimization_metrics = defaultdict(list)
        
    def record_task_execution(
        self, 
        task_id: str, 
        execution_time: float,
        energy_consumed: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ):
        """Record task execution metrics."""
        current_time = time.time()
        
        self.task_metrics['execution_times'].append(execution_time)
        self.task_metrics['energy_consumed'].append(energy_consumed)
        self.task_metrics['success_rate'].append(1.0 if success else 0.0)
        
        # Store detailed record
        record = {
            'task_id': task_id,
            'timestamp': current_time,
            'execution_time': execution_time,
            'energy_consumed': energy_consumed,
            'success': success,
            'metadata': metadata or {}
        }
        
        self.task_metrics['detailed_records'].append(record)
        
        # Keep only recent records (last 1000)
        for key in self.task_metrics:
            if isinstance(self.task_metrics[key], list) and len(self.task_metrics[key]) > 1000:
                self.task_metrics[key] = self.task_metrics[key][-1000:]
    
    def record_quantum_state(self, state_name: str, value: float):
        """Record quantum state metric."""
        self.quantum_state_metrics[state_name] = value
    
    def record_optimization_result(
        self,
        algorithm: str,
        optimization_time: float,
        final_energy: float,
        convergence_steps: int,
        metadata: Dict[str, Any] = None
    ):
        """Record optimization algorithm results."""
        record = {
            'algorithm': algorithm,
            'timestamp': time.time(),
            'optimization_time': optimization_time,
            'final_energy': final_energy,
            'convergence_steps': convergence_steps,
            'metadata': metadata or {}
        }
        
        self.optimization_metrics[algorithm].append(record)
        
        # Keep recent records
        if len(self.optimization_metrics[algorithm]) > 100:
            self.optimization_metrics[algorithm] = self.optimization_metrics[algorithm][-100:]
    
    def collect(self) -> List[QuantumMetric]:
        """Collect quantum planning metrics."""
        metrics = []
        current_time = time.time()
        
        # Task execution metrics
        if self.task_metrics['execution_times']:
            avg_execution_time = statistics.mean(self.task_metrics['execution_times'])
            metrics.append(QuantumMetric(
                name='quantum.tasks.avg_execution_time',
                value=avg_execution_time,
                timestamp=current_time,
                unit='seconds',
                tags={'source': 'quantum_planning'}
            ))
            
            if len(self.task_metrics['execution_times']) > 1:
                std_execution_time = statistics.stdev(self.task_metrics['execution_times'])
                metrics.append(QuantumMetric(
                    name='quantum.tasks.execution_time_std',
                    value=std_execution_time,
                    timestamp=current_time,
                    unit='seconds',
                    tags={'source': 'quantum_planning'}
                ))
        
        if self.task_metrics['energy_consumed']:
            total_energy = sum(self.task_metrics['energy_consumed'])
            metrics.append(QuantumMetric(
                name='quantum.tasks.total_energy_consumed',
                value=total_energy,
                timestamp=current_time,
                unit='energy_units',
                tags={'source': 'quantum_planning'}
            ))
        
        if self.task_metrics['success_rate']:
            success_rate = statistics.mean(self.task_metrics['success_rate'])
            metrics.append(QuantumMetric(
                name='quantum.tasks.success_rate',
                value=success_rate,
                timestamp=current_time,
                unit='fraction',
                tags={'source': 'quantum_planning'}
            ))
        
        # Quantum state metrics
        for state_name, value in self.quantum_state_metrics.items():
            metrics.append(QuantumMetric(
                name=f'quantum.state.{state_name}',
                value=value,
                timestamp=current_time,
                unit='quantum_units',
                tags={'source': 'quantum_state'}
            ))
        
        # Optimization metrics
        for algorithm, records in self.optimization_metrics.items():
            if records:
                recent_records = [r for r in records if current_time - r['timestamp'] < 3600]  # Last hour
                
                if recent_records:
                    avg_opt_time = statistics.mean([r['optimization_time'] for r in recent_records])
                    avg_final_energy = statistics.mean([r['final_energy'] for r in recent_records])
                    avg_convergence_steps = statistics.mean([r['convergence_steps'] for r in recent_records])
                    
                    metrics.extend([
                        QuantumMetric(
                            name=f'quantum.optimization.{algorithm}.avg_time',
                            value=avg_opt_time,
                            timestamp=current_time,
                            unit='seconds',
                            tags={'source': 'optimization', 'algorithm': algorithm}
                        ),
                        QuantumMetric(
                            name=f'quantum.optimization.{algorithm}.avg_energy',
                            value=avg_final_energy,
                            timestamp=current_time,
                            unit='energy_units',
                            tags={'source': 'optimization', 'algorithm': algorithm}
                        ),
                        QuantumMetric(
                            name=f'quantum.optimization.{algorithm}.avg_convergence_steps',
                            value=avg_convergence_steps,
                            timestamp=current_time,
                            unit='steps',
                            tags={'source': 'optimization', 'algorithm': algorithm}
                        )
                    ])
        
        return metrics
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information."""
        return {
            'name': 'QuantumPlanningMetricsCollector',
            'task_records': len(self.task_metrics.get('detailed_records', [])),
            'quantum_states_tracked': len(self.quantum_state_metrics),
            'optimization_algorithms': list(self.optimization_metrics.keys()),
            'total_optimization_records': sum(len(records) for records in self.optimization_metrics.values())
        }


class PhysicsMetricsCollector(MetricsCollector):
    """Collects physics-specific metrics and conservation law violations."""
    
    def __init__(self):
        self.conservation_violations = defaultdict(list)
        self.physics_parameters = defaultdict(list)
        self.symmetry_metrics = defaultdict(float)
        
    def record_conservation_violation(
        self,
        law_type: str,  # 'energy', 'momentum', 'charge'
        violation_magnitude: float,
        task_id: str,
        metadata: Dict[str, Any] = None
    ):
        """Record conservation law violation."""
        record = {
            'law_type': law_type,
            'violation_magnitude': violation_magnitude,
            'task_id': task_id,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.conservation_violations[law_type].append(record)
        
        # Keep recent violations
        if len(self.conservation_violations[law_type]) > 1000:
            self.conservation_violations[law_type] = self.conservation_violations[law_type][-1000:]
    
    def record_physics_parameter(self, parameter_name: str, value: float):
        """Record physics parameter measurement."""
        self.physics_parameters[parameter_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # Keep recent measurements
        if len(self.physics_parameters[parameter_name]) > 1000:
            self.physics_parameters[parameter_name] = self.physics_parameters[parameter_name][-1000:]
    
    def record_symmetry_measure(self, symmetry_type: str, measure: float):
        """Record symmetry preservation measure."""
        self.symmetry_metrics[symmetry_type] = measure
    
    def collect(self) -> List[QuantumMetric]:
        """Collect physics metrics."""
        metrics = []
        current_time = time.time()
        
        # Conservation violation metrics
        for law_type, violations in self.conservation_violations.items():
            recent_violations = [v for v in violations if current_time - v['timestamp'] < 3600]  # Last hour
            
            if recent_violations:
                violation_rate = len(recent_violations) / 3600.0  # violations per second
                avg_magnitude = statistics.mean([v['violation_magnitude'] for v in recent_violations])
                max_magnitude = max([v['violation_magnitude'] for v in recent_violations])
                
                metrics.extend([
                    QuantumMetric(
                        name=f'physics.conservation.{law_type}.violation_rate',
                        value=violation_rate,
                        timestamp=current_time,
                        unit='violations/sec',
                        tags={'source': 'physics', 'law_type': law_type}
                    ),
                    QuantumMetric(
                        name=f'physics.conservation.{law_type}.avg_magnitude',
                        value=avg_magnitude,
                        timestamp=current_time,
                        unit='violation_units',
                        tags={'source': 'physics', 'law_type': law_type}
                    ),
                    QuantumMetric(
                        name=f'physics.conservation.{law_type}.max_magnitude',
                        value=max_magnitude,
                        timestamp=current_time,
                        unit='violation_units',
                        tags={'source': 'physics', 'law_type': law_type}
                    )
                ])
        
        # Physics parameter metrics
        for param_name, measurements in self.physics_parameters.items():
            recent_measurements = [m for m in measurements if current_time - m['timestamp'] < 3600]
            
            if recent_measurements:
                values = [m['value'] for m in recent_measurements]
                avg_value = statistics.mean(values)
                std_value = statistics.stdev(values) if len(values) > 1 else 0.0
                
                metrics.extend([
                    QuantumMetric(
                        name=f'physics.parameters.{param_name}.mean',
                        value=avg_value,
                        timestamp=current_time,
                        unit='physics_units',
                        tags={'source': 'physics', 'parameter': param_name}
                    ),
                    QuantumMetric(
                        name=f'physics.parameters.{param_name}.std',
                        value=std_value,
                        timestamp=current_time,
                        unit='physics_units',
                        tags={'source': 'physics', 'parameter': param_name}
                    )
                ])
        
        # Symmetry metrics
        for symmetry_type, measure in self.symmetry_metrics.items():
            metrics.append(QuantumMetric(
                name=f'physics.symmetry.{symmetry_type}',
                value=measure,
                timestamp=current_time,
                unit='symmetry_units',
                tags={'source': 'physics', 'symmetry_type': symmetry_type}
            ))
        
        return metrics
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information."""
        return {
            'name': 'PhysicsMetricsCollector',
            'conservation_laws_monitored': list(self.conservation_violations.keys()),
            'physics_parameters_tracked': list(self.physics_parameters.keys()),
            'symmetries_measured': list(self.symmetry_metrics.keys()),
            'total_violations_recorded': sum(len(violations) for violations in self.conservation_violations.values())
        }


class MetricsAggregator:
    """Aggregates and analyzes metrics from multiple collectors."""
    
    def __init__(self, retention_period: int = 86400):  # 24 hours default
        self.collectors: List[MetricsCollector] = []
        self.metrics_storage: deque = deque(maxlen=100000)  # Store up to 100k metrics
        self.retention_period = retention_period
        
        # Background thread for collection
        self.collection_thread = None
        self.stop_collection = threading.Event()
        self.collection_interval = 10.0  # seconds
        
        # Metrics summary cache
        self.summary_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    def add_collector(self, collector: MetricsCollector):
        """Add metrics collector."""
        self.collectors.append(collector)
        logger.info(f"Added metrics collector: {collector.__class__.__name__}")
    
    def start_collection(self, interval: float = 10.0):
        """Start background metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Collection already running")
            return
        
        self.collection_interval = interval
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection with interval: {interval}s")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_collection.set()
            self.collection_thread.join(timeout=5.0)
            logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Background collection loop."""
        while not self.stop_collection.is_set():
            try:
                self.collect_all_metrics()
                self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
            
            self.stop_collection.wait(self.collection_interval)
    
    def collect_all_metrics(self):
        """Collect metrics from all registered collectors."""
        for collector in self.collectors:
            try:
                metrics = collector.collect()
                for metric in metrics:
                    self.metrics_storage.append(metric)
            except Exception as e:
                logger.error(f"Error collecting from {collector.__class__.__name__}: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        current_time = time.time()
        cutoff_time = current_time - self.retention_period
        
        # Remove old metrics
        while self.metrics_storage and self.metrics_storage[0].timestamp < cutoff_time:
            self.metrics_storage.popleft()
    
    def get_metrics_summary(
        self, 
        metric_name: str,
        time_range: Optional[Tuple[float, float]] = None,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Optional[MetricsSummary]:
        """
        Get statistical summary of metrics.
        
        Args:
            metric_name: Name of metric to summarize
            time_range: (start_time, end_time) tuple
            tags_filter: Filter metrics by tags
            
        Returns:
            MetricsSummary or None if no matching metrics
        """
        
        # Check cache
        cache_key = f"{metric_name}_{time_range}_{tags_filter}"
        if cache_key in self.summary_cache:
            cached_summary, cache_time = self.summary_cache[cache_key]
            if time.time() - cache_time < self.cache_expiry:
                return cached_summary
        
        # Filter metrics
        matching_metrics = self._filter_metrics(metric_name, time_range, tags_filter)
        
        if not matching_metrics:
            return None
        
        # Compute statistics
        values = [m.value for m in matching_metrics]
        
        if len(values) == 0:
            return None
        
        summary = MetricsSummary(
            metric_name=metric_name,
            count=len(values),
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            percentiles={
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            },
            time_range=(
                min(m.timestamp for m in matching_metrics),
                max(m.timestamp for m in matching_metrics)
            )
        )
        
        # Cache result
        self.summary_cache[cache_key] = (summary, time.time())
        
        return summary
    
    def _filter_metrics(
        self,
        metric_name: str,
        time_range: Optional[Tuple[float, float]] = None,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> List[QuantumMetric]:
        """Filter metrics by name, time range, and tags."""
        
        matching_metrics = []
        
        for metric in self.metrics_storage:
            # Name filter
            if metric.name != metric_name:
                continue
            
            # Time range filter
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= metric.timestamp <= end_time):
                    continue
            
            # Tags filter
            if tags_filter:
                match = True
                for key, value in tags_filter.items():
                    if metric.tags.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            matching_metrics.append(metric)
        
        return matching_metrics
    
    def get_all_metric_names(self) -> List[str]:
        """Get list of all metric names currently stored."""
        metric_names = set()
        for metric in self.metrics_storage:
            metric_names.add(metric.name)
        return sorted(list(metric_names))
    
    def get_metrics_by_tag(self, tag_key: str, tag_value: str) -> List[QuantumMetric]:
        """Get all metrics matching a specific tag."""
        matching_metrics = []
        for metric in self.metrics_storage:
            if metric.tags.get(tag_key) == tag_value:
                matching_metrics.append(metric)
        return matching_metrics
    
    def export_metrics(
        self, 
        format_type: str = 'json',
        time_range: Optional[Tuple[float, float]] = None
    ) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format_type: 'json', 'csv', or 'prometheus'
            time_range: Time range to export
            
        Returns:
            Formatted metrics string
        """
        
        # Filter metrics by time range
        if time_range:
            start_time, end_time = time_range
            metrics_to_export = [
                m for m in self.metrics_storage 
                if start_time <= m.timestamp <= end_time
            ]
        else:
            metrics_to_export = list(self.metrics_storage)
        
        if format_type == 'json':
            return self._export_json(metrics_to_export)
        elif format_type == 'csv':
            return self._export_csv(metrics_to_export)
        elif format_type == 'prometheus':
            return self._export_prometheus(metrics_to_export)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _export_json(self, metrics: List[QuantumMetric]) -> str:
        """Export metrics as JSON."""
        metrics_data = [metric.to_dict() for metric in metrics]
        return json.dumps({
            'metrics': metrics_data,
            'export_timestamp': time.time(),
            'total_metrics': len(metrics_data)
        }, indent=2)
    
    def _export_csv(self, metrics: List[QuantumMetric]) -> str:
        """Export metrics as CSV."""
        if not metrics:
            return "name,value,timestamp,unit,tags\n"
        
        lines = ["name,value,timestamp,unit,tags"]
        
        for metric in metrics:
            tags_str = ";".join([f"{k}={v}" for k, v in metric.tags.items()])
            line = f"{metric.name},{metric.value},{metric.timestamp},{metric.unit},{tags_str}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _export_prometheus(self, metrics: List[QuantumMetric]) -> str:
        """Export metrics in Prometheus format."""
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        lines = []
        
        for metric_name, metric_list in metrics_by_name.items():
            # Add help and type comments
            lines.append(f"# HELP {metric_name} Quantum task planning metric")
            lines.append(f"# TYPE {metric_name} gauge")
            
            # Add metric samples
            for metric in metric_list:
                tags_str = ""
                if metric.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in metric.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"
                
                lines.append(f"{metric_name}{tags_str} {metric.value} {int(metric.timestamp * 1000)}")
        
        return "\n".join(lines)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health."""
        current_time = time.time()
        
        # Recent metrics (last 5 minutes)
        recent_metrics = [
            m for m in self.metrics_storage 
            if current_time - m.timestamp < 300
        ]
        
        # Group by source
        metrics_by_source = defaultdict(int)
        for metric in recent_metrics:
            source = metric.tags.get('source', 'unknown')
            metrics_by_source[source] += 1
        
        # Collector status
        collector_info = []
        for collector in self.collectors:
            try:
                info = collector.get_collector_info()
                info['status'] = 'healthy'
                collector_info.append(info)
            except Exception as e:
                collector_info.append({
                    'name': collector.__class__.__name__,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {
            'timestamp': current_time,
            'total_metrics_stored': len(self.metrics_storage),
            'recent_metrics_count': len(recent_metrics),
            'metrics_by_source': dict(metrics_by_source),
            'collectors': collector_info,
            'collection_running': self.collection_thread is not None and self.collection_thread.is_alive(),
            'collection_interval': self.collection_interval,
            'retention_period': self.retention_period,
            'cache_size': len(self.summary_cache)
        }


class QuantumMetricsManager:
    """
    Main metrics management system for quantum task planning.
    
    Provides unified interface for metrics collection, aggregation, and analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.aggregator = MetricsAggregator(
            retention_period=config.get('retention_period', 86400)
        )
        
        # Initialize default collectors
        self._setup_default_collectors(config)
        
        # Start collection if configured
        if config.get('auto_start', True):
            collection_interval = config.get('collection_interval', 10.0)
            self.aggregator.start_collection(collection_interval)
        
        logger.info("Initialized quantum metrics manager")
    
    def _setup_default_collectors(self, config: Dict[str, Any]):
        """Setup default metrics collectors."""
        
        # System metrics
        if config.get('collect_system_metrics', True):
            self.aggregator.add_collector(SystemMetricsCollector())
        
        # Quantum planning metrics
        self.quantum_collector = QuantumPlanningMetricsCollector()
        self.aggregator.add_collector(self.quantum_collector)
        
        # Physics metrics
        self.physics_collector = PhysicsMetricsCollector()
        self.aggregator.add_collector(self.physics_collector)
    
    def record_task_execution(
        self,
        task_id: str,
        execution_time: float,
        energy_consumed: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ):
        """Record task execution metrics."""
        self.quantum_collector.record_task_execution(
            task_id, execution_time, energy_consumed, success, metadata
        )
    
    def record_conservation_violation(
        self,
        law_type: str,
        violation_magnitude: float,
        task_id: str,
        metadata: Dict[str, Any] = None
    ):
        """Record physics conservation law violation."""
        self.physics_collector.record_conservation_violation(
            law_type, violation_magnitude, task_id, metadata
        )
    
    def record_optimization_result(
        self,
        algorithm: str,
        optimization_time: float,
        final_energy: float,
        convergence_steps: int,
        metadata: Dict[str, Any] = None
    ):
        """Record optimization algorithm results."""
        self.quantum_collector.record_optimization_result(
            algorithm, optimization_time, final_energy, convergence_steps, metadata
        )
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        
        current_time = time.time()
        last_hour = (current_time - 3600, current_time)
        
        dashboard = {
            'timestamp': current_time,
            'system_overview': self.aggregator.get_system_status(),
            'performance_summaries': {},
            'alerts': []
        }
        
        # Key performance metrics
        key_metrics = [
            'quantum.tasks.avg_execution_time',
            'quantum.tasks.success_rate',
            'quantum.tasks.total_energy_consumed',
            'system.cpu.usage_percent',
            'system.memory.usage_percent'
        ]
        
        for metric_name in key_metrics:
            summary = self.aggregator.get_metrics_summary(
                metric_name, time_range=last_hour
            )
            if summary:
                dashboard['performance_summaries'][metric_name] = summary.to_dict()
        
        # Generate alerts
        dashboard['alerts'] = self._generate_alerts()
        
        return dashboard
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics."""
        alerts = []
        current_time = time.time()
        
        # High CPU usage alert
        cpu_summary = self.aggregator.get_metrics_summary(
            'system.cpu.usage_percent',
            time_range=(current_time - 300, current_time)  # Last 5 minutes
        )
        if cpu_summary and cpu_summary.mean > 90:
            alerts.append({
                'type': 'HIGH_CPU_USAGE',
                'severity': 'WARNING',
                'message': f'High CPU usage: {cpu_summary.mean:.1f}%',
                'timestamp': current_time
            })
        
        # Low task success rate alert
        success_summary = self.aggregator.get_metrics_summary(
            'quantum.tasks.success_rate',
            time_range=(current_time - 1800, current_time)  # Last 30 minutes
        )
        if success_summary and success_summary.mean < 0.8:
            alerts.append({
                'type': 'LOW_SUCCESS_RATE',
                'severity': 'CRITICAL',
                'message': f'Low task success rate: {success_summary.mean:.1%}',
                'timestamp': current_time
            })
        
        # High energy consumption alert
        energy_summary = self.aggregator.get_metrics_summary(
            'quantum.tasks.total_energy_consumed',
            time_range=(current_time - 3600, current_time)  # Last hour
        )
        if energy_summary and energy_summary.mean > 1000:  # Threshold
            alerts.append({
                'type': 'HIGH_ENERGY_CONSUMPTION',
                'severity': 'WARNING',
                'message': f'High energy consumption: {energy_summary.mean:.1f}',
                'timestamp': current_time
            })
        
        return alerts
    
    def export_metrics(self, **kwargs) -> str:
        """Export metrics using aggregator."""
        return self.aggregator.export_metrics(**kwargs)
    
    def get_metrics_summary(self, **kwargs) -> Optional[MetricsSummary]:
        """Get metrics summary using aggregator."""
        return self.aggregator.get_metrics_summary(**kwargs)
    
    def shutdown(self):
        """Shutdown metrics collection."""
        self.aggregator.stop_collection()
        logger.info("Quantum metrics manager shutdown completed")