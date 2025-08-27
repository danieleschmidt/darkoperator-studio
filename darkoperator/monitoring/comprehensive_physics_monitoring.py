"""
Comprehensive Physics-Informed Monitoring and Telemetry System.

This module provides advanced monitoring specifically designed for physics
simulations, neural operator training, and particle physics research.
Includes real-time physics constraint monitoring, performance telemetry,
and automated alerting for physics violations.
"""

import torch
import numpy as np
import time
import threading
import queue
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PhysicsMetricType(Enum):
    """Types of physics metrics to monitor."""
    ENERGY_CONSERVATION = "energy_conservation"
    MOMENTUM_CONSERVATION = "momentum_conservation"
    ANGULAR_MOMENTUM_CONSERVATION = "angular_momentum_conservation"
    LORENTZ_INVARIANCE = "lorentz_invariance"
    GAUGE_INVARIANCE = "gauge_invariance"
    UNITARITY = "unitarity"
    CAUSALITY = "causality"
    SYMMETRY_PRESERVATION = "symmetry_preservation"
    NUMERICAL_STABILITY = "numerical_stability"
    QUANTUM_COHERENCE = "quantum_coherence"

@dataclass
class PhysicsMetric:
    """Container for physics metric data."""
    metric_type: PhysicsMetricType
    value: float
    expected_value: Optional[float] = None
    tolerance: float = 1e-6
    timestamp: datetime = field(default_factory=datetime.now)
    computation_step: str = ""
    tensor_info: Dict[str, Any] = field(default_factory=dict)
    violation_detected: bool = field(init=False)
    severity: str = field(init=False)
    
    def __post_init__(self):
        if self.expected_value is not None:
            deviation = abs(self.value - self.expected_value)
            self.violation_detected = deviation > self.tolerance
            
            if deviation > 100 * self.tolerance:
                self.severity = "critical"
            elif deviation > 10 * self.tolerance:
                self.severity = "high"
            elif deviation > self.tolerance:
                self.severity = "medium"
            else:
                self.severity = "low"
        else:
            self.violation_detected = False
            self.severity = "low"

@dataclass
class SystemMetric:
    """Container for system performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingMetric:
    """Container for training-specific metrics."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    batch_size: int
    throughput: float  # samples/second
    timestamp: datetime = field(default_factory=datetime.now)

class PhysicsMonitor:
    """Real-time monitor for physics constraints and violations."""
    
    def __init__(self, buffer_size: int = 10000, alert_threshold: float = 1e-3):
        self.buffer_size = buffer_size
        self.alert_threshold = alert_threshold
        
        # Metric storage
        self.physics_metrics: Dict[PhysicsMetricType, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self.system_metrics = deque(maxlen=buffer_size)
        self.training_metrics = deque(maxlen=buffer_size)
        
        # Violation tracking
        self.violation_counts = defaultdict(int)
        self.last_violation_times = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Logger
        self.logger = logging.getLogger("PhysicsMonitor")
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Physics monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Physics monitoring stopped")
    
    def record_physics_metric(self, metric: PhysicsMetric):
        """Record a physics metric."""
        self.physics_metrics[metric.metric_type].append(metric)
        
        if metric.violation_detected and metric.severity in ["high", "critical"]:
            self.violation_counts[metric.metric_type] += 1
            self.last_violation_times[metric.metric_type] = metric.timestamp
            self._trigger_alerts(metric)
    
    def record_system_metric(self, metric: SystemMetric):
        """Record a system performance metric."""
        self.system_metrics.append(metric)
    
    def record_training_metric(self, metric: TrainingMetric):
        """Record a training metric."""
        self.training_metrics.append(metric)
    
    def add_alert_callback(self, callback: Callable[[PhysicsMetric], None]):
        """Add alert callback for violations."""
        self.alert_callbacks.append(callback)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        last_system_check = time.time()
        
        while self._monitoring:
            current_time = time.time()
            
            # System metrics collection (less frequent)
            if current_time - last_system_check >= interval:
                try:
                    system_metric = self._collect_system_metrics()
                    self.record_system_metric(system_metric)
                    last_system_check = current_time
                except Exception as e:
                    self.logger.warning(f"Failed to collect system metrics: {e}")
            
            time.sleep(min(interval / 10, 0.1))  # Check frequently for responsive stopping
    
    def _collect_system_metrics(self) -> SystemMetric:
        """Collect current system performance metrics."""
        # CPU and Memory
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io_sent = net_io.bytes_sent if net_io else 0
        network_io_recv = net_io.bytes_recv if net_io else 0
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_usage = gpu.load * 100
                gpu_memory_usage = gpu.memoryUtil * 100
        except Exception as e:
            self.logger.debug(f"GPU metrics unavailable: {e}")
        
        return SystemMetric(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_io_sent=network_io_sent,
            network_io_recv=network_io_recv
        )
    
    def _trigger_alerts(self, metric: PhysicsMetric):
        """Trigger alert callbacks for violations."""
        self.logger.warning(f"Physics violation detected: {metric.metric_type.value} "
                           f"(severity: {metric.severity}, value: {metric.value:.2e})")
        
        for callback in self.alert_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_physics_summary(self, 
                           metric_type: Optional[PhysicsMetricType] = None,
                           last_n_minutes: int = 10) -> Dict[str, Any]:
        """Get summary of physics metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        
        if metric_type:
            metrics_to_check = [metric_type]
        else:
            metrics_to_check = list(self.physics_metrics.keys())
        
        summary = {}
        
        for mtype in metrics_to_check:
            metrics = [m for m in self.physics_metrics[mtype] if m.timestamp >= cutoff_time]
            
            if not metrics:
                continue
                
            values = [m.value for m in metrics]
            violations = [m for m in metrics if m.violation_detected]
            
            summary[mtype.value] = {
                'count': len(metrics),
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'violations': len(violations),
                'violation_rate': len(violations) / len(metrics) if metrics else 0,
                'last_violation': violations[-1].timestamp.isoformat() if violations else None
            }
        
        return summary
    
    def get_system_summary(self, last_n_minutes: int = 10) -> Dict[str, Any]:
        """Get summary of system performance."""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        def safe_mean(values):
            return np.mean(values) if values else 0.0
        
        summary = {
            'cpu_usage': {
                'mean': safe_mean([m.cpu_usage for m in recent_metrics]),
                'max': max([m.cpu_usage for m in recent_metrics]) if recent_metrics else 0,
            },
            'memory_usage': {
                'mean': safe_mean([m.memory_usage for m in recent_metrics]),
                'max': max([m.memory_usage for m in recent_metrics]) if recent_metrics else 0,
            },
            'gpu_usage': {
                'mean': safe_mean([m.gpu_usage for m in recent_metrics]),
                'max': max([m.gpu_usage for m in recent_metrics]) if recent_metrics else 0,
            },
            'gpu_memory_usage': {
                'mean': safe_mean([m.gpu_memory_usage for m in recent_metrics]),
                'max': max([m.gpu_memory_usage for m in recent_metrics]) if recent_metrics else 0,
            }
        }
        
        return summary

class PhysicsConstraintChecker:
    """Automated checker for physics constraints during computation."""
    
    def __init__(self, monitor: PhysicsMonitor, tolerance: float = 1e-6):
        self.monitor = monitor
        self.tolerance = tolerance
        
    def check_energy_conservation(self, energy_initial: torch.Tensor, 
                                 energy_final: torch.Tensor,
                                 computation_step: str = "") -> PhysicsMetric:
        """Check energy conservation constraint."""
        total_initial = torch.sum(energy_initial).item()
        total_final = torch.sum(energy_final).item()
        
        conservation_error = abs(total_final - total_initial) / (abs(total_initial) + 1e-10)
        
        metric = PhysicsMetric(
            metric_type=PhysicsMetricType.ENERGY_CONSERVATION,
            value=conservation_error,
            expected_value=0.0,
            tolerance=self.tolerance,
            computation_step=computation_step,
            tensor_info={
                'initial_energy': total_initial,
                'final_energy': total_final,
                'energy_shape': list(energy_initial.shape)
            }
        )
        
        self.monitor.record_physics_metric(metric)
        return metric
    
    def check_momentum_conservation(self, momentum_initial: torch.Tensor,
                                   momentum_final: torch.Tensor,
                                   computation_step: str = "") -> PhysicsMetric:
        """Check momentum conservation constraint."""
        initial_total = torch.sum(momentum_initial, dim=0)
        final_total = torch.sum(momentum_final, dim=0)
        
        momentum_error = torch.norm(final_total - initial_total).item()
        initial_norm = torch.norm(initial_total).item()
        relative_error = momentum_error / (initial_norm + 1e-10)
        
        metric = PhysicsMetric(
            metric_type=PhysicsMetricType.MOMENTUM_CONSERVATION,
            value=relative_error,
            expected_value=0.0,
            tolerance=self.tolerance,
            computation_step=computation_step,
            tensor_info={
                'initial_momentum_norm': initial_norm,
                'final_momentum_norm': torch.norm(final_total).item(),
                'momentum_dimensions': momentum_initial.shape[-1]
            }
        )
        
        self.monitor.record_physics_metric(metric)
        return metric
    
    def check_lorentz_invariance(self, four_vector_before: torch.Tensor,
                                four_vector_after: torch.Tensor,
                                computation_step: str = "") -> PhysicsMetric:
        """Check Lorentz invariance preservation."""
        # Calculate invariant mass squared: m² = E² - p²
        def invariant_mass_sq(four_vec):
            return four_vec[..., 0]**2 - torch.sum(four_vec[..., 1:]**2, dim=-1)
        
        mass_sq_before = invariant_mass_sq(four_vector_before)
        mass_sq_after = invariant_mass_sq(four_vector_after)
        
        invariance_error = torch.mean(torch.abs(mass_sq_after - mass_sq_before)).item()
        
        metric = PhysicsMetric(
            metric_type=PhysicsMetricType.LORENTZ_INVARIANCE,
            value=invariance_error,
            expected_value=0.0,
            tolerance=self.tolerance,
            computation_step=computation_step,
            tensor_info={
                'mean_mass_sq_before': torch.mean(mass_sq_before).item(),
                'mean_mass_sq_after': torch.mean(mass_sq_after).item(),
                'batch_size': four_vector_before.shape[0]
            }
        )
        
        self.monitor.record_physics_metric(metric)
        return metric
    
    def check_unitarity(self, transformation_matrix: torch.Tensor,
                       computation_step: str = "") -> PhysicsMetric:
        """Check unitarity of transformation matrix."""
        # Check if U†U ≈ I
        conj_transpose = transformation_matrix.conj().transpose(-2, -1)
        product = torch.matmul(conj_transpose, transformation_matrix)
        
        identity = torch.eye(transformation_matrix.shape[-1], 
                           device=transformation_matrix.device,
                           dtype=transformation_matrix.dtype)
        
        unitarity_error = torch.norm(product - identity).item()
        
        metric = PhysicsMetric(
            metric_type=PhysicsMetricType.UNITARITY,
            value=unitarity_error,
            expected_value=0.0,
            tolerance=self.tolerance,
            computation_step=computation_step,
            tensor_info={
                'matrix_size': transformation_matrix.shape[-1],
                'matrix_condition_number': torch.linalg.cond(transformation_matrix).item(),
                'determinant_magnitude': torch.abs(torch.det(transformation_matrix)).item()
            }
        )
        
        self.monitor.record_physics_metric(metric)
        return metric

class PhysicsVisualizationDashboard:
    """Interactive dashboard for physics monitoring visualization."""
    
    def __init__(self, monitor: PhysicsMonitor):
        self.monitor = monitor
        
    def create_physics_metrics_plot(self, 
                                   metric_types: Optional[List[PhysicsMetricType]] = None,
                                   last_n_hours: int = 1) -> go.Figure:
        """Create interactive plot of physics metrics."""
        if metric_types is None:
            metric_types = list(self.monitor.physics_metrics.keys())
        
        cutoff_time = datetime.now() - timedelta(hours=last_n_hours)
        
        fig = make_subplots(
            rows=len(metric_types), cols=1,
            subplot_titles=[mt.value.replace('_', ' ').title() for mt in metric_types],
            vertical_spacing=0.1
        )
        
        for i, metric_type in enumerate(metric_types):
            metrics = [m for m in self.monitor.physics_metrics[metric_type] 
                      if m.timestamp >= cutoff_time]
            
            if not metrics:
                continue
                
            timestamps = [m.timestamp for m in metrics]
            values = [m.value for m in metrics]
            violations = [m.violation_detected for m in metrics]
            
            # Main metric line
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=values,
                    mode='lines+markers',
                    name=f'{metric_type.value}',
                    line=dict(color='blue'),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.2e}<br>' +
                                '<extra></extra>'
                ),
                row=i+1, col=1
            )
            
            # Violation markers
            violation_times = [t for t, v in zip(timestamps, violations) if v]
            violation_values = [val for val, v in zip(values, violations) if v]
            
            if violation_times:
                fig.add_trace(
                    go.Scatter(
                        x=violation_times, y=violation_values,
                        mode='markers',
                        name=f'{metric_type.value} Violations',
                        marker=dict(color='red', size=10, symbol='x'),
                        hovertemplate='<b>VIOLATION</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Value: %{y:.2e}<br>' +
                                    '<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="Physics Constraints Monitoring",
            height=300 * len(metric_types),
            showlegend=True
        )
        
        return fig
    
    def create_system_metrics_plot(self, last_n_hours: int = 1) -> go.Figure:
        """Create system performance metrics plot."""
        cutoff_time = datetime.now() - timedelta(hours=last_n_hours)
        recent_metrics = [m for m in self.monitor.system_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return go.Figure()
        
        timestamps = [m.timestamp for m in recent_metrics]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CPU Usage (%)', 'Memory Usage (%)', 'GPU Usage (%)', 'GPU Memory (%)'],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.cpu_usage for m in recent_metrics],
                      mode='lines', name='CPU', line=dict(color='green')),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.memory_usage for m in recent_metrics],
                      mode='lines', name='Memory', line=dict(color='orange')),
            row=1, col=2
        )
        
        # GPU Usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.gpu_usage for m in recent_metrics],
                      mode='lines', name='GPU', line=dict(color='purple')),
            row=2, col=1
        )
        
        # GPU Memory
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.gpu_memory_usage for m in recent_metrics],
                      mode='lines', name='GPU Memory', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="System Performance Monitoring",
            height=600,
            showlegend=False
        )
        
        return fig

def create_comprehensive_monitoring_demo() -> Dict[str, Any]:
    """Create a demonstration of comprehensive physics monitoring."""
    
    # Initialize monitoring system
    monitor = PhysicsMonitor(buffer_size=1000, alert_threshold=1e-3)
    constraint_checker = PhysicsConstraintChecker(monitor, tolerance=1e-4)
    
    # Add alert callback
    alerts_triggered = []
    def alert_callback(metric: PhysicsMetric):
        alerts_triggered.append({
            'type': metric.metric_type.value,
            'value': metric.value,
            'severity': metric.severity,
            'timestamp': metric.timestamp.isoformat()
        })
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring(interval=0.5)
    
    try:
        # Simulate physics computations with various constraints
        for i in range(50):
            # Energy conservation test
            energy_initial = torch.randn(10) * 5 + 100
            energy_perturbation = torch.randn(10) * (0.1 if i < 30 else 2.0)  # Introduce violations later
            energy_final = energy_initial + energy_perturbation
            
            constraint_checker.check_energy_conservation(
                energy_initial, energy_final, f"simulation_step_{i}"
            )
            
            # Momentum conservation test
            momentum_initial = torch.randn(10, 3) * 2
            momentum_final = momentum_initial + torch.randn(10, 3) * (0.01 if i < 40 else 0.5)
            
            constraint_checker.check_momentum_conservation(
                momentum_initial, momentum_final, f"simulation_step_{i}"
            )
            
            # Lorentz invariance test
            four_vector_initial = torch.randn(5, 4)
            four_vector_initial[:, 0] = torch.abs(four_vector_initial[:, 0]) + 1.0  # Ensure timelike
            
            # Small perturbation that should preserve invariance
            perturbation_scale = 0.001 if i < 35 else 0.1
            four_vector_final = four_vector_initial + torch.randn(5, 4) * perturbation_scale
            four_vector_final[:, 0] = torch.abs(four_vector_final[:, 0]) + 1.0
            
            constraint_checker.check_lorentz_invariance(
                four_vector_initial, four_vector_final, f"simulation_step_{i}"
            )
            
            # Small delay to simulate computation time
            time.sleep(0.02)
        
        # Wait for monitoring to process all metrics
        time.sleep(1.0)
        
        # Get monitoring summary
        physics_summary = monitor.get_physics_summary(last_n_minutes=5)
        system_summary = monitor.get_system_summary(last_n_minutes=5)
        
        # Create visualization
        dashboard = PhysicsVisualizationDashboard(monitor)
        physics_plot = dashboard.create_physics_metrics_plot(last_n_hours=1)
        system_plot = dashboard.create_system_metrics_plot(last_n_hours=1)
        
        results = {
            'monitoring_successful': True,
            'total_alerts_triggered': len(alerts_triggered),
            'physics_summary': physics_summary,
            'system_summary': system_summary,
            'alerts_sample': alerts_triggered[:5],  # First 5 alerts
            'violation_types_detected': list(set(alert['type'] for alert in alerts_triggered)),
            'plots_created': True
        }
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
    
    return results

if __name__ == "__main__":
    # Run demonstration
    demo_results = create_comprehensive_monitoring_demo()
    print("\n✅ Comprehensive Physics Monitoring Demo Completed Successfully")
    print(f"Total Alerts Triggered: {demo_results['total_alerts_triggered']}")
    print(f"Violation Types Detected: {demo_results['violation_types_detected']}")
    print(f"Physics Summary Keys: {list(demo_results['physics_summary'].keys())}")
    print(f"System Summary Keys: {list(demo_results['system_summary'].keys())}")
    print(f"Monitoring Successful: {demo_results['monitoring_successful']}")