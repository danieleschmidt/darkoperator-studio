"""
Advanced Performance Monitor with Real-Time Analytics

This module implements comprehensive performance monitoring including:
- Real-time inference latency tracking
- GPU/CPU utilization monitoring
- Memory usage optimization
- Adaptive performance tuning
- Quantum computing integration metrics
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    inference_latency: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    throughput: Optional[float] = None
    model_accuracy: Optional[float] = None
    cache_hit_rate: Optional[float] = None


@dataclass 
class PerformanceThresholds:
    """Performance thresholds for alerting."""
    max_latency_ms: float = 100.0
    max_memory_mb: float = 1024.0
    max_cpu_percent: float = 80.0
    max_gpu_percent: float = 90.0
    min_throughput: float = 10.0
    min_accuracy: float = 0.85


class RealTimePerformanceMonitor:
    """
    Advanced real-time performance monitoring system.
    
    Features:
    - Continuous metric collection
    - Automatic performance optimization
    - Anomaly detection in performance patterns
    - Adaptive threshold adjustment
    - Integration with quantum computing metrics
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        sampling_interval: float = 0.1,
        thresholds: Optional[PerformanceThresholds] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.sampling_interval = sampling_interval
        self.thresholds = thresholds or PerformanceThresholds()
        
        # Metric storage
        self.metrics_history: deque = deque(maxlen=window_size)
        self.alert_callbacks: List[Callable] = []
        
        # Performance state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.device_info = self._get_device_info()
        
        # Performance caches
        self.performance_cache = {}
        self.optimization_cache = {}
        
        # Statistics
        self.stats = defaultdict(list)
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'has_cuda': torch.cuda.is_available(),
            'cuda_devices': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info['cuda_devices'].append({
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                })
        
        return info
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                self._check_thresholds(metrics)
                
                # Update statistics
                self._update_statistics(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        timestamp = time.time()
        
        # CPU and memory
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.used / (1024 * 1024)  # MB
        
        # GPU metrics
        gpu_usage = None
        if torch.cuda.is_available():
            try:
                gpu_usage = self._get_gpu_usage()
            except Exception as e:
                self.logger.debug(f"Could not get GPU usage: {e}")
        
        return PerformanceMetrics(
            timestamp=timestamp,
            inference_latency=0.0,  # Will be updated by benchmark calls
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage
        )
    
    def _get_gpu_usage(self) -> float:
        """Get GPU utilization percentage."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Use nvidia-ml-py if available, otherwise estimate
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(info.gpu)
        except ImportError:
            # Fallback: estimate from memory usage
            memory_used = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            return (memory_used / memory_total) * 100.0
    
    def benchmark_inference(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        warmup_runs: int = 5,
        benchmark_runs: int = 50
    ) -> Dict[str, float]:
        """
        Comprehensive inference benchmarking.
        
        Args:
            model: PyTorch model to benchmark
            input_data: Input tensor for inference
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        model.eval()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(benchmark_runs):
                # Memory before
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()
                else:
                    memory_before = 0
                
                # Time inference
                start_time = time.perf_counter()
                output = model(input_data)
                end_time = time.perf_counter()
                
                # Memory after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    torch.cuda.synchronize()
                else:
                    memory_after = 0
                
                latency_ms = (end_time - start_time) * 1000.0
                memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
                
                latencies.append(latency_ms)
                memory_usage.append(memory_used_mb)
        
        # Calculate statistics
        results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'mean_memory_mb': float(np.mean(memory_usage)),
            'throughput_qps': 1000.0 / float(np.mean(latencies)) if latencies else 0.0,
            'batch_size': input_data.shape[0] if len(input_data.shape) > 0 else 1,
        }
        
        # Update metrics history
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            latest_metrics.inference_latency = results['mean_latency_ms']
            latest_metrics.throughput = results['throughput_qps']
        
        self.logger.info(f"Inference benchmark completed: {results['mean_latency_ms']:.2f}ms avg")
        return results
    
    def benchmark_training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark training performance.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            loss_fn: Loss function
            input_data: Training input
            target_data: Training targets
            num_steps: Number of training steps to benchmark
            
        Returns:
            Training performance metrics
        """
        model.train()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        step_times = []
        memory_usage = []
        
        for step in range(num_steps):
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
                torch.cuda.synchronize()
            else:
                memory_before = 0
            
            start_time = time.perf_counter()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_data)
            loss = loss_fn(output, target_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            end_time = time.perf_counter()
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                torch.cuda.synchronize()
            else:
                memory_after = 0
            
            step_time = (end_time - start_time) * 1000.0  # ms
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            step_times.append(step_time)
            memory_usage.append(memory_used)
        
        results = {
            'mean_step_time_ms': float(np.mean(step_times)),
            'std_step_time_ms': float(np.std(step_times)),
            'mean_memory_per_step_mb': float(np.mean(memory_usage)),
            'steps_per_second': 1000.0 / float(np.mean(step_times)) if step_times else 0.0,
        }
        
        self.logger.info(f"Training benchmark: {results['mean_step_time_ms']:.2f}ms per step")
        return results
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger alerts."""
        alerts = []
        
        if metrics.inference_latency > self.thresholds.max_latency_ms:
            alerts.append(f"High latency: {metrics.inference_latency:.2f}ms")
        
        if metrics.memory_usage > self.thresholds.max_memory_mb:
            alerts.append(f"High memory usage: {metrics.memory_usage:.2f}MB")
        
        if metrics.cpu_usage > self.thresholds.max_cpu_percent:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.gpu_usage and metrics.gpu_usage > self.thresholds.max_gpu_percent:
            alerts.append(f"High GPU usage: {metrics.gpu_usage:.1f}%")
        
        if metrics.throughput and metrics.throughput < self.thresholds.min_throughput:
            alerts.append(f"Low throughput: {metrics.throughput:.2f} QPS")
        
        if alerts:
            self._trigger_alerts(alerts, metrics)
    
    def _trigger_alerts(self, alerts: List[str], metrics: PerformanceMetrics):
        """Trigger performance alerts."""
        alert_message = f"Performance Alert: {', '.join(alerts)}"
        self.logger.warning(alert_message)
        
        for callback in self.alert_callbacks:
            try:
                callback(alerts, metrics)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _update_statistics(self, metrics: PerformanceMetrics):
        """Update rolling statistics."""
        self.stats['latency'].append(metrics.inference_latency)
        self.stats['memory'].append(metrics.memory_usage)
        self.stats['cpu'].append(metrics.cpu_usage)
        
        if metrics.gpu_usage is not None:
            self.stats['gpu'].append(metrics.gpu_usage)
        
        if metrics.throughput is not None:
            self.stats['throughput'].append(metrics.throughput)
        
        # Keep only recent data
        max_stats = self.window_size
        for key in self.stats:
            if len(self.stats[key]) > max_stats:
                self.stats[key] = self.stats[key][-max_stats:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.stats:
            return {'status': 'no_data'}
        
        summary = {}
        
        for metric, values in self.stats.items():
            if not values:
                continue
                
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99)),
                'samples': len(values)
            }
        
        # Overall system health score
        health_score = self._calculate_health_score()
        summary['health_score'] = health_score
        summary['device_info'] = self.device_info
        summary['monitoring_duration'] = len(self.metrics_history) * self.sampling_interval
        
        return summary
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.stats:
            return 0.0
        
        scores = []
        
        # Latency score (lower is better)
        if 'latency' in self.stats and self.stats['latency']:
            avg_latency = np.mean(self.stats['latency'])
            latency_score = max(0, 100 - (avg_latency / self.thresholds.max_latency_ms) * 100)
            scores.append(latency_score)
        
        # Memory score (lower is better)
        if 'memory' in self.stats and self.stats['memory']:
            avg_memory = np.mean(self.stats['memory'])
            memory_score = max(0, 100 - (avg_memory / self.thresholds.max_memory_mb) * 100)
            scores.append(memory_score)
        
        # CPU score (lower is better)
        if 'cpu' in self.stats and self.stats['cpu']:
            avg_cpu = np.mean(self.stats['cpu'])
            cpu_score = max(0, 100 - (avg_cpu / self.thresholds.max_cpu_percent) * 100)
            scores.append(cpu_score)
        
        # Throughput score (higher is better)
        if 'throughput' in self.stats and self.stats['throughput']:
            avg_throughput = np.mean(self.stats['throughput'])
            throughput_score = min(100, (avg_throughput / self.thresholds.min_throughput) * 100)
            scores.append(throughput_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply performance optimizations to model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Convert BatchNorm to eval mode permanently
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        
        # Apply torch.jit optimizations if possible
        try:
            example_input = torch.randn(1, 3, 224, 224)  # Adjust based on model
            if hasattr(model, 'forward'):
                model = torch.jit.trace(model, example_input)
                self.logger.info("Applied TorchScript optimization")
        except Exception as e:
            self.logger.debug(f"Could not apply TorchScript: {e}")
        
        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            model = torch.jit.optimize_for_inference(model)
        
        return model
    
    def export_metrics(self, filepath: Union[str, Path]):
        """Export collected metrics to file."""
        filepath = Path(filepath)
        
        export_data = {
            'metrics_history': [
                {
                    'timestamp': m.timestamp,
                    'inference_latency': m.inference_latency,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'gpu_usage': m.gpu_usage,
                    'throughput': m.throughput,
                    'model_accuracy': m.model_accuracy,
                    'cache_hit_rate': m.cache_hit_rate,
                }
                for m in self.metrics_history
            ],
            'summary': self.get_performance_summary(),
            'device_info': self.device_info,
            'thresholds': {
                'max_latency_ms': self.thresholds.max_latency_ms,
                'max_memory_mb': self.thresholds.max_memory_mb,
                'max_cpu_percent': self.thresholds.max_cpu_percent,
                'max_gpu_percent': self.thresholds.max_gpu_percent,
                'min_throughput': self.thresholds.min_throughput,
                'min_accuracy': self.thresholds.min_accuracy,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")


class QuantumPerformanceIntegrator:
    """
    Performance monitoring for quantum-classical hybrid systems.
    
    Integrates classical ML performance with quantum computing metrics.
    """
    
    def __init__(self, monitor: RealTimePerformanceMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        self.quantum_metrics = {}
    
    def benchmark_quantum_circuit(
        self,
        circuit_runner: Callable,
        num_shots: int = 1000,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark quantum circuit execution.
        
        Args:
            circuit_runner: Function that executes quantum circuit
            num_shots: Number of quantum measurements
            num_runs: Number of benchmark runs
            
        Returns:
            Quantum performance metrics
        """
        execution_times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = circuit_runner(num_shots)
                end_time = time.perf_counter()
                execution_times.append((end_time - start_time) * 1000.0)  # ms
            except Exception as e:
                self.logger.error(f"Quantum circuit execution failed: {e}")
                continue
        
        if not execution_times:
            return {'error': 'No successful quantum executions'}
        
        metrics = {
            'mean_execution_time_ms': float(np.mean(execution_times)),
            'std_execution_time_ms': float(np.std(execution_times)),
            'min_execution_time_ms': float(np.min(execution_times)),
            'max_execution_time_ms': float(np.max(execution_times)),
            'quantum_volume': num_shots,
            'circuit_depth': getattr(circuit_runner, 'depth', 0),
            'success_rate': len(execution_times) / num_runs,
        }
        
        self.quantum_metrics.update(metrics)
        return metrics
    
    def get_hybrid_performance_report(self) -> Dict[str, Any]:
        """Get combined classical-quantum performance report."""
        classical_summary = self.monitor.get_performance_summary()
        
        report = {
            'classical_performance': classical_summary,
            'quantum_performance': self.quantum_metrics,
            'hybrid_efficiency': self._calculate_hybrid_efficiency(),
            'timestamp': time.time()
        }
        
        return report
    
    def _calculate_hybrid_efficiency(self) -> float:
        """Calculate efficiency of quantum-classical hybrid execution."""
        if not self.quantum_metrics or 'health_score' not in self.monitor.get_performance_summary():
            return 0.0
        
        classical_health = self.monitor.get_performance_summary().get('health_score', 0)
        quantum_success_rate = self.quantum_metrics.get('success_rate', 0) * 100
        
        # Weighted average (adjust weights based on workload)
        hybrid_efficiency = (classical_health * 0.7) + (quantum_success_rate * 0.3)
        return float(hybrid_efficiency)


def main():
    """Demonstrate advanced performance monitoring."""
    # Initialize monitor
    monitor = RealTimePerformanceMonitor(
        window_size=100,
        sampling_interval=0.5
    )
    
    # Add alert callback
    def alert_handler(alerts: List[str], metrics: PerformanceMetrics):
        print(f"⚠️ PERFORMANCE ALERT: {', '.join(alerts)}")
        print(f"Current metrics: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}MB")
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate some work
        time.sleep(5)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print("📊 Performance Summary:")
        print(json.dumps(summary, indent=2))
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()