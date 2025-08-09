"""Advanced logging and monitoring configuration for DarkOperator."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import torch
from datetime import datetime
import traceback


class PhysicsFormatter(logging.Formatter):
    """Custom formatter for physics-specific logging."""
    
    def format(self, record):
        # Add physics context if available
        if hasattr(record, 'physics_context'):
            record.msg = f"[PHYSICS] {record.msg} | Context: {record.physics_context}"
        
        if hasattr(record, 'model_name'):
            record.msg = f"[{record.model_name}] {record.msg}"
        
        if hasattr(record, 'energy_scale'):
            record.msg = f"{record.msg} | Energy Scale: {record.energy_scale} GeV"
        
        return super().format(record)


class PerformanceMonitor:
    """Monitor performance metrics during computation."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics = {
            'start_time': None,
            'operations': 0,
            'total_time': 0,
            'memory_usage': [],
            'gpu_usage': [],
            'errors': [],
        }
        self.logger = logging.getLogger('performance')
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.metrics['start_time'] = time.time()
        self.logger.info("Performance monitoring started")
    
    def log_operation(self, operation_name: str, execution_time: float):
        """Log a single operation."""
        self.metrics['operations'] += 1
        self.metrics['total_time'] += execution_time
        
        # Log GPU memory if available
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            self.metrics['memory_usage'].append(memory_mb)
            
            if self.metrics['operations'] % self.log_interval == 0:
                self.logger.info(
                    f"Operation {operation_name}: {execution_time:.4f}s | "
                    f"GPU Memory: {memory_mb:.1f}MB | "
                    f"Total ops: {self.metrics['operations']}"
                )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
        }
        self.metrics['errors'].append(error_info)
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        elapsed_time = time.time() - (self.metrics['start_time'] or time.time())
        
        summary = {
            'elapsed_time_seconds': elapsed_time,
            'total_operations': self.metrics['operations'],
            'operations_per_second': self.metrics['operations'] / elapsed_time if elapsed_time > 0 else 0,
            'average_operation_time': self.metrics['total_time'] / self.metrics['operations'] if self.metrics['operations'] > 0 else 0,
            'error_count': len(self.metrics['errors']),
        }
        
        if self.metrics['memory_usage']:
            summary['peak_memory_mb'] = max(self.metrics['memory_usage'])
            summary['average_memory_mb'] = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
        
        return summary


class PhysicsLogger:
    """Specialized logger for physics computations."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.performance_monitor = PerformanceMonitor()
        
        if log_file:
            log_file.parent.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
            handler = logging.FileHandler(log_file)
            handler.setFormatter(PhysicsFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def log_physics_event(self, event_type: str, **kwargs):
        """Log physics-specific events."""
        extra = {'physics_context': event_type}
        extra.update(kwargs)
        
        self.logger.info(f"Physics event: {event_type}", extra=extra)
    
    def log_conservation_check(self, law: str, violation: float, tolerance: float):
        """Log conservation law checks."""
        status = "PASS" if violation <= tolerance else "FAIL"
        self.logger.info(
            f"Conservation check [{law}]: {status} (violation: {violation:.2e}, tolerance: {tolerance:.2e})",
            extra={'physics_context': 'conservation_law'}
        )
    
    def log_model_prediction(self, model_name: str, input_shape: tuple, output_shape: tuple, inference_time: float):
        """Log model prediction details."""
        self.logger.info(
            f"Model inference: {input_shape} -> {output_shape} in {inference_time:.4f}s",
            extra={'model_name': model_name}
        )
        self.performance_monitor.log_operation(f"{model_name}_inference", inference_time)
    
    def log_anomaly_detection(self, n_events: int, n_anomalies: int, threshold: float):
        """Log anomaly detection results."""
        anomaly_rate = n_anomalies / n_events if n_events > 0 else 0
        self.logger.info(
            f"Anomaly detection: {n_anomalies}/{n_events} events ({anomaly_rate:.1%}) above threshold {threshold}",
            extra={'physics_context': 'anomaly_detection'}
        )
    
    def log_training_epoch(self, epoch: int, train_loss: float, val_loss: float, learning_rate: float):
        """Log training epoch information."""
        self.logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={learning_rate:.2e}"
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_performance_monitoring: bool = True
) -> PhysicsLogger:
    """Setup comprehensive logging for DarkOperator."""
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Create physics logger
    physics_log_file = log_dir / "physics.log" if log_dir and enable_file_logging else None
    physics_logger = PhysicsLogger("darkoperator.physics", physics_log_file)
    
    if enable_performance_monitoring:
        physics_logger.performance_monitor.start_monitoring()
    
    # Configure specific loggers
    loggers = {
        'darkoperator.models': logging.getLogger('darkoperator.models'),
        'darkoperator.operators': logging.getLogger('darkoperator.operators'),
        'darkoperator.anomaly': logging.getLogger('darkoperator.anomaly'),
        'darkoperator.data': logging.getLogger('darkoperator.data'),
        'darkoperator.training': logging.getLogger('darkoperator.training'),
    }
    
    if log_dir and enable_file_logging:
        log_dir.mkdir(exist_ok=True, parents=True)
        
        for name, logger in loggers.items():
            filename = name.split('.')[-1] + '.log'
            handler = logging.FileHandler(log_dir / filename)
            handler.setFormatter(PhysicsFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
    
    return physics_logger


class ExperimentTracker:
    """Track experiment parameters and results."""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.start_time = datetime.now()
        self.config = {}
        self.results = {}
        self.artifacts = []
        
        self.logger = logging.getLogger(f'experiment.{experiment_name}')
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config.update(config)
        self.logger.info(f"Experiment configuration: {config}")
    
    def log_result(self, metric_name: str, value: Any, step: Optional[int] = None):
        """Log experiment results."""
        if metric_name not in self.results:
            self.results[metric_name] = []
        
        result_entry = {
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        self.results[metric_name].append(result_entry)
        
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Result {metric_name}: {value}{step_str}")
    
    def save_artifact(self, artifact_name: str, artifact_data: Any):
        """Save experiment artifact."""
        artifact_path = self.output_dir / f"{artifact_name}.json"
        
        try:
            with open(artifact_path, 'w') as f:
                if isinstance(artifact_data, torch.Tensor):
                    json.dump(artifact_data.cpu().numpy().tolist(), f, indent=2)
                else:
                    json.dump(artifact_data, f, indent=2, default=str)
            
            self.artifacts.append({
                'name': artifact_name,
                'path': str(artifact_path),
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved artifact: {artifact_name} -> {artifact_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
    
    def save_experiment_summary(self):
        """Save complete experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results,
            'artifacts': self.artifacts,
        }
        
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Experiment summary saved to {summary_path}")


def test_logging_system():
    """Test the logging and monitoring system."""
    print("Testing logging and monitoring system...")
    
    # Setup logging
    log_dir = Path("test_logs")
    physics_logger = setup_logging(log_level="DEBUG", log_dir=log_dir)
    
    # Test physics logging
    physics_logger.log_physics_event("collision_simulation", energy=13000, particles=1000)
    physics_logger.log_conservation_check("energy", 1e-8, 1e-6)
    physics_logger.log_model_prediction("CalorimeterOperator", (32, 1, 50, 50), (32, 1, 50, 50), 0.15)
    physics_logger.log_anomaly_detection(10000, 15, 0.95)
    physics_logger.log_training_epoch(1, 0.1234, 0.0987, 0.001)
    
    # Test experiment tracker
    tracker = ExperimentTracker("test_experiment", log_dir / "experiments")
    tracker.log_config({"model": "FourierNeuralOperator", "lr": 0.001, "batch_size": 32})
    tracker.log_result("accuracy", 0.95, step=100)
    tracker.log_result("loss", 0.05, step=100)
    tracker.save_artifact("test_data", {"values": [1, 2, 3, 4, 5]})
    tracker.save_experiment_summary()
    
    # Test performance monitoring
    performance_summary = physics_logger.performance_monitor.get_summary()
    print(f"Performance summary: {performance_summary}")
    
    print("✅ Logging system test completed!")


if __name__ == "__main__":
    test_logging_system()