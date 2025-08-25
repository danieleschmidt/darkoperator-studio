"""Advanced logging and monitoring utilities."""

import logging as base_logging  # Avoid naming conflict
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import traceback
import psutil
import torch


def setup_logging(level='INFO', log_file=None):
    """Setup basic logging configuration."""
    log_level = getattr(base_logging, level.upper(), base_logging.INFO)
    
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        base_logging.basicConfig(
            level=log_level,
            format=format_string,
            handlers=[
                base_logging.FileHandler(log_file),
                base_logging.StreamHandler()
            ]
        )
    else:
        base_logging.basicConfig(level=log_level, format=format_string)


def get_logger(name: str):
    """Get a logger instance."""
    return base_logging.getLogger(name)


class PhysicsLogger:
    """Enhanced logger for physics computations with performance monitoring."""
    
    def __init__(self, name: str = "darkoperator", log_dir: str = "./logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main logger
        self.logger = base_logging.getLogger(name)
        self.logger.setLevel(base_logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Performance tracking
        self.metrics = {}
        self.start_times = {}
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler
        console_handler = base_logging.StreamHandler(sys.stdout)
        console_handler.setLevel(base_logging.INFO)
        console_format = base_logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = base_logging.FileHandler(self.log_dir / f"{self.name}.log")
        file_handler.setLevel(base_logging.DEBUG)
        file_format = base_logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Error handler for critical issues
        error_handler = base_logging.FileHandler(self.log_dir / f"{self.name}_errors.log")
        error_handler.setLevel(base_logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
    
    def log_system_info(self):
        """Log system and environment information."""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info[f"gpu_{i}"] = f"{props.name} ({props.total_memory / (1024**3):.1f} GB)"
        
        self.logger.info(f"System Information: {json.dumps(info, indent=2)}")
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.logger.info(f"Starting {operation_name}")
        
        try:
            yield
            elapsed = time.time() - start_time
            self.metrics[operation_name] = elapsed
            self.logger.info(f"Completed {operation_name} in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Failed {operation_name} after {elapsed:.3f}s: {e}")
            raise
    
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                self.logger.debug(f"{prefix}GPU {i} Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
        
        ram = psutil.virtual_memory()
        self.logger.debug(f"{prefix}RAM Usage: {ram.percent:.1f}% ({ram.used / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB)")
    
    def log_physics_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log physics-specific metrics."""
        formatted_metrics = {k: f"{v:.6e}" if abs(v) < 1e-3 or abs(v) > 1e3 else f"{v:.4f}" 
                           for k, v in metrics.items()}
        self.logger.info(f"{prefix}Physics Metrics: {formatted_metrics}")
    
    def log_anomaly_detection(self, n_events: int, n_anomalies: int, alpha: float, top_p_values: list):
        """Log anomaly detection results."""
        detection_rate = n_anomalies / n_events if n_events > 0 else 0
        
        self.logger.info(f"Anomaly Detection Results:")
        self.logger.info(f"  Events analyzed: {n_events:,}")
        self.logger.info(f"  Anomalies found: {n_anomalies:,} ({detection_rate:.4%})")
        self.logger.info(f"  Significance level: α = {alpha:.2e}")
        
        if top_p_values:
            min_p = min(top_p_values)
            self.logger.info(f"  Most significant p-value: {min_p:.2e}")
            
            # Convert to sigma significance
            from scipy.stats import norm
            sigma = abs(norm.ppf(min_p / 2))  # Two-tailed test
            self.logger.info(f"  Maximum significance: {sigma:.2f}σ")
    
    def log_model_performance(self, model_name: str, inference_times: list, throughput: float):
        """Log model performance metrics."""
        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            
            self.logger.info(f"Model Performance ({model_name}):")
            self.logger.info(f"  Average inference time: {avg_time*1000:.2f} ms")
            self.logger.info(f"  Min/Max inference time: {min_time*1000:.2f}/{max_time*1000:.2f} ms")
            self.logger.info(f"  Throughput: {throughput:.1f} events/s")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with full traceback."""
        self.logger.error(f"Exception in {context}: {exception}")
        self.logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    def export_metrics(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export collected metrics to file."""
        if filepath is None:
            filepath = self.log_dir / f"{self.name}_metrics.json"
        
        export_data = {
            "timestamp": time.time(),
            "metrics": self.metrics,
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to: {filepath}")
        return export_data


class HealthMonitor:
    """System health monitoring for long-running physics computations."""
    
    def __init__(self, logger: PhysicsLogger):
        self.logger = logger
        self.start_time = time.time()
        self.last_memory_check = 0
        self.memory_threshold_gb = 0.9  # Alert if >90% memory used
    
    def check_system_health(self) -> bool:
        """Check system health and log warnings."""
        current_time = time.time()
        
        # Check memory every 60 seconds
        if current_time - self.last_memory_check > 60:
            self.last_memory_check = current_time
            
            # RAM check
            ram = psutil.virtual_memory()
            if ram.percent > self.memory_threshold_gb * 100:
                self.logger.logger.warning(f"High RAM usage: {ram.percent:.1f}%")
            
            # GPU memory check
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated = torch.cuda.memory_allocated(i)
                    if allocated / total_memory > self.memory_threshold_gb:
                        self.logger.logger.warning(f"High GPU {i} memory usage: {allocated/total_memory:.1%}")
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                self.logger.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        
        return True
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time