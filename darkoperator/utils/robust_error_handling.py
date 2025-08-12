"""
Robust Error Handling and Recovery for Research Components.

Production-ready error handling, logging, and recovery mechanisms for:
- Advanced spectral neural operators
- Physics-informed quantum circuits
- Conservation-aware conformal prediction
- Multi-modal fusion systems

Ensures reliability and graceful degradation under failure conditions.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch module for compatibility
    class DummyTorch:
        @staticmethod
        def cuda_is_available(): return False
        @staticmethod
        def tensor(data): return data
        class cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def empty_cache(): pass
            @staticmethod
            def memory_allocated(): return 0
            @staticmethod
            def memory_reserved(): return 0
            @staticmethod
            def max_memory_allocated(): return 0
            @staticmethod
            def device_count(): return 0
            @staticmethod
            def current_device(): return 0
            @staticmethod
            def get_device_name(): return "CPU"
            @staticmethod
            def synchronize(): pass
    torch = DummyTorch()

try:
    import numpy as np
except ImportError:
    # Create dummy numpy for compatibility
    class DummyNumpy:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data): 
            if not data: return 0
            mean = sum(data) / len(data)
            return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        @staticmethod
        def min(data): return min(data) if data else 0
        @staticmethod
        def max(data): return max(data) if data else 0
    np = DummyNumpy()
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import traceback
import warnings
from functools import wraps
from contextlib import contextmanager
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil for compatibility
    class DummyPsutil:
        @staticmethod
        def cpu_count(): return 4
        @staticmethod
        def cpu_percent(): return 50.0
        @staticmethod
        def virtual_memory():
            class VMemory:
                percent = 50.0
                available = 8 * 1024**3  # 8GB
            return VMemory()
    psutil = DummyPsutil()

import gc
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for robust handling."""
    LOW = "low"              # Minor issues, continue execution
    MEDIUM = "medium"        # Significant issues, degrade gracefully
    HIGH = "high"           # Critical issues, attempt recovery
    CRITICAL = "critical"   # System failure, abort safely


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    input_shapes: Dict[str, Tuple] = field(default_factory=dict)
    device_info: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    physics_constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RobustPhysicsError(Exception):
    """Base exception for physics-informed components."""
    
    def __init__(self, message: str, error_context: ErrorContext, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.error_context = error_context
        self.severity = severity
        super().__init__(self.message)


class ConservationViolationError(RobustPhysicsError):
    """Error for physics conservation law violations."""
    pass


class QuantumCircuitError(RobustPhysicsError):
    """Error for quantum circuit operations."""
    pass


class SpectralOperatorError(RobustPhysicsError):
    """Error for spectral neural operator operations."""
    pass


class ConformalPredictionError(RobustPhysicsError):
    """Error for conformal prediction operations."""
    pass


def robust_physics_operation(
    max_retries: int = 3,
    fallback_strategy: str = 'graceful_degradation',
    log_errors: bool = True,
    monitor_resources: bool = True
):
    """
    Decorator for robust physics operations with error handling and recovery.
    
    Args:
        max_retries: Maximum number of retry attempts
        fallback_strategy: Strategy for handling failures ('graceful_degradation', 'simplified_model', 'abort')
        log_errors: Whether to log errors and recovery attempts
        monitor_resources: Whether to monitor system resources
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create error context
            error_context = ErrorContext(
                component=func.__module__ + '.' + func.__name__,
                operation=func.__name__
            )
            
            if monitor_resources:
                error_context.memory_usage = _get_memory_usage()
                error_context.device_info = _get_device_info()
            
            # Extract input shapes for debugging
            for i, arg in enumerate(args):
                if TORCH_AVAILABLE and hasattr(arg, 'shape'):
                    error_context.input_shapes[f'arg_{i}'] = tuple(arg.shape)
            
            for key, value in kwargs.items():
                if TORCH_AVAILABLE and hasattr(value, 'shape'):
                    error_context.input_shapes[key] = tuple(value.shape)
            
            last_exception = None
            
            # Retry loop
            for attempt in range(max_retries + 1):
                try:
                    # Monitor memory before operation
                    if monitor_resources and attempt > 0:
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Validate result if it's a physics operation
                    if hasattr(result, 'shape') or isinstance(result, dict):
                        _validate_physics_result(result, error_context)
                    
                    if log_errors and attempt > 0:
                        logger.info(f"Operation {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Determine error severity
                    severity = _classify_error_severity(e, error_context)
                    
                    if log_errors:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}")
                    
                    # Critical errors should not be retried
                    if severity == ErrorSeverity.CRITICAL:
                        break
                    
                    # Apply recovery strategies before retry
                    if attempt < max_retries:
                        _apply_recovery_strategy(e, error_context, attempt)
            
            # All retries failed, apply fallback strategy
            if log_errors:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}. Applying fallback strategy: {fallback_strategy}")
            
            return _apply_fallback_strategy(
                last_exception, error_context, fallback_strategy, func, args, kwargs
            )
        
        return wrapper
    return decorator


def _get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    memory_info = {
        'cpu_memory_percent': psutil.virtual_memory().percent,
        'cpu_memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        memory_info.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
        })
    
    return memory_info


def _get_device_info() -> Dict[str, Any]:
    """Get device and system information."""
    device_info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(),
        'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available()
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device_info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
            'cuda_device_name': torch.cuda.get_device_name()
        })
    
    return device_info


def _classify_error_severity(error: Exception, context: ErrorContext) -> ErrorSeverity:
    """Classify error severity based on error type and context."""
    
    # Critical errors that should abort immediately
    if isinstance(error, (RuntimeError, MemoryError)):
        if "out of memory" in str(error).lower():
            return ErrorSeverity.CRITICAL
        if "cuda" in str(error).lower() and "error" in str(error).lower():
            return ErrorSeverity.HIGH
    
    # Physics-specific errors
    if isinstance(error, ConservationViolationError):
        return ErrorSeverity.MEDIUM
    
    if isinstance(error, (QuantumCircuitError, SpectralOperatorError)):
        return ErrorSeverity.HIGH
    
    if isinstance(error, ConformalPredictionError):
        return ErrorSeverity.MEDIUM
    
    # Numerical errors
    if isinstance(error, (ValueError, ArithmeticError)):
        if "nan" in str(error).lower() or "inf" in str(error).lower():
            return ErrorSeverity.HIGH
        return ErrorSeverity.MEDIUM
    
    # Shape/dimension errors
    if "shape" in str(error).lower() or "size" in str(error).lower():
        return ErrorSeverity.MEDIUM
    
    # Default classification
    return ErrorSeverity.MEDIUM


def _apply_recovery_strategy(error: Exception, context: ErrorContext, attempt: int):
    """Apply recovery strategies before retry."""
    
    # Memory recovery
    if "memory" in str(error).lower():
        logger.info("Applying memory recovery strategy")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Reduce precision for numerical stability
    if isinstance(error, (ArithmeticError, ValueError)):
        if "nan" in str(error).lower() or "inf" in str(error).lower():
            logger.info("Detected numerical instability, will try with reduced precision")
    
    # Device recovery
    if TORCH_AVAILABLE and torch.cuda.is_available() and "cuda" in str(error).lower():
        logger.info("Applying CUDA recovery strategy")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Wait before retry with exponential backoff
    wait_time = 0.1 * (2 ** attempt)
    time.sleep(wait_time)


def _apply_fallback_strategy(
    error: Exception,
    context: ErrorContext,
    strategy: str,
    func: Callable,
    args: Tuple,
    kwargs: Dict
) -> Any:
    """Apply fallback strategy when all retries fail."""
    
    if strategy == 'graceful_degradation':
        return _graceful_degradation_fallback(error, context, func, args, kwargs)
    elif strategy == 'simplified_model':
        return _simplified_model_fallback(error, context, func, args, kwargs)
    elif strategy == 'abort':
        raise RobustPhysicsError(
            f"Operation {func.__name__} failed after all retries: {str(error)}",
            context,
            ErrorSeverity.CRITICAL
        )
    else:
        logger.error(f"Unknown fallback strategy: {strategy}")
        raise error


def _graceful_degradation_fallback(
    error: Exception,
    context: ErrorContext,
    func: Callable,
    args: Tuple,
    kwargs: Dict
) -> Any:
    """Implement graceful degradation fallback."""
    
    func_name = func.__name__
    
    # For neural operators, return simplified output
    if 'fno' in func_name.lower() or 'spectral' in func_name.lower():
        # Try to determine output shape from inputs
        for arg in args:
            if TORCH_AVAILABLE and hasattr(arg, 'shape'):
                if len(arg.shape) >= 3:  # Spatial data
                    # Return zero tensor with appropriate shape
                    output_shape = arg.shape[:-1]  # Remove feature dimension
                    logger.warning(f"Graceful degradation: returning zero tensor for {func_name}")
                    return torch.zeros(output_shape, device=arg.device, dtype=arg.dtype)
        
        # Fallback: return empty tensor or None
        if TORCH_AVAILABLE:
            return torch.tensor([])
        else:
            return None
    
    # For quantum circuits, return classical optimization result
    if 'quantum' in func_name.lower():
        logger.warning(f"Graceful degradation: falling back to classical optimization for {func_name}")
        return {
            'optimal_schedule': [],
            'energy': float('inf'),
            'optimization_time': 0.0,
            'fallback_used': True,
            'error_message': str(error)
        }
    
    # For conformal prediction, return conservative estimates
    if 'conformal' in func_name.lower() or 'predict' in func_name.lower():
        # Try to determine batch size from inputs
        batch_size = 1
        for arg in args:
            if TORCH_AVAILABLE and hasattr(arg, 'shape'):
                batch_size = arg.shape[0]
                break
        
        logger.warning(f"Graceful degradation: returning conservative predictions for {func_name}")
        if TORCH_AVAILABLE:
            return {
                'p_values': torch.ones(batch_size) * 0.5,  # Conservative p-values
                'is_anomaly': torch.zeros(batch_size, dtype=torch.bool),
                'fallback_used': True,
                'error_message': str(error)
            }
        else:
            return {
                'p_values': [0.5] * batch_size,  # Conservative p-values
                'is_anomaly': [False] * batch_size,
                'fallback_used': True,
                'error_message': str(error)
            }
    
    # Default fallback
    logger.warning(f"Graceful degradation: returning None for {func_name}")
    return None


def _simplified_model_fallback(
    error: Exception,
    context: ErrorContext,
    func: Callable,
    args: Tuple,
    kwargs: Dict
) -> Any:
    """Implement simplified model fallback."""
    
    # This would implement simplified versions of complex operations
    # For now, defer to graceful degradation
    logger.info("Simplified model fallback not implemented, using graceful degradation")
    return _graceful_degradation_fallback(error, context, func, args, kwargs)


def _validate_physics_result(result: Union[torch.Tensor, Dict], context: ErrorContext):
    """Validate physics results for consistency."""
    
    if TORCH_AVAILABLE and hasattr(result, 'isnan'):
        # Check for NaN or Inf values
        if torch.isnan(result).any():
            raise ConservationViolationError(
                "Result contains NaN values",
                context,
                ErrorSeverity.HIGH
            )
        
        if torch.isinf(result).any():
            raise ConservationViolationError(
                "Result contains infinite values", 
                context,
                ErrorSeverity.HIGH
            )
        
        # Check for unreasonable magnitudes
        if result.abs().max() > 1e10:
            warnings.warn(f"Result has very large values (max: {result.abs().max()})")
    
    elif isinstance(result, dict):
        # Validate dictionary results
        for key, value in result.items():
            if TORCH_AVAILABLE and hasattr(value, 'isnan'):
                _validate_physics_result(value, context)


@contextmanager
def robust_physics_context(
    component_name: str,
    operation_name: str,
    expected_memory_gb: float = 1.0,
    timeout_seconds: Optional[float] = None
):
    """
    Context manager for robust physics operations.
    
    Args:
        component_name: Name of the component being executed
        operation_name: Name of the operation
        expected_memory_gb: Expected memory usage
        timeout_seconds: Optional timeout for the operation
    """
    
    start_time = time.time()
    initial_memory = _get_memory_usage()
    
    context = ErrorContext(
        component=component_name,
        operation=operation_name,
        memory_usage=initial_memory
    )
    
    try:
        # Check available memory
        available_memory = initial_memory.get('cpu_memory_available_gb', 0)
        if available_memory < expected_memory_gb:
            logger.warning(f"Low memory available: {available_memory:.2f}GB < {expected_memory_gb:.2f}GB expected")
        
        logger.debug(f"Starting {component_name}.{operation_name}")
        
        yield context
        
        execution_time = time.time() - start_time
        final_memory = _get_memory_usage()
        
        # Log performance metrics
        logger.debug(f"Completed {component_name}.{operation_name} in {execution_time:.3f}s")
        
        # Check for memory leaks
        memory_increase = (
            final_memory.get('cpu_memory_percent', 0) - 
            initial_memory.get('cpu_memory_percent', 0)
        )
        
        if memory_increase > 10:  # 10% increase
            logger.warning(f"Significant memory increase detected: {memory_increase:.1f}%")
        
        # Check timeout
        if timeout_seconds and execution_time > timeout_seconds:
            logger.warning(f"Operation exceeded timeout: {execution_time:.3f}s > {timeout_seconds}s")
    
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in {component_name}.{operation_name} after {execution_time:.3f}s: {str(e)}")
        
        # Update context with error information
        context.memory_usage.update(_get_memory_usage())
        
        # Re-raise with enhanced context
        if isinstance(e, RobustPhysicsError):
            raise e
        else:
            severity = _classify_error_severity(e, context)
            raise RobustPhysicsError(str(e), context, severity) from e


class RobustPhysicsLogger:
    """Enhanced logger for physics operations with structured logging."""
    
    def __init__(self, component_name: str, log_file: Optional[Path] = None):
        self.component_name = component_name
        self.logger = logging.getLogger(f"physics.{component_name}")
        
        if log_file:
            self._setup_file_logging(log_file)
    
    def _setup_file_logging(self, log_file: Path):
        """Setup file logging with JSON format."""
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log operation start with context."""
        context = {
            'operation': operation,
            'component': self.component_name,
            'timestamp': time.time(),
            **kwargs
        }
        self.logger.info(f"Starting {operation}", extra={'context': context})
    
    def log_operation_success(self, operation: str, duration: float, **kwargs):
        """Log successful operation completion."""
        context = {
            'operation': operation,
            'component': self.component_name,
            'duration': duration,
            'status': 'success',
            **kwargs
        }
        self.logger.info(f"Completed {operation} in {duration:.3f}s", extra={'context': context})
    
    def log_operation_error(self, operation: str, error: Exception, **kwargs):
        """Log operation error with full context."""
        context = {
            'operation': operation,
            'component': self.component_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'status': 'error',
            **kwargs
        }
        self.logger.error(f"Error in {operation}: {str(error)}", extra={'context': context})
    
    def log_physics_violation(self, violation_type: str, severity: str, details: Dict[str, Any]):
        """Log physics constraint violations."""
        context = {
            'violation_type': violation_type,
            'severity': severity,
            'component': self.component_name,
            'details': details
        }
        self.logger.warning(f"Physics violation: {violation_type} ({severity})", extra={'context': context})
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Log performance metrics."""
        context = {
            'operation': operation,
            'component': self.component_name,
            'metrics': metrics
        }
        self.logger.info(f"Performance metrics for {operation}", extra={'context': context})


class ErrorRecoveryManager:
    """Manages error recovery strategies and tracks recovery statistics."""
    
    def __init__(self):
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'recovery_strategies_used': {},
            'component_error_counts': {}
        }
    
    def record_error(self, error: RobustPhysicsError, recovered: bool = False):
        """Record error occurrence and recovery status."""
        self.recovery_stats['total_errors'] += 1
        
        if recovered:
            self.recovery_stats['recovered_errors'] += 1
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.recovery_stats['critical_errors'] += 1
        
        # Track by component
        component = error.error_context.component
        if component not in self.recovery_stats['component_error_counts']:
            self.recovery_stats['component_error_counts'][component] = 0
        self.recovery_stats['component_error_counts'][component] += 1
    
    def record_recovery_strategy(self, strategy: str):
        """Record usage of recovery strategy."""
        if strategy not in self.recovery_stats['recovery_strategies_used']:
            self.recovery_stats['recovery_strategies_used'][strategy] = 0
        self.recovery_stats['recovery_strategies_used'][strategy] += 1
    
    def get_recovery_rate(self) -> float:
        """Get overall error recovery rate."""
        if self.recovery_stats['total_errors'] == 0:
            return 1.0
        return self.recovery_stats['recovered_errors'] / self.recovery_stats['total_errors']
    
    def get_component_reliability(self, component: str) -> float:
        """Get reliability score for specific component."""
        total_errors = self.recovery_stats['component_error_counts'].get(component, 0)
        if total_errors == 0:
            return 1.0
        
        # Simple reliability metric (1 - error_rate)
        # In practice, this would be more sophisticated
        error_rate = min(total_errors / 100, 1.0)  # Normalize by assumed operation count
        return max(1.0 - error_rate, 0.0)
    
    def generate_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        return {
            'overall_recovery_rate': self.get_recovery_rate(),
            'total_operations': self.recovery_stats['total_errors'],
            'critical_errors': self.recovery_stats['critical_errors'],
            'component_reliability': {
                component: self.get_component_reliability(component)
                for component in self.recovery_stats['component_error_counts']
            },
            'most_used_recovery_strategies': sorted(
                self.recovery_stats['recovery_strategies_used'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def get_system_health_check() -> Dict[str, Any]:
    """Perform comprehensive system health check."""
    
    health_status = {
        'timestamp': time.time(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    # Memory check
    memory_info = _get_memory_usage()
    memory_healthy = (
        memory_info.get('cpu_memory_percent', 0) < 90 and
        memory_info.get('cpu_memory_available_gb', 0) > 1.0
    )
    
    health_status['checks']['memory'] = {
        'status': 'healthy' if memory_healthy else 'warning',
        'details': memory_info
    }
    
    # GPU check (if available)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_memory_percent = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                if torch.cuda.max_memory_allocated() > 0 else 0
            )
            gpu_healthy = gpu_memory_percent < 90
            
            health_status['checks']['gpu'] = {
                'status': 'healthy' if gpu_healthy else 'warning',
                'details': {
                    'memory_percent': gpu_memory_percent,
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device()
                }
            }
        except Exception as e:
            health_status['checks']['gpu'] = {
                'status': 'error',
                'details': str(e)
            }
    
    # Error recovery system check
    recovery_rate = error_recovery_manager.get_recovery_rate()
    recovery_healthy = recovery_rate > 0.8  # 80% recovery rate threshold
    
    health_status['checks']['error_recovery'] = {
        'status': 'healthy' if recovery_healthy else 'warning',
        'details': {
            'recovery_rate': recovery_rate,
            'total_errors': error_recovery_manager.recovery_stats['total_errors'],
            'critical_errors': error_recovery_manager.recovery_stats['critical_errors']
        }
    }
    
    # Overall status
    check_statuses = [check['status'] for check in health_status['checks'].values()]
    if 'error' in check_statuses:
        health_status['overall_status'] = 'error'
    elif 'warning' in check_statuses:
        health_status['overall_status'] = 'warning'
    
    return health_status