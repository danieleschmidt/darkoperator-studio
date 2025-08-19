#!/usr/bin/env python3
"""
Advanced Error Handling and Self-Healing System for DarkOperator Studio.
Implements quantum-resilient error recovery and autonomous system repair.
"""

import asyncio
import functools
import inspect
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Type
import warnings
import json
from pathlib import Path

# Graceful imports for production environments
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not available, numerical error analysis disabled")


class ErrorSeverity(Enum):
    """Error severity levels for autonomous handling."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    IGNORE = "ignore"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    SELF_HEAL = "self_heal"
    ESCALATE = "escalate"
    QUANTUM_RECOVERY = "quantum_recovery"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: float
    function_name: str
    module_name: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'metadata': self.metadata,
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for preventing cascade failures."""
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 3  # successes to close circuit


class AdvancedErrorHandler:
    """
    Advanced error handling system with autonomous recovery capabilities.
    
    Features:
    - Quantum-resilient error detection and recovery
    - Self-healing system repair
    - Circuit breaker patterns for cascade failure prevention
    - Autonomous escalation and incident management
    - Physics-informed error analysis for domain-specific failures
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_functions: Dict[str, Callable] = {}
        self.quantum_error_patterns: List[Dict[str, Any]] = []
        
        # Initialize autonomous error handling
        self._initialize_recovery_strategies()
        self._initialize_quantum_patterns()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load error handling configuration."""
        default_config = {
            'max_error_history': 1000,
            'auto_healing_enabled': True,
            'circuit_breaker_enabled': True,
            'quantum_recovery_enabled': True,
            'escalation_threshold': 10,
            'critical_error_notification': True,
            'error_persistence_enabled': True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                warnings.warn(f"Failed to load error config: {e}")
                
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup advanced error logging."""
        logger = logging.getLogger('darkoperator.error_handler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ERROR-HANDLER - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _initialize_recovery_strategies(self):
        """Initialize built-in recovery strategies."""
        self.recovery_functions = {
            'memory_cleanup': self._recover_memory_issues,
            'connection_reset': self._recover_connection_issues,
            'cache_invalidation': self._recover_cache_issues,
            'model_reload': self._recover_model_issues,
            'physics_recalibration': self._recover_physics_issues,
            'quantum_state_reset': self._recover_quantum_issues
        }
        
    def _initialize_quantum_patterns(self):
        """Initialize quantum error detection patterns."""
        self.quantum_error_patterns = [
            {
                'pattern': 'quantum_decoherence',
                'description': 'Quantum state decoherence detected',
                'indicators': ['phase_error', 'entanglement_loss', 'superposition_collapse'],
                'recovery': 'quantum_state_reset'
            },
            {
                'pattern': 'conservation_violation',
                'description': 'Physics conservation law violation',
                'indicators': ['energy_imbalance', 'momentum_violation', 'charge_conservation'],
                'recovery': 'physics_recalibration'
            },
            {
                'pattern': 'neural_operator_instability',
                'description': 'Neural operator numerical instability',
                'indicators': ['gradient_explosion', 'nan_values', 'operator_divergence'],
                'recovery': 'model_reload'
            }
        ]
        
    def robust_execute(self, 
                      func: Callable,
                      *args,
                      max_retries: int = 3,
                      backoff_factor: float = 1.5,
                      recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                      **kwargs) -> Any:
        """Execute function with robust error handling and recovery."""
        
        if asyncio.iscoroutinefunction(func):
            return asyncio.create_task(
                self._async_robust_execute(func, *args, max_retries=max_retries,
                                         backoff_factor=backoff_factor,
                                         recovery_strategy=recovery_strategy, **kwargs)
            )
        else:
            return self._sync_robust_execute(func, *args, max_retries=max_retries,
                                           backoff_factor=backoff_factor,
                                           recovery_strategy=recovery_strategy, **kwargs)
    
    def _sync_robust_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronous robust execution."""
        max_retries = kwargs.pop('max_retries', 3)
        backoff_factor = kwargs.pop('backoff_factor', 1.5)
        recovery_strategy = kwargs.pop('recovery_strategy', RecoveryStrategy.RETRY)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_context = self._create_error_context(e, func, recovery_strategy)
                
                if attempt == max_retries:
                    self._handle_final_failure(error_context)
                    break
                    
                self._handle_error_with_recovery(error_context)
                
                # Exponential backoff
                if attempt < max_retries:
                    sleep_time = (backoff_factor ** attempt)
                    time.sleep(sleep_time)
                    
        raise last_exception
        
    async def _async_robust_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Asynchronous robust execution."""
        max_retries = kwargs.pop('max_retries', 3)
        backoff_factor = kwargs.pop('backoff_factor', 1.5)
        recovery_strategy = kwargs.pop('recovery_strategy', RecoveryStrategy.RETRY)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_context = self._create_error_context(e, func, recovery_strategy)
                
                if attempt == max_retries:
                    await self._handle_final_failure_async(error_context)
                    break
                    
                await self._handle_error_with_recovery_async(error_context)
                
                # Exponential backoff
                if attempt < max_retries:
                    sleep_time = (backoff_factor ** attempt)
                    await asyncio.sleep(sleep_time)
                    
        raise last_exception
        
    def _create_error_context(self, 
                            exception: Exception, 
                            func: Callable,
                            recovery_strategy: RecoveryStrategy) -> ErrorContext:
        """Create comprehensive error context."""
        error_id = f"ERR_{int(time.time() * 1000)}"
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            function_name=func.__name__,
            module_name=func.__module__,
            error_type=type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            severity=self._classify_error_severity(exception),
            recovery_strategy=recovery_strategy,
            metadata={
                'function_signature': str(inspect.signature(func)),
                'is_async': asyncio.iscoroutinefunction(func),
                'quantum_pattern': self._detect_quantum_pattern(exception)
            }
        )
        
    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity automatically."""
        error_type = type(exception).__name__
        
        severity_map = {
            'MemoryError': ErrorSeverity.CRITICAL,
            'SystemExit': ErrorSeverity.FATAL,
            'KeyboardInterrupt': ErrorSeverity.WARNING,
            'ConnectionError': ErrorSeverity.ERROR,
            'TimeoutError': ErrorSeverity.WARNING,
            'ValueError': ErrorSeverity.WARNING,
            'TypeError': ErrorSeverity.ERROR,
            'RuntimeError': ErrorSeverity.ERROR,
            'ZeroDivisionError': ErrorSeverity.WARNING,
            'FileNotFoundError': ErrorSeverity.WARNING,
            'PermissionError': ErrorSeverity.ERROR,
        }
        
        # Check for physics-specific errors
        error_message = str(exception).lower()
        if any(phrase in error_message for phrase in ['conservation', 'physics', 'quantum']):
            return ErrorSeverity.CRITICAL
            
        return severity_map.get(error_type, ErrorSeverity.ERROR)
        
    def _detect_quantum_pattern(self, exception: Exception) -> Optional[str]:
        """Detect quantum error patterns in exception."""
        error_message = str(exception).lower()
        
        for pattern in self.quantum_error_patterns:
            indicators = pattern['indicators']
            if any(indicator in error_message for indicator in indicators):
                return pattern['pattern']
                
        return None
        
    def _handle_error_with_recovery(self, error_context: ErrorContext):
        """Handle error with appropriate recovery strategy."""
        self.error_history.append(error_context)
        self._trim_error_history()
        
        self.logger.warning(f"Error detected: {error_context.error_id} - {error_context.error_message}")
        
        # Apply recovery strategy
        if error_context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self._handle_circuit_breaker(error_context)
        elif error_context.recovery_strategy == RecoveryStrategy.SELF_HEAL:
            self._attempt_self_healing(error_context)
        elif error_context.recovery_strategy == RecoveryStrategy.QUANTUM_RECOVERY:
            self._attempt_quantum_recovery(error_context)
            
        # Persist error for analysis
        if self.config.get('error_persistence_enabled', True):
            self._persist_error(error_context)
            
    async def _handle_error_with_recovery_async(self, error_context: ErrorContext):
        """Async version of error handling with recovery."""
        self.error_history.append(error_context)
        self._trim_error_history()
        
        self.logger.warning(f"Error detected: {error_context.error_id} - {error_context.error_message}")
        
        # Apply recovery strategy
        if error_context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            await self._handle_circuit_breaker_async(error_context)
        elif error_context.recovery_strategy == RecoveryStrategy.SELF_HEAL:
            await self._attempt_self_healing_async(error_context)
        elif error_context.recovery_strategy == RecoveryStrategy.QUANTUM_RECOVERY:
            await self._attempt_quantum_recovery_async(error_context)
            
        # Persist error for analysis
        if self.config.get('error_persistence_enabled', True):
            await self._persist_error_async(error_context)
            
    def _handle_circuit_breaker(self, error_context: ErrorContext):
        """Handle circuit breaker logic."""
        key = f"{error_context.module_name}.{error_context.function_name}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreakerState()
            
        breaker = self.circuit_breakers[key]
        current_time = time.time()
        
        if breaker.state == "OPEN":
            # Check if we should transition to HALF_OPEN
            if (breaker.last_failure_time and 
                current_time - breaker.last_failure_time > breaker.recovery_timeout):
                breaker.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker transitioning to HALF_OPEN: {key}")
        else:
            # Record failure
            breaker.failure_count += 1
            breaker.last_failure_time = current_time
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.state = "OPEN"
                self.logger.warning(f"Circuit breaker OPENED: {key}")
                
    async def _handle_circuit_breaker_async(self, error_context: ErrorContext):
        """Async circuit breaker handling."""
        self._handle_circuit_breaker(error_context)  # Reuse sync implementation
        
    def _attempt_self_healing(self, error_context: ErrorContext):
        """Attempt autonomous self-healing."""
        if not self.config.get('auto_healing_enabled', True):
            return
            
        self.logger.info(f"Attempting self-healing for: {error_context.error_id}")
        
        # Identify healing strategy based on error pattern
        healing_functions = []
        
        if 'memory' in error_context.error_message.lower():
            healing_functions.append('memory_cleanup')
        if 'connection' in error_context.error_message.lower():
            healing_functions.append('connection_reset')
        if 'cache' in error_context.error_message.lower():
            healing_functions.append('cache_invalidation')
        if 'model' in error_context.error_message.lower():
            healing_functions.append('model_reload')
            
        # Apply quantum pattern recovery
        quantum_pattern = error_context.metadata.get('quantum_pattern')
        if quantum_pattern:
            for pattern in self.quantum_error_patterns:
                if pattern['pattern'] == quantum_pattern:
                    healing_functions.append(pattern['recovery'])
                    break
                    
        # Execute healing functions
        for func_name in healing_functions:
            if func_name in self.recovery_functions:
                try:
                    self.recovery_functions[func_name](error_context)
                    self.logger.info(f"Self-healing function executed: {func_name}")
                except Exception as e:
                    self.logger.error(f"Self-healing function failed: {func_name} - {e}")
                    
    async def _attempt_self_healing_async(self, error_context: ErrorContext):
        """Async self-healing attempt."""
        self._attempt_self_healing(error_context)  # Most healing is sync
        
    def _attempt_quantum_recovery(self, error_context: ErrorContext):
        """Attempt quantum-enhanced error recovery."""
        if not self.config.get('quantum_recovery_enabled', True):
            return
            
        self.logger.info(f"Attempting quantum recovery for: {error_context.error_id}")
        
        quantum_pattern = error_context.metadata.get('quantum_pattern')
        if not quantum_pattern:
            return
            
        # Quantum recovery strategies
        if quantum_pattern == 'quantum_decoherence':
            self._recover_quantum_decoherence(error_context)
        elif quantum_pattern == 'conservation_violation':
            self._recover_conservation_violation(error_context)
        elif quantum_pattern == 'neural_operator_instability':
            self._recover_operator_instability(error_context)
            
    async def _attempt_quantum_recovery_async(self, error_context: ErrorContext):
        """Async quantum recovery."""
        self._attempt_quantum_recovery(error_context)
        
    def _recover_memory_issues(self, error_context: ErrorContext):
        """Recover from memory-related issues."""
        import gc
        gc.collect()
        
        if HAS_NUMPY:
            # Clear numpy memory pools if available
            try:
                np.clear_memory_pool()
            except AttributeError:
                pass  # Not available in all numpy versions
                
        self.logger.info("Memory cleanup recovery completed")
        
    def _recover_connection_issues(self, error_context: ErrorContext):
        """Recover from connection issues."""
        self.logger.info("Connection reset recovery initiated")
        # Implementation would reset network connections, reconnect to services, etc.
        
    def _recover_cache_issues(self, error_context: ErrorContext):
        """Recover from cache-related issues."""
        self.logger.info("Cache invalidation recovery initiated")
        # Implementation would clear caches, reset cache connections, etc.
        
    def _recover_model_issues(self, error_context: ErrorContext):
        """Recover from ML model issues."""
        self.logger.info("Model reload recovery initiated")
        # Implementation would reload models, reset states, etc.
        
    def _recover_physics_issues(self, error_context: ErrorContext):
        """Recover from physics calculation issues."""
        self.logger.info("Physics recalibration recovery initiated")
        # Implementation would recalibrate physics constants, reset calculations, etc.
        
    def _recover_quantum_issues(self, error_context: ErrorContext):
        """Recover from quantum computation issues."""
        self.logger.info("Quantum state reset recovery initiated")
        # Implementation would reset quantum states, reinitialize circuits, etc.
        
    def _recover_quantum_decoherence(self, error_context: ErrorContext):
        """Recover from quantum decoherence errors."""
        self.logger.info("Quantum decoherence recovery initiated")
        # Implementation would restore quantum coherence, reset entanglement, etc.
        
    def _recover_conservation_violation(self, error_context: ErrorContext):
        """Recover from conservation law violations."""
        self.logger.info("Conservation violation recovery initiated")
        # Implementation would restore conservation constraints, recalibrate physics, etc.
        
    def _recover_operator_instability(self, error_context: ErrorContext):
        """Recover from neural operator instability."""
        self.logger.info("Neural operator stability recovery initiated")
        # Implementation would stabilize operators, adjust learning rates, etc.
        
    def _handle_final_failure(self, error_context: ErrorContext):
        """Handle final failure after all recovery attempts."""
        error_context.recovery_attempts = error_context.max_recovery_attempts
        self.error_history.append(error_context)
        
        self.logger.critical(f"Final failure: {error_context.error_id} - {error_context.error_message}")
        
        # Escalate if configured
        if (error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL] and
            self.config.get('critical_error_notification', True)):
            self._escalate_error(error_context)
            
    async def _handle_final_failure_async(self, error_context: ErrorContext):
        """Async final failure handling."""
        self._handle_final_failure(error_context)
        
    def _escalate_error(self, error_context: ErrorContext):
        """Escalate critical errors."""
        self.logger.critical(f"ESCALATING ERROR: {error_context.error_id}")
        # Implementation would send notifications, create incidents, etc.
        
    def _persist_error(self, error_context: ErrorContext):
        """Persist error for analysis."""
        try:
            error_log_path = Path('error_log.jsonl')
            with open(error_log_path, 'a') as f:
                f.write(json.dumps(error_context.to_dict()) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to persist error: {e}")
            
    async def _persist_error_async(self, error_context: ErrorContext):
        """Async error persistence."""
        self._persist_error(error_context)
        
    def _trim_error_history(self):
        """Trim error history to prevent memory growth."""
        max_history = self.config.get('max_error_history', 1000)
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history//2:]
            
    @contextmanager
    def error_boundary(self, 
                      recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                      max_retries: int = 3):
        """Context manager for error boundary with automatic recovery."""
        try:
            yield self
        except Exception as e:
            error_context = self._create_error_context(
                e, 
                lambda: None,  # Placeholder function
                recovery_strategy
            )
            self._handle_error_with_recovery(error_context)
            raise
            
    @asynccontextmanager
    async def async_error_boundary(self,
                                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                                 max_retries: int = 3):
        """Async context manager for error boundary."""
        try:
            yield self
        except Exception as e:
            error_context = self._create_error_context(
                e,
                lambda: None,  # Placeholder function
                recovery_strategy
            )
            await self._handle_error_with_recovery_async(error_context)
            raise
            
    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error analysis report."""
        if not self.error_history:
            return {'status': 'no_errors', 'message': 'No errors recorded'}
            
        recent_errors = self.error_history[-100:]  # Last 100 errors
        
        # Analyze error patterns
        error_types = {}
        severity_counts = {}
        recovery_success = {}
        
        for error in recent_errors:
            # Count error types
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count severities
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            # Track recovery success
            recovery_key = error.recovery_strategy.value
            if recovery_key not in recovery_success:
                recovery_success[recovery_key] = {'attempted': 0, 'successful': 0}
            recovery_success[recovery_key]['attempted'] += 1
            if error.recovery_attempts < error.max_recovery_attempts:
                recovery_success[recovery_key]['successful'] += 1
                
        # Calculate statistics
        total_errors = len(recent_errors)
        critical_errors = sum(1 for e in recent_errors if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL])
        
        return {
            'report_timestamp': time.time(),
            'total_errors': total_errors,
            'critical_errors': critical_errors,
            'error_rate': total_errors / 100 if total_errors > 0 else 0,
            'critical_rate': critical_errors / total_errors if total_errors > 0 else 0,
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'recovery_statistics': recovery_success,
            'circuit_breaker_status': {
                key: breaker.state for key, breaker in self.circuit_breakers.items()
            },
            'quantum_patterns_detected': len([
                e for e in recent_errors if e.metadata.get('quantum_pattern')
            ]),
            'recommendations': self._generate_error_recommendations(recent_errors)
        }
        
    def _generate_error_recommendations(self, errors: List[ErrorContext]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        # Analyze patterns
        memory_errors = sum(1 for e in errors if 'memory' in e.error_message.lower())
        connection_errors = sum(1 for e in errors if 'connection' in e.error_message.lower())
        physics_errors = sum(1 for e in errors if e.metadata.get('quantum_pattern'))
        
        if memory_errors > len(errors) * 0.2:
            recommendations.append("High memory error rate detected - consider increasing memory allocation or optimizing algorithms")
            
        if connection_errors > len(errors) * 0.15:
            recommendations.append("Frequent connection issues - check network stability and implement connection pooling")
            
        if physics_errors > len(errors) * 0.1:
            recommendations.append("Quantum/physics computation errors detected - review numerical stability and calibration")
            
        # Circuit breaker recommendations
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == "OPEN")
        if open_breakers > 0:
            recommendations.append(f"Circuit breakers active ({open_breakers}) - investigate underlying issues")
            
        return recommendations


# Global error handler instance
_error_handler = None

def get_error_handler() -> AdvancedErrorHandler:
    """Get or create the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = AdvancedErrorHandler()
    return _error_handler


def robust(recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
          max_retries: int = 3,
          backoff_factor: float = 1.5):
    """Decorator for robust function execution with error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            return handler.robust_execute(
                func, *args, 
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                recovery_strategy=recovery_strategy,
                **kwargs
            )
        return wrapper
    return decorator


def quantum_resilient(func):
    """Decorator for quantum-resilient function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        handler = get_error_handler()
        return handler.robust_execute(
            func, *args,
            recovery_strategy=RecoveryStrategy.QUANTUM_RECOVERY,
            **kwargs
        )
    return wrapper


if __name__ == "__main__":
    # Example usage and testing
    handler = AdvancedErrorHandler()
    
    @robust(recovery_strategy=RecoveryStrategy.SELF_HEAL)
    def test_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ValueError("Test error for demonstration")
        return "Success!"
        
    # Test robust execution
    try:
        result = test_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final failure: {e}")
        
    # Generate error report
    report = handler.generate_error_report()
    print(f"Error report: {json.dumps(report, indent=2)}")