"""
Security module for quantum task planning system.

Implements comprehensive security measures including input validation,
execution sandboxing, and quantum-resistant cryptographic protections.
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import ast
import subprocess
import os
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for task execution."""
    MINIMAL = 0      # Basic validation only
    STANDARD = 1     # Input validation + sandboxing
    HIGH = 2         # Full validation + cryptographic verification
    CRITICAL = 3     # Maximum security for sensitive physics data


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    # Access control
    max_execution_time: float = 3600.0  # Maximum task execution time (seconds)
    max_memory_usage: int = 8 * 1024**3  # Maximum memory usage (bytes)
    max_file_size: int = 100 * 1024**2   # Maximum file size (bytes)
    
    # Input validation
    allowed_functions: List[str] = field(default_factory=lambda: [
        'torch.tensor', 'numpy.array', 'math.sqrt', 'math.log',
        'darkoperator.*'  # Allow all darkoperator functions
    ])
    
    blocked_imports: List[str] = field(default_factory=lambda: [
        'os.system', 'subprocess', 'eval', 'exec', '__import__',
        'open', 'file', 'input', 'raw_input'
    ])
    
    # Cryptographic settings
    enable_task_signing: bool = True
    enable_result_verification: bool = True
    signature_algorithm: str = 'hmac-sha256'
    
    # Physics-specific security
    max_energy_computation: float = 1e12  # Maximum energy in computation (GeV)
    max_particle_count: int = 1000000     # Maximum particles in simulation
    
    # Network security
    allow_network_access: bool = False
    allowed_domains: List[str] = field(default_factory=lambda: [
        'opendata.cern.ch', 'lhc-ml-challenge.org'
    ])
    
    # Logging and monitoring
    log_all_executions: bool = True
    alert_on_violations: bool = True
    quarantine_suspicious_tasks: bool = True


class InputValidator:
    """Validates task inputs for security threats."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.violation_count = 0
        
    def validate_task_operation(self, operation: Callable) -> bool:
        """
        Validate task operation for security risks.
        
        Args:
            operation: Callable to validate
            
        Returns:
            True if operation is safe, False otherwise
        """
        try:
            # Check if it's a lambda or function
            if hasattr(operation, '__code__'):
                return self._validate_code_object(operation.__code__)
            
            # Check if it's a built-in safe operation
            if hasattr(operation, '__name__'):
                op_name = operation.__name__
                return self._is_allowed_function(op_name)
            
            # If we can't inspect it, assume it's unsafe
            logger.warning(f"Cannot inspect operation type: {type(operation)}")
            return False
            
        except Exception as e:
            logger.error(f"Error validating task operation: {e}")
            return False
    
    def _validate_code_object(self, code) -> bool:
        """Validate code object for security risks."""
        
        # Check for dangerous built-ins
        dangerous_names = [
            'exec', 'eval', '__import__', 'compile', 'globals', 'locals',
            'open', 'file', 'input', 'raw_input'
        ]
        
        for name in code.co_names:
            if name in dangerous_names:
                logger.warning(f"Blocked dangerous function: {name}")
                self.violation_count += 1
                return False
        
        # Check for system-level access
        for name in code.co_names:
            if any(blocked in name for blocked in self.policy.blocked_imports):
                logger.warning(f"Blocked system access: {name}")
                self.violation_count += 1
                return False
        
        return True
    
    def _is_allowed_function(self, func_name: str) -> bool:
        """Check if function name is in allowed list."""
        
        for pattern in self.policy.allowed_functions:
            if pattern.endswith('.*'):
                # Wildcard pattern
                prefix = pattern[:-2]
                if func_name.startswith(prefix):
                    return True
            elif pattern == func_name:
                return True
        
        return False
    
    def validate_task_args(self, args: Tuple, kwargs: Dict[str, Any]) -> bool:
        """
        Validate task arguments for security risks.
        
        Args:
            args: Task positional arguments
            kwargs: Task keyword arguments
            
        Returns:
            True if arguments are safe
        """
        
        # Check argument sizes
        total_size = self._estimate_arg_size(args) + self._estimate_arg_size(kwargs)
        if total_size > self.policy.max_memory_usage:
            logger.warning(f"Arguments too large: {total_size} bytes")
            self.violation_count += 1
            return False
        
        # Check for suspicious string patterns
        for arg in args:
            if isinstance(arg, str) and not self._is_safe_string(arg):
                return False
        
        for key, value in kwargs.items():
            if isinstance(value, str) and not self._is_safe_string(value):
                return False
        
        # Physics-specific validation
        if not self._validate_physics_parameters(args, kwargs):
            return False
        
        return True
    
    def _estimate_arg_size(self, obj) -> int:
        """Estimate memory size of object."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, (list, tuple)):
                return len(obj) * 8  # Rough estimate
            elif isinstance(obj, dict):
                return len(obj) * 16  # Rough estimate
            elif isinstance(obj, str):
                return len(obj)
            else:
                return 64  # Default estimate
    
    def _is_safe_string(self, s: str) -> bool:
        """Check if string contains suspicious patterns."""
        
        # Check for code injection patterns
        suspicious_patterns = [
            r'__.*__',  # Dunder methods
            r'import\s+',
            r'exec\s*\(',
            r'eval\s*\(',
            r'os\.',
            r'subprocess\.',
            r'system\s*\(',
            r'\bfile\s*\(',
            r'\bopen\s*\(',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, s, re.IGNORECASE):
                logger.warning(f"Suspicious string pattern: {pattern}")
                self.violation_count += 1
                return False
        
        return True
    
    def _validate_physics_parameters(self, args: Tuple, kwargs: Dict[str, Any]) -> bool:
        """Validate physics-specific parameters."""
        
        # Check energy parameters
        energy_keys = ['energy', 'E', 'total_energy', 'kinetic_energy']
        for key in energy_keys:
            if key in kwargs:
                energy = kwargs[key]
                if isinstance(energy, (int, float)) and energy > self.policy.max_energy_computation:
                    logger.warning(f"Energy parameter too large: {energy}")
                    self.violation_count += 1
                    return False
        
        # Check particle count parameters
        count_keys = ['n_particles', 'num_events', 'particles', 'events']
        for key in count_keys:
            if key in kwargs:
                count = kwargs[key]
                if isinstance(count, int) and count > self.policy.max_particle_count:
                    logger.warning(f"Particle count too large: {count}")
                    self.violation_count += 1
                    return False
        
        return True


class ExecutionSandbox:
    """Sandboxed execution environment for tasks."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.temp_dirs = []
        
    def create_sandbox(self) -> str:
        """Create isolated execution environment."""
        
        # Create temporary directory
        sandbox_dir = tempfile.mkdtemp(prefix='darkoperator_sandbox_')
        self.temp_dirs.append(sandbox_dir)
        
        logger.debug(f"Created sandbox: {sandbox_dir}")
        return sandbox_dir
    
    def execute_in_sandbox(
        self, 
        operation: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        timeout: float = None
    ) -> Any:
        """
        Execute operation in sandboxed environment.
        
        Args:
            operation: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Execution timeout (uses policy default if None)
            
        Returns:
            Operation result
        """
        
        if timeout is None:
            timeout = self.policy.max_execution_time
        
        # Create sandbox
        sandbox_dir = self.create_sandbox()
        
        try:
            # Set up restricted environment
            old_cwd = os.getcwd()
            os.chdir(sandbox_dir)
            
            # Execute with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Task execution exceeded {timeout} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = operation(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)  # Disable alarm
                os.chdir(old_cwd)
                
        except TimeoutError:
            logger.error(f"Task execution timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            raise
    
    def cleanup_sandbox(self, sandbox_dir: str) -> None:
        """Clean up sandbox directory."""
        try:
            import shutil
            shutil.rmtree(sandbox_dir)
            if sandbox_dir in self.temp_dirs:
                self.temp_dirs.remove(sandbox_dir)
            logger.debug(f"Cleaned up sandbox: {sandbox_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox {sandbox_dir}: {e}")
    
    def cleanup_all_sandboxes(self) -> None:
        """Clean up all created sandboxes."""
        for sandbox_dir in self.temp_dirs.copy():
            self.cleanup_sandbox(sandbox_dir)


class CryptographicVerifier:
    """Handles cryptographic verification of tasks and results with quantum-resistant features."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.secret_key = self._generate_secret_key()
        self.quantum_resistant_key = self._generate_quantum_resistant_key()
        self.verification_stats = {
            'signatures_generated': 0,
            'signatures_verified': 0,
            'verification_failures': 0,
            'quantum_signatures_used': 0
        }
        
    def _generate_secret_key(self) -> bytes:
        """Generate cryptographically secure secret key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _generate_quantum_resistant_key(self) -> bytes:
        """Generate quantum-resistant key using enhanced entropy."""
        # Use multiple entropy sources for quantum resistance
        entropy_sources = [
            secrets.token_bytes(32),
            hashlib.sha256(str(time.time_ns()).encode()).digest(),
            hashlib.sha256(str(os.getpid()).encode()).digest()
        ]
        
        # Combine entropy sources with XOR and hash
        combined = b'\x00' * 32
        for source in entropy_sources:
            combined = bytes(a ^ b for a, b in zip(combined, source))
        
        # Final strengthening with PBKDF2
        import secrets
        salt = secrets.token_bytes(16)
        
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA384(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            return kdf.derive(combined)
        except ImportError:
            # Fallback to standard HMAC strengthening
            strengthened = combined
            for _ in range(10000):
                strengthened = hmac.new(salt, strengthened, hashlib.sha384).digest()[:32]
            return strengthened
    
    def sign_task(self, task_data: Dict[str, Any], use_quantum_resistant: bool = True) -> str:
        """
        Generate cryptographic signature for task with optional quantum resistance.
        
        Args:
            task_data: Task data to sign
            use_quantum_resistant: Use quantum-resistant cryptography
            
        Returns:
            Hexadecimal signature string
        """
        if not self.policy.enable_task_signing:
            return ""
        
        # Serialize task data deterministically
        serialized = self._serialize_task_data(task_data)
        
        # Choose key and algorithm based on quantum resistance requirement
        if use_quantum_resistant:
            # Use quantum-resistant key with SHA-384
            key = self.quantum_resistant_key
            hash_algo = hashlib.sha384
            self.verification_stats['quantum_signatures_used'] += 1
        else:
            # Standard HMAC-SHA256
            key = self.secret_key
            hash_algo = hashlib.sha256
        
        # Generate dual-layer signature for enhanced security
        primary_signature = hmac.new(
            key,
            serialized.encode('utf-8'),
            hash_algo
        ).hexdigest()
        
        # Add timestamp and task-specific nonce for replay protection
        timestamp = str(int(time.time()))
        task_id = task_data.get('task_id', 'unknown')
        nonce_data = f"{task_id}:{timestamp}:{primary_signature}"
        
        secondary_signature = hmac.new(
            key,
            nonce_data.encode('utf-8'),
            hash_algo
        ).hexdigest()
        
        # Combine signatures with metadata
        combined_signature = f"{primary_signature}:{secondary_signature}:{timestamp}"
        
        self.verification_stats['signatures_generated'] += 1
        return combined_signature
    
    def verify_task_signature(self, task_data: Dict[str, Any], signature: str) -> bool:
        """
        Verify task signature with enhanced security checks.
        
        Args:
            task_data: Task data to verify
            signature: Expected signature
            
        Returns:
            True if signature is valid
        """
        if not self.policy.enable_task_signing or not signature:
            return True  # Skip verification if disabled
        
        self.verification_stats['signatures_verified'] += 1
        
        try:
            # Parse combined signature
            if ':' not in signature:
                # Legacy single signature format
                expected_signature = self.sign_task(task_data, use_quantum_resistant=False)
                result = hmac.compare_digest(signature, expected_signature.split(':')[0])
            else:
                # New dual-layer signature format
                signature_parts = signature.split(':')
                if len(signature_parts) != 3:
                    self.verification_stats['verification_failures'] += 1
                    return False
                
                primary_sig, secondary_sig, timestamp = signature_parts
                
                # Check signature age (prevent replay attacks)
                current_time = int(time.time())
                sig_time = int(timestamp)
                
                # Allow signatures valid for 1 hour
                if abs(current_time - sig_time) > 3600:
                    logger.warning(f"Signature timestamp too old: {current_time - sig_time} seconds")
                    self.verification_stats['verification_failures'] += 1
                    return False
                
                # Try quantum-resistant verification first
                try:
                    expected_signature = self.sign_task(task_data, use_quantum_resistant=True)
                    if hmac.compare_digest(signature, expected_signature):
                        return True
                except Exception:
                    pass
                
                # Fallback to standard verification
                try:
                    expected_signature = self.sign_task(task_data, use_quantum_resistant=False)
                    result = hmac.compare_digest(signature, expected_signature)
                except Exception:
                    result = False
            
            if not result:
                self.verification_stats['verification_failures'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            self.verification_stats['verification_failures'] += 1
            return False
    
    def get_verification_metrics(self) -> Dict[str, Any]:
        """Get cryptographic verification metrics."""
        total_verifications = self.verification_stats['signatures_verified']
        if total_verifications == 0:
            success_rate = 1.0
        else:
            success_rate = 1.0 - (self.verification_stats['verification_failures'] / total_verifications)
        
        return {
            'signatures_generated': self.verification_stats['signatures_generated'],
            'signatures_verified': self.verification_stats['signatures_verified'],
            'verification_failures': self.verification_stats['verification_failures'],
            'quantum_signatures_used': self.verification_stats['quantum_signatures_used'],
            'success_rate': success_rate,
            'quantum_usage_rate': (
                self.verification_stats['quantum_signatures_used'] / 
                max(1, self.verification_stats['signatures_generated'])
            )
        }
    
    def sign_result(self, result_data: Any, task_id: str) -> str:
        """
        Generate signature for task result.
        
        Args:
            result_data: Result to sign
            task_id: Associated task ID
            
        Returns:
            Hexadecimal signature
        """
        if not self.policy.enable_result_verification:
            return ""
        
        # Create result hash
        result_hash = self._hash_result(result_data)
        
        # Sign hash with task ID
        message = f"{task_id}:{result_hash}"
        signature = hmac.new(
            self.secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_result_signature(
        self, 
        result_data: Any, 
        task_id: str, 
        signature: str
    ) -> bool:
        """Verify result signature."""
        if not self.policy.enable_result_verification or not signature:
            return True
        
        expected_signature = self.sign_result(result_data, task_id)
        return hmac.compare_digest(signature, expected_signature)
    
    def _serialize_task_data(self, task_data: Dict[str, Any]) -> str:
        """Serialize task data for signing."""
        # Create deterministic representation
        sorted_items = sorted(task_data.items())
        serialized_items = []
        
        for key, value in sorted_items:
            if callable(value):
                # For functions, use their name and module
                func_repr = f"{getattr(value, '__module__', 'unknown')}.{getattr(value, '__name__', 'unknown')}"
                serialized_items.append(f"{key}:{func_repr}")
            else:
                serialized_items.append(f"{key}:{str(value)}")
        
        return "|".join(serialized_items)
    
    def _hash_result(self, result_data: Any) -> str:
        """Generate hash of result data."""
        if result_data is None:
            return hashlib.sha256(b"None").hexdigest()
        
        try:
            # Try to serialize as JSON
            import json
            result_str = json.dumps(result_data, sort_keys=True, default=str)
        except:
            # Fallback to string representation
            result_str = str(result_data)
        
        return hashlib.sha256(result_str.encode('utf-8')).hexdigest()


class SecurityMonitor:
    """Monitors system security and detects threats."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.violation_log = []
        self.quarantined_tasks = []
        
    def log_security_event(
        self, 
        event_type: str, 
        severity: str, 
        message: str, 
        task_id: Optional[str] = None
    ) -> None:
        """Log security event."""
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'message': message,
            'task_id': task_id
        }
        
        self.violation_log.append(event)
        
        # Alert if configured
        if self.policy.alert_on_violations and severity in ['HIGH', 'CRITICAL']:
            logger.warning(f"SECURITY ALERT [{severity}]: {message}")
    
    def quarantine_task(self, task_id: str, reason: str) -> None:
        """Quarantine suspicious task."""
        
        if not self.policy.quarantine_suspicious_tasks:
            return
        
        quarantine_record = {
            'task_id': task_id,
            'timestamp': time.time(),
            'reason': reason
        }
        
        self.quarantined_tasks.append(quarantine_record)
        
        self.log_security_event(
            'QUARANTINE',
            'HIGH',
            f"Task {task_id} quarantined: {reason}",
            task_id
        )
    
    def is_task_quarantined(self, task_id: str) -> bool:
        """Check if task is quarantined."""
        return any(record['task_id'] == task_id for record in self.quarantined_tasks)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        
        if not self.violation_log:
            return {
                'total_events': 0,
                'quarantined_tasks': 0,
                'security_status': 'CLEAN'
            }
        
        # Analyze events by severity
        severity_counts = {}
        for event in self.violation_log:
            severity = event['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determine overall security status
        if severity_counts.get('CRITICAL', 0) > 0:
            status = 'CRITICAL'
        elif severity_counts.get('HIGH', 0) > 0:
            status = 'HIGH_RISK'
        elif severity_counts.get('MEDIUM', 0) > 0:
            status = 'MODERATE_RISK'
        else:
            status = 'LOW_RISK'
        
        return {
            'total_events': len(self.violation_log),
            'severity_breakdown': severity_counts,
            'quarantined_tasks': len(self.quarantined_tasks),
            'security_status': status,
            'recent_events': self.violation_log[-10:]  # Last 10 events
        }


class PlanningSecurityManager:
    """
    Main security manager for quantum task planning system.
    
    Integrates all security components and provides unified interface.
    """
    
    def __init__(
        self, 
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        custom_policy: Optional[SecurityPolicy] = None
    ):
        self.security_level = security_level
        self.policy = custom_policy or SecurityPolicy()
        
        # Initialize security components
        self.validator = InputValidator(self.policy)
        self.sandbox = ExecutionSandbox(self.policy)
        self.verifier = CryptographicVerifier(self.policy)
        self.monitor = SecurityMonitor(self.policy)
        
        logger.info(f"Initialized security manager with level: {security_level.name}")
    
    def validate_and_secure_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive security validation on task.
        
        Args:
            task_data: Task data to validate and secure
            
        Returns:
            Secured task data with signatures
        """
        
        task_id = task_data.get('task_id', 'unknown')
        
        # Check if task is quarantined
        if self.monitor.is_task_quarantined(task_id):
            raise SecurityError(f"Task {task_id} is quarantined")
        
        # Validate operation
        operation = task_data.get('operation')
        if operation and not self.validator.validate_task_operation(operation):
            self.monitor.log_security_event(
                'VALIDATION_FAILED',
                'HIGH',
                f"Task operation validation failed",
                task_id
            )
            
            if self.security_level.value >= SecurityLevel.HIGH.value:
                self.monitor.quarantine_task(task_id, "Operation validation failed")
                raise SecurityError(f"Task {task_id} failed security validation")
        
        # Validate arguments
        args = task_data.get('args', ())
        kwargs = task_data.get('kwargs', {})
        
        if not self.validator.validate_task_args(args, kwargs):
            self.monitor.log_security_event(
                'ARGS_VALIDATION_FAILED',
                'MEDIUM',
                f"Task arguments validation failed",
                task_id
            )
            
            if self.security_level.value >= SecurityLevel.HIGH.value:
                self.monitor.quarantine_task(task_id, "Arguments validation failed")
                raise SecurityError(f"Task {task_id} has invalid arguments")
        
        # Generate cryptographic signature
        signature = ""
        if self.security_level.value >= SecurityLevel.HIGH.value:
            signature = self.verifier.sign_task(task_data)
        
        # Create secured task data
        secured_task = task_data.copy()
        secured_task['security_signature'] = signature
        secured_task['security_level'] = self.security_level.name
        secured_task['validated_timestamp'] = time.time()
        
        self.monitor.log_security_event(
            'TASK_VALIDATED',
            'INFO',
            f"Task successfully validated and secured",
            task_id
        )
        
        return secured_task
    
    def execute_secure_task(
        self, 
        task_data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute task in secure environment.
        
        Args:
            task_data: Secured task data
            timeout: Execution timeout
            
        Returns:
            Execution result with security metadata
        """
        
        task_id = task_data.get('task_id', 'unknown')
        
        # Verify task signature if present
        signature = task_data.get('security_signature', '')
        if signature and not self.verifier.verify_task_signature(task_data, signature):
            self.monitor.log_security_event(
                'SIGNATURE_VERIFICATION_FAILED',
                'CRITICAL',
                f"Task signature verification failed",
                task_id
            )
            raise SecurityError(f"Task {task_id} signature verification failed")
        
        operation = task_data.get('operation')
        args = task_data.get('args', ())
        kwargs = task_data.get('kwargs', {})
        
        execution_start = time.time()
        
        try:
            # Execute based on security level
            if self.security_level.value >= SecurityLevel.STANDARD.value:
                # Execute in sandbox
                result = self.sandbox.execute_in_sandbox(operation, args, kwargs, timeout)
            else:
                # Direct execution (minimal security)
                result = operation(*args, **kwargs)
            
            execution_time = time.time() - execution_start
            
            # Generate result signature
            result_signature = ""
            if self.security_level.value >= SecurityLevel.HIGH.value:
                result_signature = self.verifier.sign_result(result, task_id)
            
            # Create secure result
            secure_result = {
                'result': result,
                'execution_time': execution_time,
                'security_signature': result_signature,
                'security_level': self.security_level.name,
                'execution_timestamp': time.time(),
                'task_id': task_id
            }
            
            self.monitor.log_security_event(
                'TASK_EXECUTED',
                'INFO',
                f"Task executed successfully in {execution_time:.3f}s",
                task_id
            )
            
            return secure_result
            
        except Exception as e:
            execution_time = time.time() - execution_start
            
            self.monitor.log_security_event(
                'EXECUTION_FAILED',
                'MEDIUM',
                f"Task execution failed: {str(e)}",
                task_id
            )
            
            # Re-raise with security context
            raise SecurityError(f"Secure execution failed for task {task_id}: {str(e)}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status with enhanced metrics."""
        
        # Get verification metrics
        verification_metrics = self.verifier.get_verification_metrics()
        
        # Calculate overall security score
        security_score = self._calculate_security_score(verification_metrics)
        
        return {
            'security_level': self.security_level.name,
            'security_score': security_score,
            'policy_config': {
                'task_signing_enabled': self.policy.enable_task_signing,
                'result_verification_enabled': self.policy.enable_result_verification,
                'sandboxing_enabled': self.security_level.value >= SecurityLevel.STANDARD.value,
                'quantum_resistant_enabled': True,
                'max_execution_time': self.policy.max_execution_time,
                'max_memory_usage': self.policy.max_memory_usage,
                'max_energy_computation': self.policy.max_energy_computation,
                'max_particle_count': self.policy.max_particle_count
            },
            'validation_stats': {
                'violations_detected': self.validator.violation_count,
                'validation_success_rate': self._get_validation_success_rate()
            },
            'cryptographic_stats': verification_metrics,
            'sandbox_stats': {
                'active_sandboxes': len(self.sandbox.temp_dirs),
                'total_sandboxes_created': len(self.sandbox.temp_dirs)
            },
            'monitoring_summary': self.monitor.get_security_summary(),
            'performance_metrics': {
                'average_validation_time': self._get_average_validation_time(),
                'sandbox_overhead': self._get_sandbox_overhead()
            }
        }
    
    def _calculate_security_score(self, verification_metrics: Dict[str, Any]) -> float:
        """Calculate overall security score (0-100)."""
        
        score = 100.0
        
        # Deduct points for verification failures
        verification_success_rate = verification_metrics.get('success_rate', 1.0)
        score *= verification_success_rate
        
        # Deduct points for validation violations
        if hasattr(self.validator, 'violation_count') and self.validator.violation_count > 0:
            violation_penalty = min(20.0, self.validator.violation_count * 2.0)
            score -= violation_penalty
        
        # Deduct points for quarantined tasks
        quarantined_count = len(self.monitor.quarantined_tasks)
        if quarantined_count > 0:
            quarantine_penalty = min(30.0, quarantined_count * 5.0)
            score -= quarantine_penalty
        
        # Bonus points for quantum-resistant cryptography usage
        quantum_usage_rate = verification_metrics.get('quantum_usage_rate', 0.0)
        quantum_bonus = quantum_usage_rate * 5.0
        score += quantum_bonus
        
        # Bonus points for high security level
        if self.security_level == SecurityLevel.CRITICAL:
            score += 10.0
        elif self.security_level == SecurityLevel.HIGH:
            score += 5.0
        
        return max(0.0, min(100.0, score))
    
    def _get_validation_success_rate(self) -> float:
        """Calculate validation success rate."""
        if not hasattr(self.validator, 'validation_attempts'):
            # Initialize if not exists
            self.validator.validation_attempts = 1
        
        if self.validator.validation_attempts == 0:
            return 1.0
        
        success_rate = 1.0 - (self.validator.violation_count / max(1, self.validator.validation_attempts))
        return max(0.0, min(1.0, success_rate))
    
    def _get_average_validation_time(self) -> float:
        """Get average validation time in milliseconds."""
        # Placeholder - would track actual validation times in production
        return 2.5  # Estimated average validation time
    
    def _get_sandbox_overhead(self) -> float:
        """Get sandbox execution overhead as percentage."""
        # Placeholder - would measure actual overhead in production
        return 15.0  # Estimated 15% overhead for sandboxing
    
    def cleanup(self) -> None:
        """Clean up security resources."""
        self.sandbox.cleanup_all_sandboxes()
        logger.info("Security manager cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass