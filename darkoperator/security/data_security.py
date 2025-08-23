"""
Data security utilities for DarkOperator Studio.
"""

import re
import logging
from typing import Any, Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSecurityError(Exception):
    """Raised when data validation fails."""
    pass

def sanitize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize user inputs to prevent security vulnerabilities.
    
    Args:
        inputs: Dictionary of user inputs
        
    Returns:
        Sanitized inputs
        
    Raises:
        DataSecurityError: If inputs contain malicious content
    """
    sanitized = {}
    
    for key, value in inputs.items():
        # Sanitize key names
        if not re.match(r'^[a-zA-Z0-9_]+$', key):
            raise DataSecurityError(f"Invalid key name: {key}")
            
        # Sanitize string values
        if isinstance(value, str):
            # Check for SQL injection patterns
            dangerous_patterns = [
                r'(union|select|insert|update|delete|drop|create|alter)\s+',
                r'--',
                r'/\*',
                r'\*/',
                r'<script',
                r'javascript:'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, value.lower()):
                    logger.warning(f"Potentially dangerous pattern detected in input: {key}")
                    raise DataSecurityError(f"Malicious pattern detected in input: {key}")
                    
            # Limit string length
            if len(value) > 10000:
                raise DataSecurityError(f"Input too long for key: {key}")
                
        sanitized[key] = value
        
    return sanitized

def validate_data_source(source_path: Union[str, Path], allowed_extensions: List[str] = None) -> bool:
    """
    Validate that a data source is safe to use.
    
    Args:
        source_path: Path to data source
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if source is valid
        
    Raises:
        DataSecurityError: If source is invalid or unsafe
    """
    path = Path(source_path)
    
    # Check if file exists
    if not path.exists():
        raise DataSecurityError(f"Data source does not exist: {source_path}")
    
    # Check file extension
    if allowed_extensions is None:
        allowed_extensions = ['.h5', '.hdf5', '.csv', '.json', '.yaml', '.yml', '.pkl', '.pickle']
        
    if path.suffix.lower() not in allowed_extensions:
        raise DataSecurityError(f"File extension not allowed: {path.suffix}")
    
    # Check for path traversal attempts
    resolved_path = path.resolve()
    cwd = Path.cwd().resolve()
    
    try:
        resolved_path.relative_to(cwd)
    except ValueError:
        raise DataSecurityError(f"Path traversal attempt detected: {source_path}")
    
    # Check file size (prevent loading extremely large files)
    file_size = path.stat().st_size
    max_size = 10 * 1024 * 1024 * 1024  # 10 GB
    
    if file_size > max_size:
        raise DataSecurityError(f"File too large: {file_size} bytes")
    
    logger.info(f"Data source validated: {source_path}")
    return True