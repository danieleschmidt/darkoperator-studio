"""Basic validation utilities without external dependencies."""

import os.path
from typing import Union


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def sanitize_file_path(path: str) -> str:
    """
    Sanitize file paths to prevent directory traversal attacks.
    
    Args:
        path: Input file path
        
    Returns:
        Sanitized path
        
    Raises:
        ValidationError: If path is unsafe
    """
    # Remove null bytes
    path = path.replace('\x00', '')
    
    # Check for directory traversal attempts
    if '..' in path or path.startswith('/'):
        raise ValidationError(f"Unsafe path detected: {path}")
    
    # Normalize path
    path = os.path.normpath(path)
    
    # Additional checks
    dangerous_patterns = ['~', '$', '`', '|', ';', '&', '>', '<']
    for pattern in dangerous_patterns:
        if pattern in path:
            raise ValidationError(f"Dangerous character '{pattern}' in path: {path}")
    
    return path


def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float, name: str = "value") -> None:
    """
    Validate that a numeric value is within expected range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_string_length(value: str, max_length: int, name: str = "string") -> None:
    """
    Validate string length.
    
    Args:
        value: String to validate
        max_length: Maximum allowed length
        name: Name for error messages
        
    Raises:
        ValidationError: If string is too long
    """
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value)}")
    
    if len(value) > max_length:
        raise ValidationError(f"{name} length {len(value)} exceeds maximum {max_length}")


def validate_required_keys(data: dict, required_keys: list, name: str = "data") -> None:
    """
    Validate that dictionary contains required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        name: Name for error messages
        
    Raises:
        ValidationError: If required keys are missing
    """
    if not isinstance(data, dict):
        raise ValidationError(f"{name} must be a dictionary, got {type(data)}")
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(f"{name} missing required keys: {missing_keys}")