"""Configuration management for DarkOperator Studio."""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging


logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_file_size_gb: float = 2.0
    trusted_domains: List[str] = field(default_factory=lambda: [
        'huggingface.co', 'github.com', 'opendata.cern.ch', 'zenodo.org'
    ])
    enable_model_validation: bool = True
    require_checksums: bool = True
    max_memory_usage_gb: float = 16.0


@dataclass  
class ModelConfig:
    """Model configuration settings."""
    default_device: str = "auto"
    max_batch_size: int = 32
    inference_timeout_s: float = 300.0
    enable_mixed_precision: bool = True
    cache_dir: str = "./models"


@dataclass
class DataConfig:
    """Data processing configuration."""
    cache_dir: str = "./data"
    max_events_per_file: int = 1_000_000
    validation_split: float = 0.2
    random_seed: int = 42
    preprocessing_workers: int = 4


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    enable_performance_monitoring: bool = True
    metrics_retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_usage_gb': 12.0,
        'inference_time_ms': 1000.0,
        'error_rate': 0.05
    })


@dataclass
class Settings:
    """Main configuration settings container."""
    security: SecurityConfig = field(default_factory=SecurityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    debug: bool = False
    verbose: bool = False
    config_version: str = "1.0"


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to config file (YAML or JSON)
        
    Returns:
        Loaded settings
    """
    # Start with defaults
    settings = Settings()
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        settings = _load_from_file(config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    
    # Override with environment variables
    settings = _load_from_environment(settings)
    
    # Validate configuration
    validate_config(settings)
    
    return settings


def _load_from_file(config_path: str) -> Settings:
    """Load configuration from YAML or JSON file."""
    config_file = Path(config_path)
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
        
        # Convert nested dict to Settings object
        return _dict_to_settings(config_data)
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def _dict_to_settings(config_dict: Dict[str, Any]) -> Settings:
    """Convert configuration dictionary to Settings object."""
    settings = Settings()
    
    # Map configuration sections
    if 'security' in config_dict:
        settings.security = SecurityConfig(**config_dict['security'])
    
    if 'model' in config_dict:
        settings.model = ModelConfig(**config_dict['model'])
    
    if 'data' in config_dict:
        settings.data = DataConfig(**config_dict['data'])
    
    if 'monitoring' in config_dict:
        settings.monitoring = MonitoringConfig(**config_dict['monitoring'])
    
    # Global settings
    for key in ['debug', 'verbose', 'config_version']:
        if key in config_dict:
            setattr(settings, key, config_dict[key])
    
    return settings


def _load_from_environment(settings: Settings) -> Settings:
    """Override settings with environment variables."""
    env_mappings = {
        'DARKOPERATOR_DEBUG': ('debug', bool),
        'DARKOPERATOR_VERBOSE': ('verbose', bool),
        'DARKOPERATOR_LOG_LEVEL': ('monitoring.log_level', str),
        'DARKOPERATOR_LOG_DIR': ('monitoring.log_dir', str),
        'DARKOPERATOR_CACHE_DIR': ('data.cache_dir', str),
        'DARKOPERATOR_MODEL_CACHE': ('model.cache_dir', str),
        'DARKOPERATOR_MAX_BATCH_SIZE': ('model.max_batch_size', int),
        'DARKOPERATOR_DEVICE': ('model.default_device', str),
        'DARKOPERATOR_MAX_MEMORY_GB': ('security.max_memory_usage_gb', float),
    }
    
    for env_var, (setting_path, setting_type) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                # Convert to appropriate type
                if setting_type == bool:
                    value = env_value.lower() in ['true', '1', 'yes', 'on']
                else:
                    value = setting_type(env_value)
                
                # Set nested attribute
                _set_nested_attr(settings, setting_path, value)
                logger.debug(f"Set {setting_path} = {value} from {env_var}")
                
            except Exception as e:
                logger.warning(f"Failed to parse {env_var}={env_value}: {e}")
    
    return settings


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set nested attribute using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def validate_config(settings: Settings) -> None:
    """
    Validate configuration settings.
    
    Args:
        settings: Settings to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate paths exist or can be created
    for path_attr in [
        settings.data.cache_dir,
        settings.model.cache_dir,
        settings.monitoring.log_dir
    ]:
        try:
            Path(path_attr).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create directory {path_attr}: {e}")
    
    # Validate numeric ranges
    if settings.security.max_file_size_gb <= 0:
        raise ValueError("max_file_size_gb must be positive")
    
    if settings.model.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    
    if not 0 < settings.data.validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    
    # Validate log level
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if settings.monitoring.log_level.upper() not in valid_log_levels:
        raise ValueError(f"log_level must be one of: {valid_log_levels}")
    
    # Validate device setting
    if settings.model.default_device not in ['auto', 'cpu', 'cuda']:
        if not settings.model.default_device.startswith('cuda:'):
            raise ValueError("default_device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    
    logger.info("Configuration validation passed")


def save_config(settings: Settings, config_path: str) -> None:
    """Save configuration to file."""
    config_file = Path(config_path)
    
    # Convert to dictionary
    config_dict = {
        'config_version': settings.config_version,
        'debug': settings.debug,
        'verbose': settings.verbose,
        'security': {
            'max_file_size_gb': settings.security.max_file_size_gb,
            'trusted_domains': settings.security.trusted_domains,
            'enable_model_validation': settings.security.enable_model_validation,
            'require_checksums': settings.security.require_checksums,
            'max_memory_usage_gb': settings.security.max_memory_usage_gb,
        },
        'model': {
            'default_device': settings.model.default_device,
            'max_batch_size': settings.model.max_batch_size,
            'inference_timeout_s': settings.model.inference_timeout_s,
            'enable_mixed_precision': settings.model.enable_mixed_precision,
            'cache_dir': settings.model.cache_dir,
        },
        'data': {
            'cache_dir': settings.data.cache_dir,
            'max_events_per_file': settings.data.max_events_per_file,
            'validation_split': settings.data.validation_split,
            'random_seed': settings.data.random_seed,
            'preprocessing_workers': settings.data.preprocessing_workers,
        },
        'monitoring': {
            'log_level': settings.monitoring.log_level,
            'log_dir': settings.monitoring.log_dir,
            'enable_performance_monitoring': settings.monitoring.enable_performance_monitoring,
            'metrics_retention_days': settings.monitoring.metrics_retention_days,
            'alert_thresholds': settings.monitoring.alert_thresholds,
        }
    }
    
    # Save as YAML or JSON based on extension
    with open(config_file, 'w') as f:
        if config_file.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")


# Default configuration file locations
DEFAULT_CONFIG_PATHS = [
    "./darkoperator.yml",
    "./config/darkoperator.yml", 
    os.path.expanduser("~/.darkoperator/config.yml"),
    "/etc/darkoperator/config.yml"
]


def find_config_file() -> Optional[str]:
    """Find configuration file in default locations."""
    for path in DEFAULT_CONFIG_PATHS:
        if Path(path).exists():
            return path
    return None