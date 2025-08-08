"""
Global deployment configuration for quantum task planning system.

Supports multi-region deployment with compliance frameworks including
GDPR, CCPA, PDPA and cross-platform compatibility.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    
    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    CA_CENTRAL_1 = "ca-central-1"
    
    # Europe
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"
    
    # Asia Pacific
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_SOUTH_1 = "ap-south-1"
    
    # Others
    AU_SOUTHEAST_2 = "au-southeast-2"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    PIPEDA = "pipeda"      # Personal Information Protection (Canada)
    SOX = "sox"            # Sarbanes-Oxley Act
    HIPAA = "hipaa"        # Health Insurance Portability (US)
    ISO27001 = "iso27001"  # Information Security Standard


class Platform(Enum):
    """Supported deployment platforms."""
    
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "k8s"
    DOCKER = "docker"
    BARE_METAL = "bare_metal"


@dataclass
class ComplianceConfig:
    """Configuration for compliance requirements."""
    
    frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Data residency requirements
    data_residency_regions: List[Region] = field(default_factory=list)
    cross_border_transfer_allowed: bool = False
    
    # Privacy settings
    data_retention_days: int = 365
    anonymization_required: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    
    # Audit requirements
    audit_logging_enabled: bool = True
    audit_log_retention_years: int = 7
    
    # Access controls
    multi_factor_authentication: bool = True
    role_based_access_control: bool = True
    data_classification_levels: List[str] = field(default_factory=lambda: [
        "public", "internal", "confidential", "restricted"
    ])
    
    # Physics data specific
    research_data_sharing_allowed: bool = True
    scientific_collaboration_regions: List[Region] = field(default_factory=list)
    cern_opendata_compliance: bool = True


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    
    region: Region
    primary: bool = False
    
    # Infrastructure
    platform: Platform = Platform.AWS
    availability_zones: List[str] = field(default_factory=list)
    instance_types: Dict[str, str] = field(default_factory=dict)
    
    # Networking
    vpc_cidr: str = "10.0.0.0/16"
    public_subnets: List[str] = field(default_factory=list)
    private_subnets: List[str] = field(default_factory=list)
    
    # Storage
    storage_class: str = "standard"
    backup_enabled: bool = True
    backup_regions: List[Region] = field(default_factory=list)
    
    # Compute
    auto_scaling_enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    
    # Physics computing specific
    gpu_enabled: bool = True
    gpu_instance_type: str = "p3.2xlarge"
    high_memory_instances: bool = True
    quantum_simulation_optimized: bool = True
    
    # Compliance
    compliance_config: Optional[ComplianceConfig] = None


@dataclass
class LocalizationConfig:
    """Localization and internationalization configuration."""
    
    # Supported languages (ISO 639-1)
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "ja", "zh", "ko", "it", "pt", "ru"
    ])
    
    default_language: str = "en"
    
    # Regional formats
    date_formats: Dict[str, str] = field(default_factory=lambda: {
        "en": "%Y-%m-%d %H:%M:%S UTC",
        "es": "%d/%m/%Y %H:%M:%S UTC",
        "fr": "%d/%m/%Y %H:%M:%S UTC", 
        "de": "%d.%m.%Y %H:%M:%S UTC",
        "ja": "%Y年%m月%d日 %H:%M:%S UTC",
        "zh": "%Y年%m月%d日 %H:%M:%S UTC"
    })
    
    number_formats: Dict[str, str] = field(default_factory=lambda: {
        "en": "1,234.56",
        "es": "1.234,56",
        "fr": "1 234,56",
        "de": "1.234,56",
        "ja": "1,234.56",
        "zh": "1,234.56"
    })
    
    # Physics units localization
    unit_systems: Dict[str, str] = field(default_factory=lambda: {
        "en": "SI",    # International System of Units
        "es": "SI",
        "fr": "SI", 
        "de": "SI",
        "ja": "SI",
        "zh": "SI"
    })
    
    # Scientific notation preferences
    scientific_notation: Dict[str, bool] = field(default_factory=lambda: {
        "en": True,
        "es": True,
        "fr": True,
        "de": True,
        "ja": True,
        "zh": True
    })


@dataclass
class SecurityConfig:
    """Global security configuration."""
    
    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    
    # Network security
    tls_version: str = "1.3"
    certificate_authority: str = "Let's Encrypt"
    
    # Authentication
    jwt_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    
    # API security
    rate_limiting_enabled: bool = True
    rate_limit_per_minute: int = 1000
    api_key_required: bool = True
    
    # Data security
    pii_detection_enabled: bool = True
    data_loss_prevention: bool = True
    
    # Physics-specific security
    physics_data_classification: bool = True
    research_data_watermarking: bool = True
    collaborative_research_security: bool = True


class GlobalConfiguration:
    """
    Global configuration manager for multi-region deployment.
    
    Handles region-specific settings, compliance requirements,
    and cross-region coordination.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "global_config.json"
        
        # Default configurations
        self.regions: Dict[Region, RegionConfig] = {}
        self.compliance = ComplianceConfig()
        self.localization = LocalizationConfig()
        self.security = SecurityConfig()
        
        # Runtime state
        self.active_regions: List[Region] = []
        self.primary_region: Optional[Region] = None
        
        # Load configuration if exists
        self.load_configuration()
        
        logger.info(f"Initialized global configuration with {len(self.regions)} regions")
    
    def add_region(self, region_config: RegionConfig) -> None:
        """Add region configuration."""
        
        self.regions[region_config.region] = region_config
        
        if region_config.primary:
            if self.primary_region is not None:
                logger.warning(f"Replacing primary region {self.primary_region} with {region_config.region}")
            self.primary_region = region_config.region
        
        logger.info(f"Added region configuration: {region_config.region.value}")
    
    def set_compliance_framework(self, framework: ComplianceFramework) -> None:
        """Add compliance framework requirement."""
        
        if framework not in self.compliance.frameworks:
            self.compliance.frameworks.append(framework)
            
            # Apply framework-specific defaults
            self._apply_compliance_defaults(framework)
            
            logger.info(f"Added compliance framework: {framework.value}")
    
    def _apply_compliance_defaults(self, framework: ComplianceFramework) -> None:
        """Apply default settings for compliance framework."""
        
        if framework == ComplianceFramework.GDPR:
            # GDPR requirements
            self.compliance.data_retention_days = min(self.compliance.data_retention_days, 365)
            self.compliance.anonymization_required = True
            self.compliance.cross_border_transfer_allowed = False
            self.compliance.audit_logging_enabled = True
            
            # Limit data residency to EU regions
            eu_regions = [Region.EU_WEST_1, Region.EU_CENTRAL_1, Region.EU_NORTH_1]
            if not self.compliance.data_residency_regions:
                self.compliance.data_residency_regions = eu_regions
        
        elif framework == ComplianceFramework.CCPA:
            # CCPA requirements
            self.compliance.data_retention_days = min(self.compliance.data_retention_days, 730)
            self.compliance.audit_logging_enabled = True
            
        elif framework == ComplianceFramework.PDPA:
            # PDPA requirements (Singapore)
            self.compliance.data_residency_regions = [Region.AP_SOUTHEAST_1]
            self.compliance.cross_border_transfer_allowed = False
            
        elif framework == ComplianceFramework.ISO27001:
            # ISO 27001 security requirements
            self.security.encryption_algorithm = "AES-256-GCM"
            self.security.key_rotation_days = 90
            self.security.tls_version = "1.3"
            self.compliance.multi_factor_authentication = True
    
    def get_compliant_regions(self) -> List[Region]:
        """Get list of regions that meet compliance requirements."""
        
        if not self.compliance.data_residency_regions:
            # No specific residency requirements
            return list(self.regions.keys())
        
        compliant_regions = []
        
        for region in self.regions.keys():
            if region in self.compliance.data_residency_regions:
                compliant_regions.append(region)
        
        return compliant_regions
    
    def get_region_config(self, region: Region) -> Optional[RegionConfig]:
        """Get configuration for specific region."""
        return self.regions.get(region)
    
    def is_cross_region_transfer_allowed(
        self, 
        source_region: Region, 
        target_region: Region
    ) -> bool:
        """Check if data transfer between regions is allowed."""
        
        if not self.compliance.cross_border_transfer_allowed:
            # Check if both regions are in same compliance zone
            compliant_regions = self.get_compliant_regions()
            return (source_region in compliant_regions and 
                   target_region in compliant_regions)
        
        return True
    
    def get_localized_format(self, format_type: str, language: str = "en") -> str:
        """Get localized format string."""
        
        if format_type == "date":
            return self.localization.date_formats.get(language, 
                   self.localization.date_formats["en"])
        elif format_type == "number":
            return self.localization.number_formats.get(language,
                   self.localization.number_formats["en"])
        elif format_type == "unit_system":
            return self.localization.unit_systems.get(language, "SI")
        else:
            return ""
    
    def validate_configuration(self) -> List[str]:
        """Validate global configuration and return any issues."""
        
        issues = []
        
        # Check for primary region
        if not self.primary_region:
            issues.append("No primary region specified")
        
        # Check compliance requirements
        for framework in self.compliance.frameworks:
            if framework == ComplianceFramework.GDPR:
                eu_regions = [r for r in self.regions.keys() 
                            if r.value.startswith('eu-')]
                if not eu_regions and not self.compliance.data_residency_regions:
                    issues.append("GDPR compliance requires EU regions")
        
        # Check region configurations
        for region, config in self.regions.items():
            if not config.availability_zones:
                issues.append(f"Region {region.value} has no availability zones")
            
            if config.gpu_enabled and not config.gpu_instance_type:
                issues.append(f"Region {region.value} has GPU enabled but no GPU instance type")
        
        # Check security settings
        if not self.security.tls_version:
            issues.append("TLS version not specified")
        
        # Check localization
        if not self.localization.supported_languages:
            issues.append("No supported languages specified")
        
        return issues
    
    def save_configuration(self, path: Optional[str] = None) -> None:
        """Save configuration to JSON file."""
        
        save_path = path or self.config_path
        
        config_dict = {
            "regions": {
                region.value: {
                    "region": region.value,
                    "primary": config.primary,
                    "platform": config.platform.value,
                    "availability_zones": config.availability_zones,
                    "instance_types": config.instance_types,
                    "vpc_cidr": config.vpc_cidr,
                    "public_subnets": config.public_subnets,
                    "private_subnets": config.private_subnets,
                    "storage_class": config.storage_class,
                    "backup_enabled": config.backup_enabled,
                    "backup_regions": [r.value for r in config.backup_regions],
                    "auto_scaling_enabled": config.auto_scaling_enabled,
                    "min_instances": config.min_instances,
                    "max_instances": config.max_instances,
                    "target_cpu_utilization": config.target_cpu_utilization,
                    "gpu_enabled": config.gpu_enabled,
                    "gpu_instance_type": config.gpu_instance_type,
                    "high_memory_instances": config.high_memory_instances,
                    "quantum_simulation_optimized": config.quantum_simulation_optimized
                }
                for region, config in self.regions.items()
            },
            "compliance": {
                "frameworks": [f.value for f in self.compliance.frameworks],
                "data_residency_regions": [r.value for r in self.compliance.data_residency_regions],
                "cross_border_transfer_allowed": self.compliance.cross_border_transfer_allowed,
                "data_retention_days": self.compliance.data_retention_days,
                "anonymization_required": self.compliance.anonymization_required,
                "encryption_at_rest": self.compliance.encryption_at_rest,
                "encryption_in_transit": self.compliance.encryption_in_transit,
                "audit_logging_enabled": self.compliance.audit_logging_enabled,
                "audit_log_retention_years": self.compliance.audit_log_retention_years,
                "multi_factor_authentication": self.compliance.multi_factor_authentication,
                "role_based_access_control": self.compliance.role_based_access_control,
                "data_classification_levels": self.compliance.data_classification_levels,
                "research_data_sharing_allowed": self.compliance.research_data_sharing_allowed,
                "scientific_collaboration_regions": [r.value for r in self.compliance.scientific_collaboration_regions],
                "cern_opendata_compliance": self.compliance.cern_opendata_compliance
            },
            "localization": {
                "supported_languages": self.localization.supported_languages,
                "default_language": self.localization.default_language,
                "date_formats": self.localization.date_formats,
                "number_formats": self.localization.number_formats,
                "unit_systems": self.localization.unit_systems,
                "scientific_notation": self.localization.scientific_notation
            },
            "security": {
                "encryption_algorithm": self.security.encryption_algorithm,
                "key_rotation_days": self.security.key_rotation_days,
                "tls_version": self.security.tls_version,
                "certificate_authority": self.security.certificate_authority,
                "jwt_expiry_hours": self.security.jwt_expiry_hours,
                "refresh_token_expiry_days": self.security.refresh_token_expiry_days,
                "rate_limiting_enabled": self.security.rate_limiting_enabled,
                "rate_limit_per_minute": self.security.rate_limit_per_minute,
                "api_key_required": self.security.api_key_required,
                "pii_detection_enabled": self.security.pii_detection_enabled,
                "data_loss_prevention": self.security.data_loss_prevention,
                "physics_data_classification": self.security.physics_data_classification,
                "research_data_watermarking": self.security.research_data_watermarking,
                "collaborative_research_security": self.security.collaborative_research_security
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {save_path}")
    
    def load_configuration(self, path: Optional[str] = None) -> bool:
        """Load configuration from JSON file."""
        
        load_path = path or self.config_path
        
        if not os.path.exists(load_path):
            logger.info(f"Configuration file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'r') as f:
                config_dict = json.load(f)
            
            # Load regions
            if 'regions' in config_dict:
                for region_name, region_data in config_dict['regions'].items():
                    region = Region(region_name)
                    
                    region_config = RegionConfig(
                        region=region,
                        primary=region_data.get('primary', False),
                        platform=Platform(region_data.get('platform', 'aws')),
                        availability_zones=region_data.get('availability_zones', []),
                        instance_types=region_data.get('instance_types', {}),
                        vpc_cidr=region_data.get('vpc_cidr', '10.0.0.0/16'),
                        public_subnets=region_data.get('public_subnets', []),
                        private_subnets=region_data.get('private_subnets', []),
                        storage_class=region_data.get('storage_class', 'standard'),
                        backup_enabled=region_data.get('backup_enabled', True),
                        backup_regions=[Region(r) for r in region_data.get('backup_regions', [])],
                        auto_scaling_enabled=region_data.get('auto_scaling_enabled', True),
                        min_instances=region_data.get('min_instances', 1),
                        max_instances=region_data.get('max_instances', 10),
                        target_cpu_utilization=region_data.get('target_cpu_utilization', 70.0),
                        gpu_enabled=region_data.get('gpu_enabled', True),
                        gpu_instance_type=region_data.get('gpu_instance_type', 'p3.2xlarge'),
                        high_memory_instances=region_data.get('high_memory_instances', True),
                        quantum_simulation_optimized=region_data.get('quantum_simulation_optimized', True)
                    )
                    
                    self.add_region(region_config)
            
            # Load compliance
            if 'compliance' in config_dict:
                comp_data = config_dict['compliance']
                self.compliance = ComplianceConfig(
                    frameworks=[ComplianceFramework(f) for f in comp_data.get('frameworks', [])],
                    data_residency_regions=[Region(r) for r in comp_data.get('data_residency_regions', [])],
                    cross_border_transfer_allowed=comp_data.get('cross_border_transfer_allowed', False),
                    data_retention_days=comp_data.get('data_retention_days', 365),
                    anonymization_required=comp_data.get('anonymization_required', True),
                    encryption_at_rest=comp_data.get('encryption_at_rest', True),
                    encryption_in_transit=comp_data.get('encryption_in_transit', True),
                    audit_logging_enabled=comp_data.get('audit_logging_enabled', True),
                    audit_log_retention_years=comp_data.get('audit_log_retention_years', 7),
                    multi_factor_authentication=comp_data.get('multi_factor_authentication', True),
                    role_based_access_control=comp_data.get('role_based_access_control', True),
                    data_classification_levels=comp_data.get('data_classification_levels', []),
                    research_data_sharing_allowed=comp_data.get('research_data_sharing_allowed', True),
                    scientific_collaboration_regions=[Region(r) for r in comp_data.get('scientific_collaboration_regions', [])],
                    cern_opendata_compliance=comp_data.get('cern_opendata_compliance', True)
                )
            
            # Load localization
            if 'localization' in config_dict:
                loc_data = config_dict['localization']
                self.localization = LocalizationConfig(
                    supported_languages=loc_data.get('supported_languages', ["en"]),
                    default_language=loc_data.get('default_language', "en"),
                    date_formats=loc_data.get('date_formats', {}),
                    number_formats=loc_data.get('number_formats', {}),
                    unit_systems=loc_data.get('unit_systems', {}),
                    scientific_notation=loc_data.get('scientific_notation', {})
                )
            
            # Load security
            if 'security' in config_dict:
                sec_data = config_dict['security']
                self.security = SecurityConfig(
                    encryption_algorithm=sec_data.get('encryption_algorithm', 'AES-256-GCM'),
                    key_rotation_days=sec_data.get('key_rotation_days', 90),
                    tls_version=sec_data.get('tls_version', '1.3'),
                    certificate_authority=sec_data.get('certificate_authority', "Let's Encrypt"),
                    jwt_expiry_hours=sec_data.get('jwt_expiry_hours', 24),
                    refresh_token_expiry_days=sec_data.get('refresh_token_expiry_days', 30),
                    rate_limiting_enabled=sec_data.get('rate_limiting_enabled', True),
                    rate_limit_per_minute=sec_data.get('rate_limit_per_minute', 1000),
                    api_key_required=sec_data.get('api_key_required', True),
                    pii_detection_enabled=sec_data.get('pii_detection_enabled', True),
                    data_loss_prevention=sec_data.get('data_loss_prevention', True),
                    physics_data_classification=sec_data.get('physics_data_classification', True),
                    research_data_watermarking=sec_data.get('research_data_watermarking', True),
                    collaborative_research_security=sec_data.get('collaborative_research_security', True)
                )
            
            logger.info(f"Loaded configuration from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def create_default_configuration(self) -> None:
        """Create default multi-region configuration."""
        
        # Add US East (primary)
        us_east_config = RegionConfig(
            region=Region.US_EAST_1,
            primary=True,
            platform=Platform.AWS,
            availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
            instance_types={
                "compute": "c5.2xlarge",
                "memory": "r5.2xlarge", 
                "gpu": "p3.2xlarge"
            },
            public_subnets=["10.0.1.0/24", "10.0.2.0/24"],
            private_subnets=["10.0.10.0/24", "10.0.20.0/24"],
            backup_regions=[Region.US_WEST_2],
            gpu_enabled=True,
            quantum_simulation_optimized=True
        )
        self.add_region(us_east_config)
        
        # Add EU West (GDPR compliance)
        eu_west_config = RegionConfig(
            region=Region.EU_WEST_1,
            platform=Platform.AWS,
            availability_zones=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
            instance_types={
                "compute": "c5.2xlarge",
                "memory": "r5.2xlarge",
                "gpu": "p3.2xlarge"
            },
            public_subnets=["10.1.1.0/24", "10.1.2.0/24"],
            private_subnets=["10.1.10.0/24", "10.1.20.0/24"],
            backup_regions=[Region.EU_CENTRAL_1],
            gpu_enabled=True,
            quantum_simulation_optimized=True
        )
        self.add_region(eu_west_config)
        
        # Add Asia Pacific 
        ap_southeast_config = RegionConfig(
            region=Region.AP_SOUTHEAST_1,
            platform=Platform.AWS,
            availability_zones=["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"],
            instance_types={
                "compute": "c5.2xlarge",
                "memory": "r5.2xlarge",
                "gpu": "p3.2xlarge"
            },
            public_subnets=["10.2.1.0/24", "10.2.2.0/24"],
            private_subnets=["10.2.10.0/24", "10.2.20.0/24"],
            backup_regions=[Region.AP_NORTHEAST_1],
            gpu_enabled=True,
            quantum_simulation_optimized=True
        )
        self.add_region(ap_southeast_config)
        
        # Set compliance frameworks
        self.set_compliance_framework(ComplianceFramework.GDPR)
        self.set_compliance_framework(ComplianceFramework.CCPA)
        
        # Configure research collaboration
        self.compliance.scientific_collaboration_regions = [
            Region.US_EAST_1, Region.EU_WEST_1, Region.AP_SOUTHEAST_1
        ]
        self.compliance.research_data_sharing_allowed = True
        self.compliance.cern_opendata_compliance = True
        
        logger.info("Created default multi-region configuration")
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of deployment configuration."""
        
        return {
            "total_regions": len(self.regions),
            "primary_region": self.primary_region.value if self.primary_region else None,
            "compliant_regions": [r.value for r in self.get_compliant_regions()],
            "compliance_frameworks": [f.value for f in self.compliance.frameworks],
            "supported_languages": self.localization.supported_languages,
            "security_features": {
                "encryption_at_rest": self.compliance.encryption_at_rest,
                "encryption_in_transit": self.compliance.encryption_in_transit,
                "multi_factor_auth": self.compliance.multi_factor_authentication,
                "audit_logging": self.compliance.audit_logging_enabled
            },
            "physics_features": {
                "gpu_enabled_regions": sum(1 for config in self.regions.values() if config.gpu_enabled),
                "quantum_optimized_regions": sum(1 for config in self.regions.values() if config.quantum_simulation_optimized),
                "cern_opendata_compliant": self.compliance.cern_opendata_compliance,
                "research_collaboration_enabled": self.compliance.research_data_sharing_allowed
            },
            "validation_issues": self.validate_configuration()
        }
    
    def setup_disaster_recovery(self) -> None:
        """Setup disaster recovery configuration."""
        
        for region_config in self.regions.values():
            if region_config.backup_regions:
                logger.info(f"Disaster recovery configured for {region_config.region.value}")
                
                # Ensure backup regions have compatible configuration
                for backup_region in region_config.backup_regions:
                    if backup_region in self.regions:
                        backup_config = self.regions[backup_region]
                        
                        # Ensure backup region has similar capabilities
                        if region_config.gpu_enabled and not backup_config.gpu_enabled:
                            logger.warning(f"Backup region {backup_region.value} lacks GPU support")
                        
                        if region_config.quantum_simulation_optimized and not backup_config.quantum_simulation_optimized:
                            logger.warning(f"Backup region {backup_region.value} not quantum optimized")
    
    def get_optimal_region_for_workload(self, requirements: Dict[str, Any]) -> Optional[Region]:
        """Select optimal region based on workload requirements."""
        
        gpu_required = requirements.get('gpu_required', False)
        high_memory = requirements.get('high_memory', False)
        quantum_simulation = requirements.get('quantum_simulation', False)
        compliance_required = requirements.get('compliance', [])
        
        candidate_regions = []
        
        # Filter regions based on compliance
        if compliance_required:
            compliant_regions = self.get_compliant_regions()
        else:
            compliant_regions = list(self.regions.keys())
        
        for region in compliant_regions:
            config = self.regions[region]
            
            # Check requirements
            if gpu_required and not config.gpu_enabled:
                continue
            if high_memory and not config.high_memory_instances:
                continue
            if quantum_simulation and not config.quantum_simulation_optimized:
                continue
            
            candidate_regions.append(region)
        
        # Prefer primary region if it meets requirements
        if self.primary_region and self.primary_region in candidate_regions:
            return self.primary_region
        
        # Return first suitable region
        return candidate_regions[0] if candidate_regions else None
    
    def generate_terraform_config(self, output_dir: str = "./terraform") -> None:
        """Generate Terraform infrastructure configuration."""
        
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main configuration
        main_tf = os.path.join(output_dir, "main.tf")
        with open(main_tf, 'w') as f:
            f.write(self._generate_main_terraform())
        
        # Variables
        vars_tf = os.path.join(output_dir, "variables.tf")
        with open(vars_tf, 'w') as f:
            f.write(self._generate_variables_terraform())
        
        # Outputs
        outputs_tf = os.path.join(output_dir, "outputs.tf")
        with open(outputs_tf, 'w') as f:
            f.write(self._generate_outputs_terraform())
        
        logger.info(f"Generated Terraform configuration in {output_dir}")
    
    def _generate_main_terraform(self) -> str:
        """Generate main Terraform configuration."""
        
        config = []
        config.append("terraform {")
        config.append("  required_version = \">= 1.0\"")
        config.append("  required_providers {")
        config.append("    aws = {")
        config.append("      source  = \"hashicorp/aws\"")
        config.append("      version = \"~> 5.0\"")
        config.append("    }")
        config.append("  }")
        config.append("}")
        config.append("")
        
        # Provider configurations for each region
        for region, region_config in self.regions.items():
            if region_config.platform == Platform.AWS:
                config.append(f"provider \"aws\" {{")
                config.append(f"  alias  = \"{region.value.replace('-', '_')}\"")
                config.append(f"  region = \"{region.value}\"")
                config.append("}")
                config.append("")
        
        # VPC configurations
        for region, region_config in self.regions.items():
            if region_config.platform == Platform.AWS:
                region_alias = region.value.replace('-', '_')
                
                config.append(f"# VPC for {region.value}")
                config.append(f"resource \"aws_vpc\" \"darkoperator_{region_alias}\" {{")
                config.append(f"  provider   = aws.{region_alias}")
                config.append(f"  cidr_block = \"{region_config.vpc_cidr}\"")
                config.append("")
                config.append("  enable_dns_hostnames = true")
                config.append("  enable_dns_support   = true")
                config.append("")
                config.append("  tags = {")
                config.append(f"    Name        = \"darkoperator-{region.value}\"")
                config.append("    Environment = var.environment")
                config.append("    Project     = \"darkoperator\"")
                config.append("  }")
                config.append("}")
                config.append("")
                
                # Subnets
                for i, subnet_cidr in enumerate(region_config.public_subnets):
                    config.append(f"resource \"aws_subnet\" \"public_{region_alias}_{i}\" {{")
                    config.append(f"  provider          = aws.{region_alias}")
                    config.append(f"  vpc_id            = aws_vpc.darkoperator_{region_alias}.id")
                    config.append(f"  cidr_block        = \"{subnet_cidr}\"")
                    if i < len(region_config.availability_zones):
                        config.append(f"  availability_zone = \"{region_config.availability_zones[i]}\"")
                    config.append("  map_public_ip_on_launch = true")
                    config.append("")
                    config.append("  tags = {")
                    config.append(f"    Name = \"darkoperator-public-{region.value}-{i}\"")
                    config.append("  }")
                    config.append("}")
                    config.append("")
        
        return "\n".join(config)
    
    def _generate_variables_terraform(self) -> str:
        """Generate Terraform variables."""
        
        variables = []
        variables.append("variable \"environment\" {")
        variables.append("  description = \"Deployment environment\"")
        variables.append("  type        = string")
        variables.append("  default     = \"production\"")
        variables.append("}")
        variables.append("")
        
        variables.append("variable \"project_name\" {")
        variables.append("  description = \"Project name\"")
        variables.append("  type        = string")
        variables.append("  default     = \"darkoperator\"")
        variables.append("}")
        variables.append("")
        
        # GPU instance types
        variables.append("variable \"gpu_instance_types\" {")
        variables.append("  description = \"GPU instance types by region\"")
        variables.append("  type        = map(string)")
        variables.append("  default = {")
        
        for region, config in self.regions.items():
            if config.gpu_enabled:
                variables.append(f"    \"{region.value}\" = \"{config.gpu_instance_type}\"")
        
        variables.append("  }")
        variables.append("}")
        variables.append("")
        
        return "\n".join(variables)
    
    def _generate_outputs_terraform(self) -> str:
        """Generate Terraform outputs."""
        
        outputs = []
        
        for region, region_config in self.regions.items():
            if region_config.platform == Platform.AWS:
                region_alias = region.value.replace('-', '_')
                
                outputs.append(f"output \"vpc_id_{region_alias}\" {{")
                outputs.append(f"  description = \"VPC ID for {region.value}\"")
                outputs.append(f"  value       = aws_vpc.darkoperator_{region_alias}.id")
                outputs.append("}")
                outputs.append("")
        
        return "\n".join(outputs)
    
    def generate_kubernetes_manifests(self, output_dir: str = "./k8s") -> None:
        """Generate Kubernetes deployment manifests."""
        
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Namespace
        namespace_yaml = os.path.join(output_dir, "namespace.yaml")
        with open(namespace_yaml, 'w') as f:
            f.write(self._generate_k8s_namespace())
        
        # ConfigMap for global configuration
        configmap_yaml = os.path.join(output_dir, "configmap.yaml")
        with open(configmap_yaml, 'w') as f:
            f.write(self._generate_k8s_configmap())
        
        # Deployment
        deployment_yaml = os.path.join(output_dir, "deployment.yaml")
        with open(deployment_yaml, 'w') as f:
            f.write(self._generate_k8s_deployment())
        
        # Service
        service_yaml = os.path.join(output_dir, "service.yaml")
        with open(service_yaml, 'w') as f:
            f.write(self._generate_k8s_service())
        
        # HPA (Horizontal Pod Autoscaler)
        hpa_yaml = os.path.join(output_dir, "hpa.yaml")
        with open(hpa_yaml, 'w') as f:
            f.write(self._generate_k8s_hpa())
        
        logger.info(f"Generated Kubernetes manifests in {output_dir}")
    
    def _generate_k8s_namespace(self) -> str:
        """Generate Kubernetes namespace."""
        
        return """apiVersion: v1
kind: Namespace
metadata:
  name: darkoperator
  labels:
    name: darkoperator
    project: darkoperator-studio
"""
    
    def _generate_k8s_configmap(self) -> str:
        """Generate Kubernetes ConfigMap."""
        
        # Export configuration as JSON string
        config_json = json.dumps(self.get_deployment_summary(), indent=2)
        
        return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: darkoperator-config
  namespace: darkoperator
data:
  global-config.json: |
{config_json}
  supported-languages: "{','.join(self.localization.supported_languages)}"
  compliance-frameworks: "{','.join([f.value for f in self.compliance.frameworks])}"
"""
    
    def _generate_k8s_deployment(self) -> str:
        """Generate Kubernetes Deployment."""
        
        gpu_support = any(config.gpu_enabled for config in self.regions.values())
        
        deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: darkoperator-api
  namespace: darkoperator
  labels:
    app: darkoperator-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: darkoperator-api
  template:
    metadata:
      labels:
        app: darkoperator-api
    spec:
      containers:
      - name: darkoperator-api
        image: darkoperator/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: CONFIG_PATH
          value: "/config/global-config.json"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi" 
            cpu: "1000m" """
        
        if gpu_support:
            deployment += """
            nvidia.com/gpu: 1"""
        
        deployment += """
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: darkoperator-config
"""
        
        if gpu_support:
            deployment += """      nodeSelector:
        accelerator: nvidia-tesla-k80
"""
        
        return deployment
    
    def _generate_k8s_service(self) -> str:
        """Generate Kubernetes Service."""
        
        return """apiVersion: v1
kind: Service
metadata:
  name: darkoperator-api-service
  namespace: darkoperator
  labels:
    app: darkoperator-api
spec:
  selector:
    app: darkoperator-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
"""
    
    def _generate_k8s_hpa(self) -> str:
        """Generate Kubernetes Horizontal Pod Autoscaler."""
        
        return """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: darkoperator-api-hpa
  namespace: darkoperator
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: darkoperator-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
    
    def generate_docker_compose(self, output_file: str = "docker-compose.yml") -> None:
        """Generate Docker Compose configuration."""
        
        gpu_support = any(config.gpu_enabled for config in self.regions.values())
        
        compose_config = {
            "version": "3.8",
            "services": {
                "darkoperator-api": {
                    "image": "darkoperator/api:latest",
                    "ports": ["8000:8000"],
                    "environment": {
                        "ENVIRONMENT": "production",
                        "CONFIG_PATH": "/app/config/global-config.json",
                        "SUPPORTED_LANGUAGES": ",".join(self.localization.supported_languages),
                        "COMPLIANCE_FRAMEWORKS": ",".join([f.value for f in self.compliance.frameworks])
                    },
                    "volumes": [
                        "./config:/app/config:ro"
                    ],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "darkoperator-worker": {
                    "image": "darkoperator/worker:latest",
                    "environment": {
                        "WORKER_TYPE": "quantum-scheduler",
                        "CONFIG_PATH": "/app/config/global-config.json"
                    },
                    "volumes": [
                        "./config:/app/config:ro"
                    ],
                    "restart": "unless-stopped",
                    "depends_on": ["darkoperator-api"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "restart": "unless-stopped",
                    "volumes": ["redis_data:/data"]
                },
                "postgres": {
                    "image": "postgres:15",
                    "environment": {
                        "POSTGRES_DB": "darkoperator",
                        "POSTGRES_USER": "darkoperator",
                        "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}"
                    },
                    "ports": ["5432:5432"],
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data",
                        "./init.sql:/docker-entrypoint-initdb.d/init.sql"
                    ],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "redis_data": {},
                "postgres_data": {}
            }
        }
        
        # Add GPU support if enabled
        if gpu_support:
            compose_config["services"]["darkoperator-worker"]["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "count": 1,
                                "capabilities": ["gpu"]
                            }
                        ]
                    }
                }
            }
        
        # Add monitoring stack
        compose_config["services"]["prometheus"] = {
            "image": "prom/prometheus:latest",
            "ports": ["9090:9090"],
            "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"],
            "restart": "unless-stopped"
        }
        
        compose_config["services"]["grafana"] = {
            "image": "grafana/grafana:latest",
            "ports": ["3000:3000"],
            "environment": {
                "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD}"
            },
            "volumes": ["grafana_data:/var/lib/grafana"],
            "restart": "unless-stopped"
        }
        
        compose_config["volumes"]["grafana_data"] = {}
        
        # Write docker-compose.yml
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info(f"Generated Docker Compose configuration: {output_file}")
    
    def export_for_cloud_formation(self, output_file: str = "cloudformation.json") -> None:
        """Export configuration as AWS CloudFormation template."""
        
        cf_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "DarkOperator Studio multi-region deployment",
            "Parameters": {
                "Environment": {
                    "Type": "String",
                    "Default": "production",
                    "Description": "Deployment environment"
                },
                "InstanceType": {
                    "Type": "String",
                    "Default": "t3.medium",
                    "Description": "EC2 instance type"
                }
            },
            "Resources": {},
            "Outputs": {}
        }
        
        # Add resources for each region
        for region, config in self.regions.items():
            if config.platform == Platform.AWS:
                region_name = region.value.replace('-', '')
                
                # VPC
                cf_template["Resources"][f"VPC{region_name}"] = {
                    "Type": "AWS::EC2::VPC",
                    "Properties": {
                        "CidrBlock": config.vpc_cidr,
                        "EnableDnsHostnames": True,
                        "EnableDnsSupport": True,
                        "Tags": [
                            {"Key": "Name", "Value": f"darkoperator-{region.value}"},
                            {"Key": "Project", "Value": "darkoperator"}
                        ]
                    }
                }
                
                # Internet Gateway
                cf_template["Resources"][f"IGW{region_name}"] = {
                    "Type": "AWS::EC2::InternetGateway",
                    "Properties": {
                        "Tags": [
                            {"Key": "Name", "Value": f"darkoperator-igw-{region.value}"}
                        ]
                    }
                }
                
                # Attach Gateway
                cf_template["Resources"][f"AttachGateway{region_name}"] = {
                    "Type": "AWS::EC2::VPCGatewayAttachment",
                    "Properties": {
                        "VpcId": {"Ref": f"VPC{region_name}"},
                        "InternetGatewayId": {"Ref": f"IGW{region_name}"}
                    }
                }
        
        # Save CloudFormation template
        with open(output_file, 'w') as f:
            json.dump(cf_template, f, indent=2)
        
        logger.info(f"Generated CloudFormation template: {output_file}")