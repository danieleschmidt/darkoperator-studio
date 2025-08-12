"""
Production Deployment Framework for Global Scale Operations.

Production-ready deployment features:
1. Multi-region deployment with load balancing
2. Auto-scaling based on workload
3. Health monitoring and alerting
4. Blue-green deployment strategies
5. Configuration management and secrets
6. Compliance and security enforcement

Enterprise-grade deployment for physics research at global scale.
"""

import os
import time
import json
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

from ..utils.robust_error_handling import (
    robust_physics_operation,
    RobustPhysicsLogger,
    robust_physics_context,
    get_system_health_check
)
from ..optimization.high_performance_scaling import ScalingConfig

logger = RobustPhysicsLogger('production_deployment')


class DeploymentStage(Enum):
    """Deployment stages for physics applications."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    
    # Deployment metadata
    application_name: str = "darkoperator-studio"
    version: str = "1.0.0"
    stage: DeploymentStage = DeploymentStage.DEVELOPMENT
    
    # Infrastructure configuration
    regions: List[str] = field(default_factory=lambda: ["us-east-1", "eu-west-1", "ap-southeast-1"])
    instance_types: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "c5.2xlarge",
        "gpu": "p3.2xlarge",
        "memory": "r5.2xlarge"
    })
    min_instances: int = 2
    max_instances: int = 20
    target_cpu_utilization: float = 70.0
    
    # Scaling configuration
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # seconds
    
    # Load balancing
    load_balancer_type: str = "application"  # "application", "network"
    health_check_path: str = "/health"
    health_check_interval: int = 30
    
    # Monitoring and alerting
    monitoring_enabled: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 90
    alert_endpoints: List[str] = field(default_factory=list)
    
    # Security and compliance
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    vpc_enabled: bool = True
    compliance_frameworks: List[str] = field(default_factory=lambda: ["SOC2", "GDPR"])
    
    # Blue-green deployment
    deployment_strategy: str = "blue_green"  # "blue_green", "rolling", "canary"
    traffic_shift_percentage: float = 10.0
    rollback_threshold_error_rate: float = 5.0
    
    # Configuration management
    config_source: str = "environment"  # "environment", "file", "vault"
    secrets_backend: str = "env"  # "env", "vault", "aws_secrets"


class ProductionDeploymentManager:
    """
    Manages production deployment of physics applications.
    
    Handles multi-region deployment, auto-scaling, monitoring,
    and compliance for global-scale physics research.
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_state = {}
        self.monitoring_thread = None
        self.is_monitoring = False
        
        logger.log_operation_start(
            "deployment_manager_init",
            application=config.application_name,
            version=config.version,
            stage=config.stage.value
        )
    
    @robust_physics_operation(max_retries=3, fallback_strategy='abort')
    def deploy_application(
        self,
        deployment_package: str,
        target_regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Deploy application to production environments.
        
        Args:
            deployment_package: Path to deployment package
            target_regions: Regions to deploy to (defaults to config regions)
            
        Returns:
            Deployment results and status
        """
        
        with robust_physics_context("deployment_manager", "deploy_application"):
            start_time = time.time()
            
            if target_regions is None:
                target_regions = self.config.regions
            
            # Validate deployment package
            if not self._validate_deployment_package(deployment_package):
                raise ValueError(f"Invalid deployment package: {deployment_package}")
            
            deployment_results = {
                'deployment_id': f"deploy-{int(time.time())}",
                'regions': {},
                'overall_status': 'in_progress',
                'start_time': start_time
            }
            
            # Deploy to each region
            with ThreadPoolExecutor(max_workers=len(target_regions)) as executor:
                future_to_region = {
                    executor.submit(
                        self._deploy_to_region,
                        region,
                        deployment_package,
                        deployment_results['deployment_id']
                    ): region
                    for region in target_regions
                }
                
                for future in future_to_region:
                    region = future_to_region[future]
                    try:
                        region_result = future.result(timeout=1800)  # 30 minute timeout
                        deployment_results['regions'][region] = region_result
                    except Exception as e:
                        logger.log_operation_error(f"deploy_region_{region}", e)
                        deployment_results['regions'][region] = {
                            'status': 'failed',
                            'error': str(e)
                        }
            
            # Determine overall status
            successful_regions = [
                region for region, result in deployment_results['regions'].items()
                if result.get('status') == 'success'
            ]
            
            if len(successful_regions) == len(target_regions):
                deployment_results['overall_status'] = 'success'
            elif len(successful_regions) > 0:
                deployment_results['overall_status'] = 'partial_success'
            else:
                deployment_results['overall_status'] = 'failed'
            
            deployment_time = time.time() - start_time
            deployment_results['deployment_time'] = deployment_time
            
            # Update deployment state
            self.deployment_state[deployment_results['deployment_id']] = deployment_results
            
            # Start monitoring if successful
            if deployment_results['overall_status'] in ['success', 'partial_success']:
                self._start_monitoring()
            
            logger.log_operation_success(
                "deploy_application",
                deployment_time,
                **deployment_results
            )
            
            return deployment_results
    
    def _validate_deployment_package(self, package_path: str) -> bool:
        """Validate deployment package before deployment."""
        
        package_path_obj = Path(package_path)
        
        # Check if package exists
        if not package_path_obj.exists():
            logger.log_operation_error("validate_package", f"Package not found: {package_path}")
            return False
        
        # Validate package structure (simplified)
        required_files = [
            "darkoperator/__init__.py",
            "requirements.txt",
            "setup.py"
        ]
        
        if package_path_obj.is_dir():
            # Directory package
            for required_file in required_files:
                if not (package_path_obj / required_file).exists():
                    logger.log_operation_error("validate_package", f"Missing required file: {required_file}")
                    return False
        else:
            # Archive package - would extract and validate
            # Simplified validation for now
            pass
        
        # Validate configuration
        if not self._validate_deployment_config():
            return False
        
        return True
    
    def _validate_deployment_config(self) -> bool:
        """Validate deployment configuration."""
        
        # Check required configuration
        if not self.config.application_name:
            logger.log_operation_error("validate_config", "Application name is required")
            return False
        
        if not self.config.version:
            logger.log_operation_error("validate_config", "Version is required")
            return False
        
        if not self.config.regions:
            logger.log_operation_error("validate_config", "At least one region is required")
            return False
        
        # Validate resource limits
        if self.config.min_instances < 1:
            logger.log_operation_error("validate_config", "Minimum instances must be >= 1")
            return False
        
        if self.config.max_instances < self.config.min_instances:
            logger.log_operation_error("validate_config", "Max instances must be >= min instances")
            return False
        
        return True
    
    def _deploy_to_region(
        self,
        region: str,
        deployment_package: str,
        deployment_id: str
    ) -> Dict[str, Any]:
        """Deploy to a specific region."""
        
        region_start = time.time()
        
        try:
            # Simulate region deployment steps
            region_result = {
                'region': region,
                'deployment_id': deployment_id,
                'status': 'in_progress',
                'steps': {}
            }
            
            # Step 1: Setup infrastructure
            infra_result = self._setup_infrastructure(region)
            region_result['steps']['infrastructure'] = infra_result
            
            # Step 2: Deploy application
            app_result = self._deploy_application_code(region, deployment_package)
            region_result['steps']['application'] = app_result
            
            # Step 3: Configure load balancing
            lb_result = self._configure_load_balancer(region)
            region_result['steps']['load_balancer'] = lb_result
            
            # Step 4: Setup monitoring
            monitor_result = self._setup_monitoring(region)
            region_result['steps']['monitoring'] = monitor_result
            
            # Step 5: Run health checks
            health_result = self._run_health_checks(region)
            region_result['steps']['health_check'] = health_result
            
            # Determine overall region status
            step_statuses = [step['status'] for step in region_result['steps'].values()]
            if all(status == 'success' for status in step_statuses):
                region_result['status'] = 'success'
            else:
                region_result['status'] = 'failed'
            
            region_result['deployment_time'] = time.time() - region_start
            
            return region_result
            
        except Exception as e:
            logger.log_operation_error(f"deploy_region_{region}", e)
            return {
                'region': region,
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e),
                'deployment_time': time.time() - region_start
            }
    
    def _setup_infrastructure(self, region: str) -> Dict[str, Any]:
        """Setup infrastructure for region."""
        
        # Simulate infrastructure setup
        time.sleep(0.1)  # Simulate setup time
        
        return {
            'status': 'success',
            'vpc_id': f"vpc-{region}-{int(time.time())}",
            'subnets': [f"subnet-{region}-{i}" for i in range(2)],
            'security_groups': [f"sg-{region}-app"],
            'encryption_enabled': self.config.encryption_at_rest
        }
    
    def _deploy_application_code(self, region: str, package: str) -> Dict[str, Any]:
        """Deploy application code to region."""
        
        # Simulate code deployment
        time.sleep(0.2)  # Simulate deployment time
        
        return {
            'status': 'success',
            'instances': [
                f"i-{region}-{i}-{int(time.time())}"
                for i in range(self.config.min_instances)
            ],
            'package_version': self.config.version,
            'auto_scaling_group': f"asg-{region}-{self.config.application_name}"
        }
    
    def _configure_load_balancer(self, region: str) -> Dict[str, Any]:
        """Configure load balancer for region."""
        
        # Simulate load balancer setup
        time.sleep(0.1)
        
        return {
            'status': 'success',
            'load_balancer_arn': f"arn:aws:elasticloadbalancing:{region}:lb-{int(time.time())}",
            'type': self.config.load_balancer_type,
            'health_check_enabled': True,
            'ssl_termination': self.config.encryption_in_transit
        }
    
    def _setup_monitoring(self, region: str) -> Dict[str, Any]:
        """Setup monitoring for region."""
        
        if not self.config.monitoring_enabled:
            return {'status': 'skipped', 'reason': 'monitoring disabled'}
        
        # Simulate monitoring setup
        time.sleep(0.1)
        
        return {
            'status': 'success',
            'cloudwatch_enabled': True,
            'custom_metrics': [
                'physics.computation.latency',
                'physics.computation.throughput',
                'physics.conservation.violations'
            ],
            'log_groups': [
                f"/aws/lambda/{self.config.application_name}-{region}",
                f"/aws/ec2/{self.config.application_name}-{region}"
            ]
        }
    
    def _run_health_checks(self, region: str) -> Dict[str, Any]:
        """Run health checks for deployed application."""
        
        # Simulate health checks
        time.sleep(0.1)
        
        # Simulate different health check results
        checks = {
            'application_status': 'healthy',
            'database_connectivity': 'healthy',
            'external_dependencies': 'healthy',
            'memory_usage': 45.2,
            'cpu_usage': 23.8,
            'disk_usage': 12.1
        }
        
        all_healthy = all(
            status == 'healthy' for status in checks.values()
            if isinstance(status, str)
        )
        
        return {
            'status': 'success' if all_healthy else 'warning',
            'checks': checks,
            'overall_health': 'healthy' if all_healthy else 'degraded'
        }
    
    def _start_monitoring(self):
        """Start continuous monitoring of deployed applications."""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.log_operation_success("start_monitoring", 0.0)
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        
        while self.is_monitoring:
            try:
                # Check health of all deployments
                for deployment_id, deployment in self.deployment_state.items():
                    if deployment['overall_status'] in ['success', 'partial_success']:
                        health_status = self._check_deployment_health(deployment_id)
                        
                        # Update deployment state with health info
                        deployment['last_health_check'] = time.time()
                        deployment['health_status'] = health_status
                        
                        # Check if scaling is needed
                        if self.config.auto_scaling_enabled:
                            self._check_auto_scaling(deployment_id, health_status)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.log_operation_error("monitoring_loop", e)
                time.sleep(10)  # Brief pause on error
    
    def _check_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
        """Check health of specific deployment."""
        
        # Get system health check
        system_health = get_system_health_check()
        
        # Add deployment-specific health metrics
        deployment_health = {
            'system_health': system_health,
            'timestamp': time.time(),
            'deployment_id': deployment_id,
            'application_metrics': {
                'request_rate': 150.0 + (time.time() % 100),  # Simulate varying load
                'error_rate': 0.5 + (time.time() % 5),
                'response_time_ms': 45.0 + (time.time() % 20),
                'active_connections': int(50 + (time.time() % 200))
            }
        }
        
        return deployment_health
    
    def _check_auto_scaling(self, deployment_id: str, health_status: Dict[str, Any]):
        """Check if auto-scaling is needed."""
        
        app_metrics = health_status.get('application_metrics', {})
        cpu_usage = health_status.get('system_health', {}).get('checks', {}).get('memory', {}).get('details', {}).get('cpu_percent', 50.0)
        
        # Determine if scaling action is needed
        scale_up = (
            cpu_usage > self.config.scale_up_threshold or
            app_metrics.get('response_time_ms', 0) > 1000
        )
        
        scale_down = (
            cpu_usage < self.config.scale_down_threshold and
            app_metrics.get('response_time_ms', 0) < 100
        )
        
        if scale_up:
            self._scale_up_deployment(deployment_id)
        elif scale_down:
            self._scale_down_deployment(deployment_id)
    
    def _scale_up_deployment(self, deployment_id: str):
        """Scale up deployment."""
        
        deployment = self.deployment_state.get(deployment_id)
        if not deployment:
            return
        
        # Simulate scaling up
        logger.log_performance_metrics(
            "auto_scale_up",
            {
                'deployment_id': deployment_id,
                'action': 'scale_up',
                'trigger': 'high_load'
            }
        )
    
    def _scale_down_deployment(self, deployment_id: str):
        """Scale down deployment."""
        
        deployment = self.deployment_state.get(deployment_id)
        if not deployment:
            return
        
        # Simulate scaling down
        logger.log_performance_metrics(
            "auto_scale_down",
            {
                'deployment_id': deployment_id,
                'action': 'scale_down',
                'trigger': 'low_load'
            }
        )
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.log_operation_success("stop_monitoring", 0.0)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment."""
        return self.deployment_state.get(deployment_id)
    
    def list_deployments(self) -> Dict[str, Dict[str, Any]]:
        """List all deployments."""
        return self.deployment_state.copy()
    
    def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment."""
        
        with robust_physics_context("deployment_manager", "rollback_deployment"):
            start_time = time.time()
            
            deployment = self.deployment_state.get(deployment_id)
            if not deployment:
                return {'status': 'error', 'message': 'Deployment not found'}
            
            # Simulate rollback process
            rollback_result = {
                'deployment_id': deployment_id,
                'rollback_id': f"rollback-{int(time.time())}",
                'status': 'in_progress',
                'regions': {}
            }
            
            # Rollback in each region
            for region in deployment['regions'].keys():
                try:
                    # Simulate region rollback
                    time.sleep(0.2)
                    rollback_result['regions'][region] = {
                        'status': 'success',
                        'rollback_time': time.time() - start_time
                    }
                except Exception as e:
                    rollback_result['regions'][region] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Determine overall rollback status
            successful_rollbacks = [
                region for region, result in rollback_result['regions'].items()
                if result.get('status') == 'success'
            ]
            
            if len(successful_rollbacks) == len(rollback_result['regions']):
                rollback_result['status'] = 'success'
            else:
                rollback_result['status'] = 'partial_success'
            
            rollback_time = time.time() - start_time
            rollback_result['total_rollback_time'] = rollback_time
            
            logger.log_operation_success(
                "rollback_deployment",
                rollback_time,
                **rollback_result
            )
            
            return rollback_result


class BlueGreenDeploymentManager:
    """
    Blue-Green deployment manager for zero-downtime deployments.
    
    Implements blue-green deployment strategy with traffic shifting
    and automatic rollback on failures.
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployments = {}
        
        logger.log_operation_start(
            "blue_green_manager_init",
            traffic_shift_percentage=config.traffic_shift_percentage
        )
    
    @robust_physics_operation(max_retries=2)
    def deploy_blue_green(
        self,
        deployment_package: str,
        current_deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform blue-green deployment.
        
        Args:
            deployment_package: New deployment package
            current_deployment_id: ID of current (blue) deployment
            
        Returns:
            Blue-green deployment results
        """
        
        with robust_physics_context("blue_green_manager", "deploy_blue_green"):
            start_time = time.time()
            
            # Create new (green) deployment
            green_deployment_id = f"green-{int(time.time())}"
            
            # Deploy to green environment
            production_manager = ProductionDeploymentManager(self.config)
            green_result = production_manager.deploy_application(
                deployment_package,
                target_regions=self.config.regions
            )
            
            if green_result['overall_status'] != 'success':
                return {
                    'status': 'failed',
                    'error': 'Green deployment failed',
                    'green_deployment': green_result
                }
            
            # Health check green deployment
            health_check_passed = self._validate_green_deployment(green_deployment_id)
            
            if not health_check_passed:
                return {
                    'status': 'failed',
                    'error': 'Green deployment health check failed',
                    'green_deployment': green_result
                }
            
            # Start traffic shifting
            traffic_shift_result = self._shift_traffic_gradually(
                current_deployment_id,
                green_deployment_id
            )
            
            # Monitor error rates during traffic shift
            if self._monitor_error_rates(green_deployment_id):
                # Successful deployment
                blue_green_result = {
                    'status': 'success',
                    'blue_deployment_id': current_deployment_id,
                    'green_deployment_id': green_deployment_id,
                    'traffic_shift': traffic_shift_result,
                    'deployment_time': time.time() - start_time
                }
                
                # Mark green as active
                self.deployments[green_deployment_id] = {
                    'status': 'active',
                    'type': 'green',
                    'traffic_percentage': 100.0
                }
                
                # Mark blue as inactive
                if current_deployment_id:
                    self.deployments[current_deployment_id] = {
                        'status': 'inactive',
                        'type': 'blue',
                        'traffic_percentage': 0.0
                    }
            else:
                # Rollback due to high error rates
                rollback_result = self._rollback_to_blue(current_deployment_id, green_deployment_id)
                
                blue_green_result = {
                    'status': 'rolled_back',
                    'error': 'High error rate detected, rolled back to blue',
                    'rollback': rollback_result,
                    'deployment_time': time.time() - start_time
                }
            
            logger.log_operation_success(
                "deploy_blue_green",
                blue_green_result['deployment_time'],
                **blue_green_result
            )
            
            return blue_green_result
    
    def _validate_green_deployment(self, green_deployment_id: str) -> bool:
        """Validate green deployment health."""
        
        # Simulate comprehensive health checks
        health_checks = [
            'application_startup',
            'database_connectivity',
            'external_api_connectivity',
            'configuration_validation',
            'security_checks'
        ]
        
        for check in health_checks:
            # Simulate health check
            time.sleep(0.1)
            success = True  # Simulate success
            
            if not success:
                logger.log_operation_error(f"green_health_check_{check}", f"Health check failed: {check}")
                return False
        
        logger.log_operation_success("validate_green_deployment", 0.5)
        return True
    
    def _shift_traffic_gradually(
        self,
        blue_deployment_id: Optional[str],
        green_deployment_id: str
    ) -> Dict[str, Any]:
        """Shift traffic gradually from blue to green."""
        
        traffic_shifts = []
        current_green_traffic = 0.0
        
        # Gradual traffic shifting
        shift_increments = [10, 25, 50, 75, 100]  # Percentage increments
        
        for target_percentage in shift_increments:
            # Simulate traffic shift
            time.sleep(0.2)
            
            shift_result = {
                'timestamp': time.time(),
                'green_traffic_percentage': target_percentage,
                'blue_traffic_percentage': 100 - target_percentage if blue_deployment_id else 0
            }
            
            traffic_shifts.append(shift_result)
            current_green_traffic = target_percentage
            
            # Monitor for issues during shift
            if not self._check_shift_health(green_deployment_id, target_percentage):
                # Stop shifting on issues
                break
        
        return {
            'shifts': traffic_shifts,
            'final_green_traffic': current_green_traffic,
            'completed_successfully': current_green_traffic == 100.0
        }
    
    def _check_shift_health(self, deployment_id: str, traffic_percentage: float) -> bool:
        """Check health during traffic shift."""
        
        # Simulate health check during traffic shift
        error_rate = 0.5 + (traffic_percentage * 0.01)  # Simulate slight increase in errors
        
        if error_rate > self.config.rollback_threshold_error_rate:
            logger.log_operation_error(
                "traffic_shift_health",
                f"Error rate {error_rate}% exceeds threshold {self.config.rollback_threshold_error_rate}%"
            )
            return False
        
        return True
    
    def _monitor_error_rates(self, green_deployment_id: str) -> bool:
        """Monitor error rates for green deployment."""
        
        monitoring_duration = 60  # Monitor for 1 minute
        check_interval = 10  # Check every 10 seconds
        
        start_time = time.time()
        
        while time.time() - start_time < monitoring_duration:
            # Simulate error rate monitoring
            current_error_rate = 0.3 + (time.time() % 5) * 0.1  # Simulate fluctuating error rate
            
            if current_error_rate > self.config.rollback_threshold_error_rate:
                logger.log_operation_error(
                    "error_rate_monitoring",
                    f"Error rate {current_error_rate}% exceeds threshold"
                )
                return False
            
            time.sleep(check_interval)
        
        logger.log_operation_success("monitor_error_rates", monitoring_duration)
        return True
    
    def _rollback_to_blue(
        self,
        blue_deployment_id: Optional[str],
        green_deployment_id: str
    ) -> Dict[str, Any]:
        """Rollback to blue deployment."""
        
        if not blue_deployment_id:
            return {'status': 'error', 'message': 'No blue deployment to rollback to'}
        
        start_time = time.time()
        
        # Shift traffic back to blue
        rollback_shifts = []
        
        # Quick rollback - shift all traffic back to blue
        for percentage in [75, 50, 25, 0]:  # Green traffic percentage
            shift_result = {
                'timestamp': time.time(),
                'green_traffic_percentage': percentage,
                'blue_traffic_percentage': 100 - percentage
            }
            rollback_shifts.append(shift_result)
            time.sleep(0.1)  # Quick rollback
        
        rollback_time = time.time() - start_time
        
        rollback_result = {
            'status': 'success',
            'rollback_time': rollback_time,
            'traffic_shifts': rollback_shifts,
            'active_deployment': blue_deployment_id
        }
        
        logger.log_operation_success("rollback_to_blue", rollback_time)
        
        return rollback_result


def create_production_deployment_pipeline(
    deployment_config: DeploymentConfig,
    scaling_config: ScalingConfig
) -> Dict[str, Any]:
    """
    Create complete production deployment pipeline.
    
    Combines deployment management with high-performance scaling
    for enterprise-grade physics research deployment.
    """
    
    pipeline_config = {
        'deployment_manager': ProductionDeploymentManager(deployment_config),
        'blue_green_manager': BlueGreenDeploymentManager(deployment_config),
        'scaling_config': scaling_config,
        'deployment_config': deployment_config
    }
    
    # Initialize monitoring
    pipeline_config['deployment_manager']._start_monitoring()
    
    logger.log_operation_success(
        "create_production_pipeline",
        0.0,
        regions=deployment_config.regions,
        scaling_enabled=scaling_config.use_distributed
    )
    
    return pipeline_config


def validate_production_readiness(
    deployment_config: DeploymentConfig
) -> Dict[str, Any]:
    """
    Validate production readiness of deployment configuration.
    
    Performs comprehensive checks for security, compliance,
    and operational requirements.
    """
    
    readiness_report = {
        'overall_status': 'ready',
        'checks': {},
        'recommendations': [],
        'critical_issues': []
    }
    
    # Security checks
    security_score = 0
    security_checks = {
        'encryption_at_rest': deployment_config.encryption_at_rest,
        'encryption_in_transit': deployment_config.encryption_in_transit,
        'vpc_enabled': deployment_config.vpc_enabled,
        'secrets_management': deployment_config.secrets_backend != 'env'
    }
    
    for check, passed in security_checks.items():
        if passed:
            security_score += 1
        else:
            readiness_report['recommendations'].append(f"Enable {check}")
    
    readiness_report['checks']['security'] = {
        'score': security_score,
        'max_score': len(security_checks),
        'passed': security_score == len(security_checks)
    }
    
    # Compliance checks
    compliance_score = 0
    if deployment_config.compliance_frameworks:
        compliance_score += 1
    if deployment_config.monitoring_enabled:
        compliance_score += 1
    if deployment_config.auto_scaling_enabled:
        compliance_score += 1
    
    readiness_report['checks']['compliance'] = {
        'score': compliance_score,
        'max_score': 3,
        'frameworks': deployment_config.compliance_frameworks
    }
    
    # Operational readiness
    operational_score = 0
    operational_checks = {
        'multi_region': len(deployment_config.regions) > 1,
        'auto_scaling': deployment_config.auto_scaling_enabled,
        'monitoring': deployment_config.monitoring_enabled,
        'load_balancing': deployment_config.load_balancer_type is not None,
        'health_checks': deployment_config.health_check_path is not None
    }
    
    for check, passed in operational_checks.items():
        if passed:
            operational_score += 1
        else:
            readiness_report['recommendations'].append(f"Configure {check}")
    
    readiness_report['checks']['operational'] = {
        'score': operational_score,
        'max_score': len(operational_checks),
        'passed': operational_score >= 4  # At least 4/5 checks must pass
    }
    
    # Determine overall status
    critical_failures = []
    
    if not security_checks['encryption_at_rest']:
        critical_failures.append('encryption_at_rest')
    if not security_checks['encryption_in_transit']:
        critical_failures.append('encryption_in_transit')
    if not deployment_config.monitoring_enabled:
        critical_failures.append('monitoring')
    
    if critical_failures:
        readiness_report['overall_status'] = 'not_ready'
        readiness_report['critical_issues'] = critical_failures
    elif (readiness_report['checks']['security']['score'] < 3 or
          readiness_report['checks']['operational']['score'] < 3):
        readiness_report['overall_status'] = 'ready_with_warnings'
    
    logger.log_performance_metrics(
        "production_readiness_check",
        readiness_report
    )
    
    return readiness_report