"""
Autonomous Production Scaler for Global Deployment

This module implements intelligent auto-scaling for DarkOperator Studio including:
- Real-time load balancing across global regions
- Predictive scaling based on physics event rates
- Quantum-classical resource optimization
- Cost-aware scaling decisions
- Multi-cloud deployment coordination
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
import psutil
import requests


class ScalingDecision(Enum):
    """Possible scaling decisions."""
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    MAINTAIN = "MAINTAIN"
    MIGRATE = "MIGRATE"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    disk_io: float
    active_connections: int
    inference_queue_length: int
    average_latency_ms: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 100
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_gpu_utilization: float = 85.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # seconds
    evaluation_period: int = 60  # seconds


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region_name: str
    endpoint_url: str
    max_capacity: int
    cost_per_hour: float
    network_latency_ms: float
    availability_zones: List[str] = field(default_factory=list)
    gpu_types: List[str] = field(default_factory=list)


class AutonomousProductionScaler:
    """
    Intelligent auto-scaling system for global DarkOperator deployment.
    
    Features:
    - Predictive scaling based on physics workload patterns
    - Cost optimization across multiple cloud providers
    - Regional load balancing for optimal performance
    - Quantum-classical resource coordination
    """
    
    def __init__(
        self,
        regions: List[RegionConfig],
        scaling_policy: Optional[ScalingPolicy] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.regions = {region.region_name: region for region in regions}
        self.policy = scaling_policy or ScalingPolicy()
        
        # Current deployment state
        self.current_instances: Dict[str, int] = {name: 1 for name in self.regions}
        self.resource_history: Dict[str, List[ResourceMetrics]] = {
            name: [] for name in self.regions
        }
        
        # Scaling state
        self.last_scaling_time: Dict[str, float] = {name: 0 for name in self.regions}
        self.is_scaling_active = False
        
        # Predictive models
        self.load_predictor = self._initialize_load_predictor()
        self.cost_optimizer = CostOptimizer(self.regions)
        
        # Physics-specific metrics
        self.physics_workload_patterns = {}
        self.event_rate_history = []
        
    def _initialize_load_predictor(self) -> 'LoadPredictor':
        """Initialize predictive load forecasting model."""
        return LoadPredictor()
    
    async def start_autonomous_scaling(self):
        """Start the autonomous scaling system."""
        if self.is_scaling_active:
            self.logger.warning("Scaling already active")
            return
        
        self.is_scaling_active = True
        self.logger.info("Starting autonomous production scaling")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._scaling_decision_loop()),
            asyncio.create_task(self._cost_optimization_loop()),
            asyncio.create_task(self._physics_workload_analyzer()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in scaling system: {e}")
        finally:
            self.is_scaling_active = False
    
    async def stop_autonomous_scaling(self):
        """Stop the autonomous scaling system."""
        self.is_scaling_active = False
        self.logger.info("Autonomous scaling stopped")
    
    async def _monitoring_loop(self):
        """Continuously monitor resource utilization across all regions."""
        while self.is_scaling_active:
            try:
                # Collect metrics from all regions
                for region_name in self.regions:
                    metrics = await self._collect_region_metrics(region_name)
                    self.resource_history[region_name].append(metrics)
                    
                    # Keep only recent history
                    max_history = 1000
                    if len(self.resource_history[region_name]) > max_history:
                        self.resource_history[region_name] = \
                            self.resource_history[region_name][-max_history:]
                
                await asyncio.sleep(self.policy.evaluation_period)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_region_metrics(self, region_name: str) -> ResourceMetrics:
        """Collect resource metrics for a specific region."""
        # In production, this would query actual region endpoints
        # Here we simulate realistic metrics
        
        timestamp = time.time()
        
        # Simulate metrics based on current load
        base_cpu = 50.0 + np.random.normal(0, 10)
        base_memory = 60.0 + np.random.normal(0, 15)
        base_gpu = 70.0 + np.random.normal(0, 12)
        
        # Add physics workload patterns
        physics_factor = self._get_physics_load_factor(timestamp)
        
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_usage=max(0, min(100, base_cpu * physics_factor)),
            memory_usage=max(0, min(100, base_memory * physics_factor)),
            gpu_usage=max(0, min(100, base_gpu * physics_factor)),
            network_io=np.random.uniform(10, 100),  # MB/s
            disk_io=np.random.uniform(5, 50),       # MB/s
            active_connections=int(np.random.uniform(10, 1000)),
            inference_queue_length=int(np.random.uniform(0, 50)),
            average_latency_ms=np.random.uniform(10, 200)
        )
        
        return metrics
    
    def _get_physics_load_factor(self, timestamp: float) -> float:
        """Calculate load factor based on physics event patterns."""
        # Simulate LHC-like beam schedule
        hour = (timestamp % 86400) / 3600  # Hour of day
        
        # Higher load during "beam time" (8am - 6pm CERN time)
        if 8 <= hour <= 18:
            base_factor = 1.2  # Higher load during physics runs
        else:
            base_factor = 0.6  # Lower load during maintenance
        
        # Add periodic variations (simulating bunch crossings)
        periodic_factor = 1.0 + 0.3 * np.sin(2 * np.pi * timestamp / 3600)  # Hourly cycle
        
        return base_factor * periodic_factor
    
    async def _scaling_decision_loop(self):
        """Main loop for making scaling decisions."""
        while self.is_scaling_active:
            try:
                # Analyze each region
                for region_name in self.regions:
                    decision = await self._make_scaling_decision(region_name)
                    
                    if decision != ScalingDecision.MAINTAIN:
                        await self._execute_scaling_decision(region_name, decision)
                
                await asyncio.sleep(self.policy.evaluation_period)
                
            except Exception as e:
                self.logger.error(f"Error in scaling decision loop: {e}")
                await asyncio.sleep(30)
    
    async def _make_scaling_decision(self, region_name: str) -> ScalingDecision:
        """Make intelligent scaling decision for a region."""
        if not self.resource_history[region_name]:
            return ScalingDecision.MAINTAIN
        
        # Check cooldown period
        time_since_last_scaling = time.time() - self.last_scaling_time[region_name]
        if time_since_last_scaling < self.policy.cooldown_period:
            return ScalingDecision.MAINTAIN
        
        # Get recent metrics
        recent_metrics = self.resource_history[region_name][-5:]  # Last 5 measurements
        
        if not recent_metrics:
            return ScalingDecision.MAINTAIN
        
        # Calculate average utilization
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_gpu = np.mean([m.gpu_usage for m in recent_metrics])
        avg_latency = np.mean([m.average_latency_ms for m in recent_metrics])
        
        # Get predictive load forecast
        predicted_load = await self._predict_future_load(region_name)
        
        # Make decision based on current and predicted metrics
        current_instances = self.current_instances[region_name]
        
        # Scale up conditions
        scale_up_needed = (
            avg_cpu > self.policy.scale_up_threshold or
            avg_memory > self.policy.scale_up_threshold or
            avg_gpu > self.policy.scale_up_threshold or
            avg_latency > 150.0 or  # High latency threshold
            predicted_load > 1.2    # Predicted 20% load increase
        )
        
        # Scale down conditions
        scale_down_needed = (
            avg_cpu < self.policy.scale_down_threshold and
            avg_memory < self.policy.scale_down_threshold and
            avg_gpu < self.policy.scale_down_threshold and
            avg_latency < 50.0 and  # Low latency
            predicted_load < 0.8 and  # Predicted load decrease
            current_instances > self.policy.min_instances
        )
        
        if scale_up_needed and current_instances < self.policy.max_instances:
            return ScalingDecision.SCALE_UP
        elif scale_down_needed:
            return ScalingDecision.SCALE_DOWN
        else:
            return ScalingDecision.MAINTAIN
    
    async def _predict_future_load(self, region_name: str) -> float:
        """Predict future load for a region."""
        if not self.resource_history[region_name]:
            return 1.0
        
        # Use simple trend analysis (in production, use ML models)
        recent_cpu = [m.cpu_usage for m in self.resource_history[region_name][-10:]]
        
        if len(recent_cpu) < 2:
            return 1.0
        
        # Calculate trend
        x = np.arange(len(recent_cpu))
        trend = np.polyfit(x, recent_cpu, 1)[0]  # Linear trend
        
        # Predict relative change
        prediction_factor = 1.0 + (trend / 100.0)  # Convert trend to factor
        
        return max(0.1, min(3.0, prediction_factor))  # Clamp between 0.1 and 3.0
    
    async def _execute_scaling_decision(
        self,
        region_name: str,
        decision: ScalingDecision
    ):
        """Execute a scaling decision."""
        current_instances = self.current_instances[region_name]
        
        if decision == ScalingDecision.SCALE_UP:
            new_instances = min(current_instances + 1, self.policy.max_instances)
            action = "scale up"
        elif decision == ScalingDecision.SCALE_DOWN:
            new_instances = max(current_instances - 1, self.policy.min_instances)
            action = "scale down"
        else:
            return
        
        # Calculate cost impact
        cost_impact = await self._calculate_cost_impact(region_name, new_instances)
        
        self.logger.info(
            f"Scaling {region_name}: {action} from {current_instances} to {new_instances} instances"
            f" (cost impact: ${cost_impact:.2f}/hour)"
        )
        
        # Execute scaling (in production, this would call cloud APIs)
        success = await self._perform_scaling_operation(region_name, new_instances)
        
        if success:
            self.current_instances[region_name] = new_instances
            self.last_scaling_time[region_name] = time.time()
            
            # Log scaling event
            await self._log_scaling_event(region_name, action, current_instances, new_instances)
        else:
            self.logger.error(f"Failed to scale {region_name}")
    
    async def _perform_scaling_operation(
        self,
        region_name: str,
        target_instances: int
    ) -> bool:
        """Perform the actual scaling operation."""
        # In production, this would call cloud provider APIs
        # For demonstration, we simulate the operation
        
        try:
            # Simulate API call delay
            await asyncio.sleep(np.random.uniform(1, 3))
            
            # Simulate 95% success rate
            success = np.random.random() > 0.05
            
            if success:
                self.logger.info(f"Successfully scaled {region_name} to {target_instances} instances")
            else:
                self.logger.error(f"Scaling operation failed for {region_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during scaling operation: {e}")
            return False
    
    async def _calculate_cost_impact(self, region_name: str, new_instances: int) -> float:
        """Calculate hourly cost impact of scaling decision."""
        region = self.regions[region_name]
        current_instances = self.current_instances[region_name]
        
        current_cost = current_instances * region.cost_per_hour
        new_cost = new_instances * region.cost_per_hour
        
        return new_cost - current_cost
    
    async def _cost_optimization_loop(self):
        """Continuously optimize costs across regions."""
        while self.is_scaling_active:
            try:
                # Run cost optimization every 15 minutes
                await asyncio.sleep(900)
                
                optimization = await self.cost_optimizer.optimize_regional_distribution(
                    self.current_instances,
                    self.resource_history
                )
                
                if optimization['savings_potential'] > 50.0:  # $50/hour savings
                    self.logger.info(f"Cost optimization opportunity: ${optimization['savings_potential']:.2f}/hour")
                    await self._apply_cost_optimization(optimization)
                
            except Exception as e:
                self.logger.error(f"Error in cost optimization: {e}")
                await asyncio.sleep(300)
    
    async def _apply_cost_optimization(self, optimization: Dict[str, Any]):
        """Apply cost optimization recommendations."""
        recommendations = optimization.get('recommendations', [])
        
        for rec in recommendations:
            if rec['action'] == 'migrate_workload':
                await self._migrate_workload(
                    rec['from_region'],
                    rec['to_region'],
                    rec['instances']
                )
    
    async def _migrate_workload(
        self,
        from_region: str,
        to_region: str,
        instances: int
    ):
        """Migrate workload between regions for cost optimization."""
        self.logger.info(f"Migrating {instances} instances from {from_region} to {to_region}")
        
        # Gradual migration to avoid service disruption
        for i in range(instances):
            # Scale up target region
            await self._perform_scaling_operation(
                to_region,
                self.current_instances[to_region] + 1
            )
            
            # Wait for stability
            await asyncio.sleep(30)
            
            # Scale down source region
            await self._perform_scaling_operation(
                from_region,
                self.current_instances[from_region] - 1
            )
            
            # Update instance counts
            self.current_instances[to_region] += 1
            self.current_instances[from_region] -= 1
    
    async def _physics_workload_analyzer(self):
        """Analyze physics workload patterns for better prediction."""
        while self.is_scaling_active:
            try:
                # Analyze event patterns every 5 minutes
                await asyncio.sleep(300)
                
                # Simulate physics event rate analysis
                current_event_rate = self._simulate_physics_event_rate()
                self.event_rate_history.append({
                    'timestamp': time.time(),
                    'event_rate': current_event_rate,
                    'event_type': self._classify_physics_events(current_event_rate)
                })
                
                # Keep only recent history
                if len(self.event_rate_history) > 288:  # 24 hours at 5-min intervals
                    self.event_rate_history = self.event_rate_history[-288:]
                
                # Update workload patterns
                await self._update_physics_patterns()
                
            except Exception as e:
                self.logger.error(f"Error in physics workload analysis: {e}")
                await asyncio.sleep(60)
    
    def _simulate_physics_event_rate(self) -> float:
        """Simulate realistic physics event rates."""
        # Simulate LHC bunch crossing rate and trigger rates
        base_rate = 40e6  # 40 MHz bunch crossing rate
        trigger_rate = base_rate * np.random.uniform(1e-6, 1e-4)  # Trigger reduction
        
        return trigger_rate
    
    def _classify_physics_events(self, event_rate: float) -> str:
        """Classify physics events based on rate."""
        if event_rate > 1000:
            return "high_luminosity"
        elif event_rate > 100:
            return "normal_physics"
        elif event_rate > 10:
            return "low_luminosity"
        else:
            return "cosmic_rays"
    
    async def _update_physics_patterns(self):
        """Update physics workload patterns for prediction."""
        if len(self.event_rate_history) < 10:
            return
        
        # Analyze patterns
        recent_rates = [e['event_rate'] for e in self.event_rate_history[-24:]]  # Last 2 hours
        
        pattern_analysis = {
            'mean_rate': float(np.mean(recent_rates)),
            'rate_variance': float(np.var(recent_rates)),
            'trend': float(np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]),
            'peak_detection': self._detect_rate_peaks(recent_rates)
        }
        
        self.physics_workload_patterns = pattern_analysis
    
    def _detect_rate_peaks(self, rates: List[float]) -> Dict[str, Any]:
        """Detect peaks in event rates."""
        if len(rates) < 5:
            return {'peaks_detected': 0}
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(rates) - 1):
            if rates[i] > rates[i-1] and rates[i] > rates[i+1]:
                peaks.append(i)
        
        return {
            'peaks_detected': len(peaks),
            'average_peak_height': float(np.mean([rates[p] for p in peaks])) if peaks else 0.0,
            'peak_frequency': len(peaks) / len(rates) if rates else 0.0
        }
    
    async def _log_scaling_event(
        self,
        region_name: str,
        action: str,
        old_instances: int,
        new_instances: int
    ):
        """Log scaling events for analysis."""
        event = {
            'timestamp': time.time(),
            'region': region_name,
            'action': action,
            'old_instances': old_instances,
            'new_instances': new_instances,
            'trigger_metrics': self.resource_history[region_name][-1].__dict__ if self.resource_history[region_name] else {},
            'physics_context': self.physics_workload_patterns
        }
        
        # In production, send to logging/monitoring system
        self.logger.info(f"Scaling event logged: {json.dumps(event, indent=2)}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status across all regions."""
        total_instances = sum(self.current_instances.values())
        total_cost = sum(
            self.current_instances[name] * region.cost_per_hour
            for name, region in self.regions.items()
        )
        
        status = {
            'total_instances': total_instances,
            'total_hourly_cost': total_cost,
            'regional_distribution': self.current_instances.copy(),
            'scaling_active': self.is_scaling_active,
            'physics_patterns': self.physics_workload_patterns,
            'recent_scaling_events': len([
                region for region, last_time in self.last_scaling_time.items()
                if time.time() - last_time < 3600  # Last hour
            ])
        }
        
        return status


class LoadPredictor:
    """Simple load prediction model."""
    
    def __init__(self):
        self.history_window = 50
    
    async def predict_load(
        self,
        metrics_history: List[ResourceMetrics],
        prediction_horizon: int = 5
    ) -> List[float]:
        """Predict future load based on historical metrics."""
        if len(metrics_history) < 3:
            return [1.0] * prediction_horizon
        
        # Use simple trend extrapolation
        recent_cpu = [m.cpu_usage for m in metrics_history[-self.history_window:]]
        
        # Fit linear trend
        x = np.arange(len(recent_cpu))
        trend_coef = np.polyfit(x, recent_cpu, 1)[0] if len(recent_cpu) > 1 else 0
        
        # Predict future values
        predictions = []
        last_value = recent_cpu[-1]
        
        for i in range(1, prediction_horizon + 1):
            predicted_value = last_value + (trend_coef * i)
            # Normalize to 0-100 range and convert to load factor
            load_factor = max(0.1, min(3.0, predicted_value / 100.0))
            predictions.append(load_factor)
        
        return predictions


class CostOptimizer:
    """Cost optimization across regions."""
    
    def __init__(self, regions: Dict[str, RegionConfig]):
        self.regions = regions
        self.logger = logging.getLogger(__name__)
    
    async def optimize_regional_distribution(
        self,
        current_instances: Dict[str, int],
        resource_history: Dict[str, List[ResourceMetrics]]
    ) -> Dict[str, Any]:
        """Optimize instance distribution across regions for cost."""
        
        # Calculate current cost
        current_cost = sum(
            instances * self.regions[region].cost_per_hour
            for region, instances in current_instances.items()
        )
        
        # Find optimal distribution
        optimal_distribution = self._find_optimal_distribution(
            current_instances, resource_history
        )
        
        # Calculate potential savings
        optimal_cost = sum(
            instances * self.regions[region].cost_per_hour
            for region, instances in optimal_distribution.items()
        )
        
        savings = current_cost - optimal_cost
        
        # Generate recommendations
        recommendations = []
        for region in current_instances:
            current = current_instances[region]
            optimal = optimal_distribution[region]
            
            if current > optimal:
                recommendations.append({
                    'action': 'migrate_workload',
                    'from_region': region,
                    'to_region': self._find_cheapest_alternative(region),
                    'instances': current - optimal
                })
        
        return {
            'current_cost': current_cost,
            'optimal_cost': optimal_cost,
            'savings_potential': savings,
            'recommendations': recommendations,
            'optimal_distribution': optimal_distribution
        }
    
    def _find_optimal_distribution(
        self,
        current_instances: Dict[str, int],
        resource_history: Dict[str, List[ResourceMetrics]]
    ) -> Dict[str, int]:
        """Find optimal instance distribution."""
        total_instances = sum(current_instances.values())
        
        # Sort regions by cost efficiency (cost per performance unit)
        region_efficiency = []
        
        for region_name, region_config in self.regions.items():
            # Calculate average performance (inverse of latency)
            if resource_history[region_name]:
                avg_latency = np.mean([
                    m.average_latency_ms for m in resource_history[region_name][-10:]
                ])
                performance = 1.0 / max(avg_latency, 1.0)  # Avoid division by zero
            else:
                performance = 1.0 / region_config.network_latency_ms
            
            efficiency = performance / region_config.cost_per_hour
            region_efficiency.append((region_name, efficiency))
        
        # Sort by efficiency (higher is better)
        region_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute instances to most efficient regions first
        optimal_distribution = {region: 0 for region in current_instances}
        remaining_instances = total_instances
        
        for region_name, _ in region_efficiency:
            if remaining_instances <= 0:
                break
            
            # Assign instances up to region capacity
            region_capacity = self.regions[region_name].max_capacity
            assigned = min(remaining_instances, region_capacity)
            optimal_distribution[region_name] = assigned
            remaining_instances -= assigned
        
        return optimal_distribution
    
    def _find_cheapest_alternative(self, current_region: str) -> str:
        """Find the cheapest alternative region."""
        current_cost = self.regions[current_region].cost_per_hour
        
        alternatives = [
            (name, config.cost_per_hour)
            for name, config in self.regions.items()
            if name != current_region and config.cost_per_hour < current_cost
        ]
        
        if not alternatives:
            return current_region
        
        # Return the cheapest alternative
        return min(alternatives, key=lambda x: x[1])[0]


async def main():
    """Demonstrate autonomous production scaling."""
    # Define regions
    regions = [
        RegionConfig("us-east-1", "https://api-us-east.darkoperator.ai", 50, 2.5, 10),
        RegionConfig("eu-west-1", "https://api-eu-west.darkoperator.ai", 30, 2.8, 15),
        RegionConfig("ap-southeast-1", "https://api-ap-se.darkoperator.ai", 20, 2.2, 25),
    ]
    
    # Create scaler
    scaler = AutonomousProductionScaler(regions)
    
    try:
        # Start scaling for demonstration
        await asyncio.wait_for(scaler.start_autonomous_scaling(), timeout=60)
    except asyncio.TimeoutError:
        pass
    finally:
        await scaler.stop_autonomous_scaling()
    
    # Print final status
    status = scaler.get_deployment_status()
    print("📈 Final Deployment Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    asyncio.run(main())