"""
Intelligent Auto-scaling for DarkOperator Studio

Implements smart horizontal and vertical scaling with predictive algorithms,
resource optimization, and adaptive load balancing for distributed physics computations.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import math
import warnings


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    throughput_ops_per_sec: float
    response_time_p95_ms: float
    error_rate: float
    timestamp: float
    

@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    action: str  # "scale_up", "scale_down", "no_action"
    target_instances: int
    current_instances: int
    confidence: float
    reasoning: List[str]
    estimated_impact: Dict[str, float]
    timestamp: float


@dataclass
class ResourceNode:
    """Represents a compute resource node."""
    node_id: str
    node_type: str  # "cpu", "gpu", "high_memory"
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    status: str  # "active", "idle", "overloaded", "failed"
    current_load: float
    specialization: List[str]  # ["neural_operators", "anomaly_detection", etc.]
    cost_per_hour: float
    startup_time_sec: float
    

class PredictiveLoadForecaster:
    """Predicts future load patterns for proactive scaling."""
    
    def __init__(self, forecast_horizon_minutes: int = 30):
        self.forecast_horizon_minutes = forecast_horizon_minutes
        self.load_history = deque(maxlen=1000)  # Store 1000 data points
        self.seasonal_patterns = {}
        self.trend_analysis = {"slope": 0.0, "confidence": 0.0}
        
    def record_load_metrics(self, metrics: ScalingMetrics):
        """Record load metrics for pattern learning."""
        load_point = {
            'timestamp': metrics.timestamp,
            'cpu_utilization': metrics.cpu_utilization,
            'memory_utilization': metrics.memory_utilization,
            'queue_length': metrics.queue_length,
            'throughput': metrics.throughput_ops_per_sec,
            'hour_of_day': time.localtime(metrics.timestamp).tm_hour,
            'day_of_week': time.localtime(metrics.timestamp).tm_wday,
            'composite_load': self._calculate_composite_load(metrics)
        }
        
        self.load_history.append(load_point)
        
        # Update patterns periodically
        if len(self.load_history) >= 50:
            self._update_patterns()
            
    def _calculate_composite_load(self, metrics: ScalingMetrics) -> float:
        """Calculate composite load score."""
        # Weighted combination of metrics
        cpu_weight = 0.4
        memory_weight = 0.3
        queue_weight = 0.2
        error_weight = 0.1
        
        # Normalize queue length (assume max reasonable queue is 1000)
        normalized_queue = min(metrics.queue_length / 1000.0, 1.0)
        
        composite = (
            cpu_weight * metrics.cpu_utilization +
            memory_weight * metrics.memory_utilization +
            queue_weight * normalized_queue +
            error_weight * metrics.error_rate
        )
        
        return min(composite, 1.0)  # Cap at 1.0
        
    def _update_patterns(self):
        """Update seasonal patterns and trend analysis."""
        if len(self.load_history) < 50:
            return
            
        recent_data = list(self.load_history)
        
        # Analyze hourly patterns
        hourly_loads = defaultdict(list)
        for point in recent_data:
            hourly_loads[point['hour_of_day']].append(point['composite_load'])
            
        # Calculate average load per hour
        self.seasonal_patterns['hourly'] = {}
        for hour, loads in hourly_loads.items():
            self.seasonal_patterns['hourly'][hour] = sum(loads) / len(loads)
            
        # Simple trend analysis using recent points
        if len(recent_data) >= 20:
            recent_loads = [p['composite_load'] for p in recent_data[-20:]]
            recent_times = [p['timestamp'] for p in recent_data[-20:]]
            
            # Simple linear regression for trend
            if len(set(recent_times)) > 1:  # Avoid division by zero
                time_range = max(recent_times) - min(recent_times)
                if time_range > 0:
                    load_change = recent_loads[-1] - recent_loads[0]
                    slope = load_change / time_range
                    
                    self.trend_analysis = {
                        'slope': slope,
                        'confidence': min(len(recent_data) / 100.0, 1.0)
                    }
                    
    def forecast_load(self, minutes_ahead: int) -> Dict[str, Any]:
        """Forecast load for specified minutes ahead."""
        if not self.load_history:
            return {
                'predicted_load': 0.5,
                'confidence': 0.0,
                'components': {'seasonal': 0.5, 'trend': 0.0}
            }
            
        current_time = time.time()
        target_time = current_time + (minutes_ahead * 60)
        target_hour = time.localtime(target_time).tm_hour
        
        # Get seasonal component
        seasonal_load = self.seasonal_patterns.get('hourly', {}).get(target_hour, 0.5)
        
        # Get trend component
        trend_component = self.trend_analysis['slope'] * (minutes_ahead * 60)
        trend_component = max(-0.5, min(0.5, trend_component))  # Limit trend impact
        
        # Combine components
        predicted_load = seasonal_load + trend_component
        predicted_load = max(0.0, min(1.0, predicted_load))  # Keep in valid range
        
        # Calculate confidence based on data availability and consistency
        base_confidence = min(len(self.load_history) / 200.0, 0.8)
        trend_confidence = self.trend_analysis['confidence']
        combined_confidence = (base_confidence + trend_confidence) / 2.0
        
        return {
            'predicted_load': predicted_load,
            'confidence': combined_confidence,
            'components': {
                'seasonal': seasonal_load,
                'trend': trend_component,
                'current': self.load_history[-1]['composite_load'] if self.load_history else 0.5
            },
            'forecast_horizon_minutes': minutes_ahead
        }


class ResourcePool:
    """Manages pool of available compute resources."""
    
    def __init__(self):
        self.nodes: Dict[str, ResourceNode] = {}
        self.node_capabilities = {}
        self.allocation_strategy = "workload_aware"  # "round_robin", "least_loaded", "workload_aware"
        
    def add_node(self, node: ResourceNode):
        """Add a node to the resource pool."""
        self.nodes[node.node_id] = node
        self._analyze_node_capabilities(node)
        
    def remove_node(self, node_id: str):
        """Remove a node from the resource pool."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            
    def _analyze_node_capabilities(self, node: ResourceNode):
        """Analyze and categorize node capabilities."""
        capabilities = []
        
        if node.gpu_count > 0:
            capabilities.append("gpu_compute")
            if node.gpu_count >= 4:
                capabilities.append("high_performance_gpu")
                
        if node.memory_gb > 32:
            capabilities.append("high_memory")
            
        if node.cpu_cores > 16:
            capabilities.append("high_cpu")
            
        # Physics-specific capabilities based on specialization
        for spec in node.specialization:
            if "neural" in spec.lower():
                capabilities.append("ml_optimized")
            if "physics" in spec.lower():
                capabilities.append("physics_compute")
                
        self.node_capabilities[node.node_id] = capabilities
        
    def find_best_nodes(self, workload_type: str, required_instances: int, 
                       requirements: Optional[Dict[str, Any]] = None) -> List[ResourceNode]:
        """Find best nodes for a specific workload."""
        requirements = requirements or {}
        
        # Filter nodes based on requirements
        suitable_nodes = []
        for node in self.nodes.values():
            if node.status not in ["active", "idle"]:
                continue
                
            # Check basic requirements
            if requirements.get("min_memory_gb", 0) > node.memory_gb:
                continue
            if requirements.get("min_cpu_cores", 0) > node.cpu_cores:
                continue
            if requirements.get("requires_gpu", False) and node.gpu_count == 0:
                continue
                
            suitable_nodes.append(node)
            
        # Score nodes based on suitability for workload
        scored_nodes = []
        for node in suitable_nodes:
            score = self._score_node_for_workload(node, workload_type)
            scored_nodes.append((score, node))
            
        # Sort by score (highest first) and return top nodes
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        return [node for _, node in scored_nodes[:required_instances]]
        
    def _score_node_for_workload(self, node: ResourceNode, workload_type: str) -> float:
        """Score how suitable a node is for a specific workload."""
        base_score = 0.5
        
        # Load-based scoring (prefer less loaded nodes)
        load_penalty = node.current_load * 0.3
        base_score -= load_penalty
        
        # Specialization bonus
        if workload_type in node.specialization:
            base_score += 0.3
            
        # Capability matching
        capabilities = self.node_capabilities.get(node.node_id, [])
        
        if workload_type in ["neural_operator_inference", "anomaly_detection"]:
            if "ml_optimized" in capabilities:
                base_score += 0.2
            if "gpu_compute" in capabilities:
                base_score += 0.15
                
        elif workload_type in ["physics_simulation", "calorimeter_simulation"]:
            if "physics_compute" in capabilities:
                base_score += 0.2
            if "high_cpu" in capabilities:
                base_score += 0.1
                
        # Cost efficiency (prefer cheaper nodes for similar capability)
        if node.cost_per_hour < 1.0:  # Assume $1/hour as baseline
            base_score += 0.1
        elif node.cost_per_hour > 3.0:
            base_score -= 0.1
            
        return max(0.0, min(1.0, base_score))
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource pool status."""
        if not self.nodes:
            return {'total_nodes': 0}
            
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for n in self.nodes.values() if n.status == "active")
        total_cpu_cores = sum(n.cpu_cores for n in self.nodes.values())
        total_memory_gb = sum(n.memory_gb for n in self.nodes.values())
        total_gpus = sum(n.gpu_count for n in self.nodes.values())
        avg_load = sum(n.current_load for n in self.nodes.values()) / total_nodes
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'total_cpu_cores': total_cpu_cores,
            'total_memory_gb': total_memory_gb,
            'total_gpus': total_gpus,
            'average_load': avg_load,
            'node_types': list(set(n.node_type for n in self.nodes.values())),
            'capabilities_available': list(set().union(*self.node_capabilities.values()))
        }


class IntelligentAutoscaler:
    """Main intelligent auto-scaling controller."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 100):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Components
        self.load_forecaster = PredictiveLoadForecaster()
        self.resource_pool = ResourcePool()
        
        # Scaling parameters
        self.scale_up_threshold = 0.7  # 70% load
        self.scale_down_threshold = 0.3  # 30% load
        self.scale_up_cooldown_minutes = 5
        self.scale_down_cooldown_minutes = 10
        
        # State tracking
        self.last_scaling_action = None
        self.last_scaling_time = 0
        self.scaling_history = deque(maxlen=100)
        
        # Advanced features
        self.predictive_scaling_enabled = True
        self.cost_optimization_enabled = True
        
    def evaluate_scaling_decision(self, current_metrics: ScalingMetrics) -> ScalingDecision:
        """Evaluate whether scaling action is needed."""
        
        # Record metrics for learning
        self.load_forecaster.record_load_metrics(current_metrics)
        
        # Check cooldown periods
        time_since_last_scaling = time.time() - self.last_scaling_time
        
        # Get load forecast for predictive scaling
        forecast = self.load_forecaster.forecast_load(15)  # 15 minutes ahead
        predicted_load = forecast['predicted_load']
        forecast_confidence = forecast['confidence']
        
        # Current load assessment
        current_load = self.load_forecaster._calculate_composite_load(current_metrics)
        
        # Decision logic
        reasoning = []
        action = "no_action"
        target_instances = self.current_instances
        confidence = 0.5
        
        # Scale up conditions
        scale_up_needed = False
        
        if current_load > self.scale_up_threshold:
            scale_up_needed = True
            reasoning.append(f"Current load {current_load:.2f} exceeds threshold {self.scale_up_threshold}")
            
        if (self.predictive_scaling_enabled and 
            predicted_load > self.scale_up_threshold and 
            forecast_confidence > 0.5):
            scale_up_needed = True
            reasoning.append(f"Predicted load {predicted_load:.2f} exceeds threshold")
            
        if (current_metrics.queue_length > 100 and 
            current_metrics.response_time_p95_ms > 1000):
            scale_up_needed = True
            reasoning.append("High queue length and response time")
            
        # Scale down conditions
        scale_down_needed = False
        
        if (current_load < self.scale_down_threshold and 
            predicted_load < self.scale_down_threshold and
            current_metrics.queue_length < 10):
            scale_down_needed = True
            reasoning.append(f"Current and predicted load below threshold {self.scale_down_threshold}")
            
        # Apply cooldown logic
        if scale_up_needed:
            if time_since_last_scaling < self.scale_up_cooldown_minutes * 60:
                reasoning.append("Scale-up cooldown period active")
                scale_up_needed = False
                
        if scale_down_needed:
            if time_since_last_scaling < self.scale_down_cooldown_minutes * 60:
                reasoning.append("Scale-down cooldown period active")
                scale_down_needed = False
                
        # Determine action and target
        if scale_up_needed and self.current_instances < self.max_instances:
            action = "scale_up"
            # Intelligent scale-up size
            if current_load > 0.9 or current_metrics.queue_length > 500:
                scale_factor = 2.0  # Aggressive scaling
            else:
                scale_factor = 1.5  # Conservative scaling
                
            target_instances = min(
                self.max_instances,
                int(self.current_instances * scale_factor)
            )
            confidence = 0.8 if current_load > 0.8 else 0.6
            
        elif scale_down_needed and self.current_instances > self.min_instances:
            action = "scale_down"
            # Conservative scale-down
            target_instances = max(
                self.min_instances,
                int(self.current_instances * 0.7)
            )
            confidence = 0.7
            
        # Cost optimization considerations
        if self.cost_optimization_enabled and action == "scale_up":
            # Check if we can handle load with different instance types
            current_hour = time.localtime().tm_hour
            if 18 <= current_hour <= 6:  # Off-peak hours
                reasoning.append("Off-peak scaling optimization applied")
                target_instances = min(target_instances, self.current_instances + 1)
                
        # Estimate impact
        estimated_impact = self._estimate_scaling_impact(
            current_metrics, action, target_instances
        )
        
        decision = ScalingDecision(
            action=action,
            target_instances=target_instances,
            current_instances=self.current_instances,
            confidence=confidence,
            reasoning=reasoning,
            estimated_impact=estimated_impact,
            timestamp=time.time()
        )
        
        return decision
        
    def execute_scaling_decision(self, decision: ScalingDecision, 
                               workload_type: str = "mixed") -> Dict[str, Any]:
        """Execute a scaling decision."""
        
        if decision.action == "no_action":
            return {"status": "no_action_taken", "reason": "No scaling needed"}
            
        # Record scaling decision
        self.scaling_history.append(decision)
        self.last_scaling_action = decision.action
        self.last_scaling_time = time.time()
        
        execution_result = {"status": "success", "details": {}}
        
        if decision.action == "scale_up":
            # Find additional nodes needed
            additional_instances = decision.target_instances - self.current_instances
            
            best_nodes = self.resource_pool.find_best_nodes(
                workload_type, additional_instances
            )
            
            if len(best_nodes) >= additional_instances:
                # Activate nodes
                for node in best_nodes[:additional_instances]:
                    node.status = "active"
                    
                self.current_instances = decision.target_instances
                execution_result["details"]["nodes_activated"] = [n.node_id for n in best_nodes[:additional_instances]]
                
            else:
                # Partial scaling if not enough nodes available
                available_nodes = len(best_nodes)
                self.current_instances += available_nodes
                execution_result["status"] = "partial_success"
                execution_result["details"]["requested"] = additional_instances
                execution_result["details"]["activated"] = available_nodes
                
        elif decision.action == "scale_down":
            # Find nodes to deactivate (prefer least loaded)
            active_nodes = [n for n in self.resource_pool.nodes.values() if n.status == "active"]
            active_nodes.sort(key=lambda x: x.current_load)
            
            instances_to_remove = self.current_instances - decision.target_instances
            nodes_to_deactivate = active_nodes[:instances_to_remove]
            
            for node in nodes_to_deactivate:
                node.status = "idle"
                
            self.current_instances = decision.target_instances
            execution_result["details"]["nodes_deactivated"] = [n.node_id for n in nodes_to_deactivate]
            
        return execution_result
        
    def _estimate_scaling_impact(self, current_metrics: ScalingMetrics, 
                               action: str, target_instances: int) -> Dict[str, float]:
        """Estimate impact of scaling action."""
        
        if action == "no_action":
            return {"load_change": 0.0, "cost_change": 0.0, "response_time_change": 0.0}
            
        instance_change = target_instances - self.current_instances
        
        # Load impact (simplified model)
        if instance_change > 0:
            load_reduction = min(0.3, instance_change * 0.1)  # Each instance reduces load by ~10%
            response_time_improvement = -min(200, instance_change * 50)  # ms improvement
        else:
            load_increase = min(0.2, abs(instance_change) * 0.15)
            response_time_degradation = min(300, abs(instance_change) * 75)  # ms degradation
            load_reduction = -load_increase
            response_time_improvement = response_time_degradation
            
        # Cost impact (assume $1/hour per instance baseline)
        cost_change = instance_change * 1.0  # $1/hour change
        
        return {
            "load_change": load_reduction,
            "cost_change": cost_change,
            "response_time_change": response_time_improvement
        }
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        
        resource_summary = self.resource_pool.get_resource_summary()
        
        # Recent scaling history
        recent_actions = list(self.scaling_history)[-10:] if self.scaling_history else []
        
        # Performance metrics
        if self.scaling_history:
            recent_decisions = list(self.scaling_history)[-20:]
            avg_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)
            action_distribution = defaultdict(int)
            for decision in recent_decisions:
                action_distribution[decision.action] += 1
        else:
            avg_confidence = 0.0
            action_distribution = {}
            
        return {
            "timestamp": time.time(),
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "last_scaling_action": self.last_scaling_action,
            "time_since_last_scaling_minutes": (time.time() - self.last_scaling_time) / 60,
            "scaling_thresholds": {
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold
            },
            "resource_pool": resource_summary,
            "recent_decisions": len(recent_actions),
            "decision_confidence_avg": avg_confidence,
            "action_distribution": dict(action_distribution),
            "features_enabled": {
                "predictive_scaling": self.predictive_scaling_enabled,
                "cost_optimization": self.cost_optimization_enabled
            }
        }


# Global autoscaler instance
_global_autoscaler = None

def get_autoscaler() -> IntelligentAutoscaler:
    """Get or create global autoscaler."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = IntelligentAutoscaler()
    return _global_autoscaler


def create_sample_resource_pool() -> ResourcePool:
    """Create a sample resource pool for testing."""
    pool = ResourcePool()
    
    # Add various node types
    nodes = [
        ResourceNode("node-cpu-1", "cpu", 8, 32, 0, "idle", 0.2, ["physics_simulation"], 0.8, 60),
        ResourceNode("node-gpu-1", "gpu", 16, 64, 2, "idle", 0.1, ["neural_operators", "anomaly_detection"], 2.5, 120),
        ResourceNode("node-gpu-2", "gpu", 16, 64, 4, "idle", 0.0, ["neural_operators"], 3.2, 120),
        ResourceNode("node-mem-1", "high_memory", 12, 128, 0, "idle", 0.3, ["data_processing"], 1.5, 45),
        ResourceNode("node-hybrid-1", "hybrid", 20, 96, 1, "active", 0.6, ["physics_simulation", "neural_operators"], 2.0, 90)
    ]
    
    for node in nodes:
        pool.add_node(node)
        
    return pool


if __name__ == "__main__":
    # Example usage and testing
    print("🔄 DarkOperator Intelligent Auto-scaling System")
    
    # Create autoscaler with sample resource pool
    autoscaler = IntelligentAutoscaler(min_instances=2, max_instances=20)
    autoscaler.resource_pool = create_sample_resource_pool()
    
    # Simulate scaling scenarios
    test_scenarios = [
        ScalingMetrics(0.8, 0.6, 150, 450, 800, 0.02, time.time()),  # High load
        ScalingMetrics(0.9, 0.8, 300, 200, 1500, 0.05, time.time() + 300),  # Very high load
        ScalingMetrics(0.3, 0.2, 20, 800, 200, 0.01, time.time() + 600),  # Low load
        ScalingMetrics(0.5, 0.4, 80, 600, 400, 0.015, time.time() + 900),  # Medium load
    ]
    
    print("\n🧪 Testing scaling decisions:")
    for i, metrics in enumerate(test_scenarios):
        decision = autoscaler.evaluate_scaling_decision(metrics)
        
        print(f"\nScenario {i+1}:")
        print(f"  Load: {autoscaler.load_forecaster._calculate_composite_load(metrics):.2f}")
        print(f"  Decision: {decision.action}")
        print(f"  Target instances: {decision.target_instances}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning[0] if decision.reasoning else 'No specific reason'}")
        
        # Execute decision
        if decision.action != "no_action":
            result = autoscaler.execute_scaling_decision(decision, "neural_operator_inference")
            print(f"  Execution: {result['status']}")
            
        # Small delay between scenarios
        time.sleep(0.1)
    
    # Get final status
    status = autoscaler.get_scaling_status()
    print(f"\n📊 Final Scaling Status:")
    print(f"Current instances: {status['current_instances']}")
    print(f"Resource pool nodes: {status['resource_pool']['total_nodes']}")
    print(f"Average decision confidence: {status['decision_confidence_avg']:.2f}")
    print(f"Recent decisions: {status['recent_decisions']}")
    
    print("✅ Intelligent auto-scaling test completed!")