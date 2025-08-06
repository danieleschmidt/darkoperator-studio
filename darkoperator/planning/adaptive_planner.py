"""
Adaptive task planner that learns from physics simulation patterns.

Integrates with neural operators to dynamically optimize task execution
based on real-time physics constraints and anomaly detection requirements.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
import logging
from dataclasses import dataclass
from collections import defaultdict, deque

from ..operators.base import PhysicsOperator
from ..anomaly.base import BaseAnomalyDetector
from .quantum_scheduler import QuantumScheduler, QuantumTask, TaskPriority


logger = logging.getLogger(__name__)


@dataclass
class AdaptiveContext:
    """Context information for adaptive planning decisions."""
    
    # Physics constraints
    energy_budget: float = 100.0
    momentum_constraint: np.ndarray = None
    time_horizon: float = 3600.0  # Planning horizon in seconds
    
    # Anomaly detection context
    anomaly_threshold: float = 1e-6
    detection_confidence: float = 0.95
    background_rate: float = 1000.0  # Events per second
    
    # Resource constraints  
    cpu_cores: int = 4
    gpu_memory: float = 8.0  # GB
    network_bandwidth: float = 1000.0  # MB/s
    
    # Learning parameters
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    memory_size: int = 10000


class AdaptivePlanner:
    """
    Physics-informed adaptive task planner.
    
    Uses reinforcement learning principles to optimize task scheduling
    based on physics simulation requirements and anomaly detection goals.
    """
    
    def __init__(
        self,
        physics_operator: Optional[PhysicsOperator] = None,
        anomaly_detector: Optional[BaseAnomalyDetector] = None,
        context: Optional[AdaptiveContext] = None
    ):
        self.physics_operator = physics_operator
        self.anomaly_detector = anomaly_detector
        self.context = context or AdaptiveContext()
        
        # Quantum scheduler for task execution
        self.quantum_scheduler = QuantumScheduler(
            max_workers=self.context.cpu_cores,
            physics_operator=physics_operator
        )
        
        # Adaptive learning components
        self.task_history: deque = deque(maxlen=self.context.memory_size)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_weights: Dict[str, float] = {
            'physics_conservation': 1.0,
            'anomaly_sensitivity': 1.0,
            'computational_efficiency': 1.0,
            'resource_utilization': 0.5
        }
        
        # Neural network for plan optimization (simple feedforward)
        self.plan_optimizer = self._create_plan_optimizer()
        
        logger.info("Initialized adaptive planner with physics-informed scheduling")
    
    def _create_plan_optimizer(self) -> torch.nn.Module:
        """Create neural network for plan optimization."""
        
        class PlanOptimizer(torch.nn.Module):
            def __init__(self, input_dim=20, hidden_dim=64, output_dim=10):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, output_dim),
                    torch.nn.Softmax(dim=-1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        optimizer_net = PlanOptimizer()
        return optimizer_net
    
    def create_adaptive_plan(
        self,
        objectives: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create adaptive execution plan based on objectives and constraints.
        
        Args:
            objectives: List of planning objectives with priorities
            constraints: Additional constraints beyond context
            
        Returns:
            Adaptive plan with task schedule and resource allocation
        """
        
        start_time = time.time()
        
        # Merge constraints with context
        effective_constraints = self._merge_constraints(constraints)
        
        # Generate candidate tasks from objectives
        candidate_tasks = self._generate_tasks_from_objectives(objectives)
        
        # Apply physics-informed optimization
        optimized_tasks = self._apply_physics_optimization(candidate_tasks)
        
        # Create quantum task schedule
        task_schedule = self._create_quantum_schedule(optimized_tasks)
        
        # Allocate resources adaptively
        resource_allocation = self._adaptive_resource_allocation(task_schedule)
        
        # Generate monitoring strategy
        monitoring_plan = self._create_monitoring_plan(task_schedule)
        
        planning_time = time.time() - start_time
        
        adaptive_plan = {
            'task_schedule': task_schedule,
            'resource_allocation': resource_allocation,
            'monitoring_plan': monitoring_plan,
            'constraints': effective_constraints,
            'planning_time': planning_time,
            'adaptation_metadata': {
                'exploration_rate': self.context.exploration_rate,
                'weights': self.adaptation_weights.copy(),
                'physics_informed': self.physics_operator is not None
            }
        }
        
        logger.info(f"Generated adaptive plan with {len(task_schedule)} tasks in {planning_time:.3f}s")
        return adaptive_plan
    
    def _merge_constraints(self, additional_constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge context constraints with additional constraints."""
        constraints = {
            'energy_budget': self.context.energy_budget,
            'time_horizon': self.context.time_horizon,
            'cpu_cores': self.context.cpu_cores,
            'gpu_memory': self.context.gpu_memory,
            'anomaly_threshold': self.context.anomaly_threshold
        }
        
        if additional_constraints:
            constraints.update(additional_constraints)
        
        return constraints
    
    def _generate_tasks_from_objectives(self, objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate candidate tasks from high-level objectives."""
        tasks = []
        
        for i, objective in enumerate(objectives):
            obj_type = objective.get('type', 'generic')
            obj_priority = objective.get('priority', 'medium')
            
            if obj_type == 'physics_simulation':
                tasks.extend(self._generate_physics_tasks(objective, i))
            elif obj_type == 'anomaly_detection':
                tasks.extend(self._generate_anomaly_tasks(objective, i))
            elif obj_type == 'data_processing':
                tasks.extend(self._generate_data_tasks(objective, i))
            else:
                tasks.append(self._generate_generic_task(objective, i))
        
        return tasks
    
    def _generate_physics_tasks(self, objective: Dict[str, Any], obj_id: int) -> List[Dict[str, Any]]:
        """Generate physics simulation tasks."""
        tasks = []
        
        # Main simulation task
        tasks.append({
            'task_id': f'physics_sim_{obj_id}',
            'name': f"Physics Simulation {obj_id}",
            'type': 'physics_simulation',
            'operation': self._dummy_physics_simulation,
            'args': (objective.get('n_events', 10000),),
            'priority': TaskPriority.EXCITED_1,
            'energy_requirement': objective.get('computational_cost', 10.0),
            'dependencies': []
        })
        
        # Conservation law validation
        if self.physics_operator:
            tasks.append({
                'task_id': f'conservation_check_{obj_id}',
                'name': f"Conservation Law Check {obj_id}",
                'type': 'validation',
                'operation': self._check_conservation_laws,
                'args': (),
                'priority': TaskPriority.EXCITED_2,
                'energy_requirement': 1.0,
                'dependencies': [f'physics_sim_{obj_id}']
            })
        
        return tasks
    
    def _generate_anomaly_tasks(self, objective: Dict[str, Any], obj_id: int) -> List[Dict[str, Any]]:
        """Generate anomaly detection tasks."""
        tasks = []
        
        # Anomaly scanning task
        tasks.append({
            'task_id': f'anomaly_scan_{obj_id}',
            'name': f"Anomaly Detection {obj_id}",
            'type': 'anomaly_detection',
            'operation': self._dummy_anomaly_detection,
            'args': (objective.get('data_size', 1000000),),
            'priority': TaskPriority.GROUND_STATE,  # Highest priority
            'energy_requirement': objective.get('computational_cost', 5.0),
            'dependencies': []
        })
        
        # Statistical analysis
        tasks.append({
            'task_id': f'statistical_analysis_{obj_id}',
            'name': f"Statistical Analysis {obj_id}",
            'type': 'analysis',
            'operation': self._statistical_analysis,
            'args': (),
            'priority': TaskPriority.EXCITED_2,
            'energy_requirement': 2.0,
            'dependencies': [f'anomaly_scan_{obj_id}']
        })
        
        return tasks
    
    def _generate_data_tasks(self, objective: Dict[str, Any], obj_id: int) -> List[Dict[str, Any]]:
        """Generate data processing tasks."""
        return [{
            'task_id': f'data_proc_{obj_id}',
            'name': f"Data Processing {obj_id}",
            'type': 'data_processing',
            'operation': self._dummy_data_processing,
            'args': (objective.get('data_size', 100000),),
            'priority': TaskPriority.EXCITED_3,
            'energy_requirement': objective.get('computational_cost', 2.0),
            'dependencies': []
        }]
    
    def _generate_generic_task(self, objective: Dict[str, Any], obj_id: int) -> Dict[str, Any]:
        """Generate generic task."""
        return {
            'task_id': f'generic_{obj_id}',
            'name': f"Generic Task {obj_id}",
            'type': 'generic',
            'operation': lambda: f"Completed objective {obj_id}",
            'args': (),
            'priority': TaskPriority.EXCITED_2,
            'energy_requirement': 1.0,
            'dependencies': []
        }
    
    def _apply_physics_optimization(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply physics-informed optimization to tasks."""
        if not self.physics_operator:
            return tasks
        
        # Optimize task order based on physics constraints
        optimized_tasks = []
        
        for task in tasks:
            # Apply conservation law constraints
            if 'energy_requirement' in task:
                # Scale energy based on physics accuracy requirements
                conservation_factor = 1.0
                if hasattr(self.physics_operator, 'preserve_energy') and self.physics_operator.preserve_energy:
                    conservation_factor *= 1.2  # Extra cost for energy conservation
                if hasattr(self.physics_operator, 'preserve_momentum') and self.physics_operator.preserve_momentum:
                    conservation_factor *= 1.1  # Extra cost for momentum conservation
                
                task['energy_requirement'] *= conservation_factor
            
            # Add momentum vector based on task type
            if task['type'] == 'physics_simulation':
                task['momentum'] = np.array([1.0, 0.0, 0.0])  # Forward momentum
            elif task['type'] == 'anomaly_detection':
                task['momentum'] = np.array([0.0, 1.0, 0.0])  # Perpendicular search
            else:
                task['momentum'] = np.array([0.0, 0.0, 0.0])  # No preferred direction
            
            optimized_tasks.append(task)
        
        return optimized_tasks
    
    def _create_quantum_schedule(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Create quantum-optimized task schedule."""
        # Submit tasks to quantum scheduler
        for task in tasks:
            self.quantum_scheduler.submit_task(
                task_id=task['task_id'],
                name=task['name'],
                operation=task['operation'],
                *task.get('args', ()),
                priority=task.get('priority', TaskPriority.EXCITED_2),
                dependencies=task.get('dependencies', []),
                energy_requirement=task.get('energy_requirement', 1.0),
                **task.get('kwargs', {})
            )
        
        # Get optimal schedule
        scheduled_tasks = self.quantum_scheduler.quantum_anneal_schedule()
        return [task.task_id for task in scheduled_tasks]
    
    def _adaptive_resource_allocation(self, task_schedule: List[str]) -> Dict[str, Any]:
        """Allocate resources adaptively based on task requirements."""
        allocation = {
            'cpu_allocation': {},
            'gpu_allocation': {},
            'memory_allocation': {},
            'network_allocation': {}
        }
        
        # Simple allocation strategy (could be more sophisticated)
        total_tasks = len(task_schedule)
        if total_tasks > 0:
            cpu_per_task = self.context.cpu_cores / min(total_tasks, self.context.cpu_cores)
            memory_per_task = self.context.gpu_memory / total_tasks
            bandwidth_per_task = self.context.network_bandwidth / total_tasks
            
            for task_id in task_schedule:
                allocation['cpu_allocation'][task_id] = cpu_per_task
                allocation['memory_allocation'][task_id] = memory_per_task
                allocation['network_allocation'][task_id] = bandwidth_per_task
        
        return allocation
    
    def _create_monitoring_plan(self, task_schedule: List[str]) -> Dict[str, Any]:
        """Create monitoring plan for adaptive execution."""
        return {
            'metrics_to_track': [
                'execution_time',
                'energy_consumption',
                'physics_conservation_error',
                'anomaly_detection_rate',
                'resource_utilization'
            ],
            'monitoring_frequency': 1.0,  # seconds
            'adaptation_triggers': [
                'physics_constraint_violation',
                'anomaly_threshold_exceeded',
                'resource_exhaustion'
            ],
            'feedback_loop': True
        }
    
    def execute_adaptive_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptive plan with real-time learning and adaptation."""
        execution_start = time.time()
        
        # Execute using quantum scheduler
        results = self.quantum_scheduler.execute_quantum_schedule()
        
        # Collect performance metrics
        execution_time = time.time() - execution_start
        metrics = self._collect_execution_metrics(results, execution_time)
        
        # Update adaptation weights based on performance
        self._update_adaptation_weights(metrics)
        
        # Store in history for learning
        execution_record = {
            'plan': plan,
            'results': results,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.task_history.append(execution_record)
        
        # Prepare final results
        final_results = {
            'task_results': results['results'],
            'execution_statistics': results['statistics'],
            'performance_metrics': metrics,
            'adaptation_info': {
                'updated_weights': self.adaptation_weights.copy(),
                'learning_progress': self._compute_learning_progress()
            }
        }
        
        logger.info(f"Completed adaptive plan execution in {execution_time:.3f}s")
        return final_results
    
    def _collect_execution_metrics(self, results: Dict[str, Any], execution_time: float) -> Dict[str, float]:
        """Collect performance metrics from execution."""
        stats = results['statistics']
        
        metrics = {
            'total_execution_time': execution_time,
            'task_success_rate': stats.get('quantum_efficiency', 0.0),
            'average_task_time': execution_time / max(stats.get('total_tasks', 1), 1),
            'physics_conservation_score': 0.95,  # Placeholder - would compute from actual physics
            'anomaly_detection_efficiency': 0.90,  # Placeholder
            'resource_utilization': 0.85  # Placeholder
        }
        
        return metrics
    
    def _update_adaptation_weights(self, metrics: Dict[str, float]) -> None:
        """Update adaptation weights based on performance metrics."""
        learning_rate = self.context.learning_rate
        
        # Update based on performance feedback
        if metrics['physics_conservation_score'] < 0.9:
            self.adaptation_weights['physics_conservation'] += learning_rate
        
        if metrics['anomaly_detection_efficiency'] < 0.95:
            self.adaptation_weights['anomaly_sensitivity'] += learning_rate
        
        if metrics['resource_utilization'] > 0.95:
            self.adaptation_weights['computational_efficiency'] += learning_rate
        
        # Normalize weights
        total_weight = sum(self.adaptation_weights.values())
        for key in self.adaptation_weights:
            self.adaptation_weights[key] /= total_weight
    
    def _compute_learning_progress(self) -> float:
        """Compute learning progress metric."""
        if len(self.task_history) < 2:
            return 0.0
        
        recent_performance = [
            record['metrics']['task_success_rate'] 
            for record in list(self.task_history)[-5:]
        ]
        
        if len(recent_performance) >= 2:
            return np.mean(recent_performance[-3:]) - np.mean(recent_performance[:2])
        
        return 0.0
    
    # Dummy task implementations for demonstration
    def _dummy_physics_simulation(self, n_events: int) -> Dict[str, Any]:
        """Dummy physics simulation."""
        time.sleep(0.1)  # Simulate computation
        return {'n_events': n_events, 'energy_total': n_events * 100.0, 'status': 'completed'}
    
    def _dummy_anomaly_detection(self, data_size: int) -> Dict[str, Any]:
        """Dummy anomaly detection."""
        time.sleep(0.05)
        n_anomalies = max(1, int(data_size * 1e-6))  # Rare events
        return {'data_size': data_size, 'anomalies_found': n_anomalies, 'status': 'completed'}
    
    def _dummy_data_processing(self, data_size: int) -> Dict[str, Any]:
        """Dummy data processing."""
        time.sleep(0.02)
        return {'processed_size': data_size, 'status': 'completed'}
    
    def _check_conservation_laws(self) -> Dict[str, Any]:
        """Check physics conservation laws."""
        time.sleep(0.01)
        return {'energy_conserved': True, 'momentum_conserved': True, 'status': 'validated'}
    
    def _statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis."""
        time.sleep(0.03)
        return {'significance': 3.2, 'p_value': 1e-3, 'confidence': 0.997, 'status': 'analyzed'}
    
    def get_adaptation_state(self) -> Dict[str, Any]:
        """Get current adaptation state and learning metrics."""
        return {
            'adaptation_weights': self.adaptation_weights.copy(),
            'task_history_size': len(self.task_history),
            'learning_progress': self._compute_learning_progress(),
            'quantum_metrics': self.quantum_scheduler.get_quantum_metrics(),
            'context': {
                'exploration_rate': self.context.exploration_rate,
                'learning_rate': self.context.learning_rate,
                'physics_informed': self.physics_operator is not None
            }
        }