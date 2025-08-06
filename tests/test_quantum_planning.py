"""
Comprehensive test suite for quantum task planning system.

Tests all components including quantum scheduler, adaptive planner,
neural planner, physics optimizer, and security systems.
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from darkoperator.planning.quantum_scheduler import (
    QuantumScheduler, QuantumTask, QuantumTaskGraph, TaskPriority
)
from darkoperator.planning.adaptive_planner import (
    AdaptivePlanner, AdaptiveContext
)
from darkoperator.planning.neural_planner import (
    NeuralTaskPlanner, TaskTransformer, PhysicsInformedPlanner, PlanningContext
)
from darkoperator.planning.physics_optimizer import (
    PhysicsOptimizer, PhysicsConstraints, PhysicsValidator
)
from darkoperator.security.planning_security import (
    PlanningSecurityManager, SecurityLevel, SecurityPolicy, SecurityError
)
from darkoperator.optimization.quantum_optimization import (
    QuantumAnnealer, QuantumOptimizationConfig, QuantumState,
    ParallelQuantumOptimizer
)
from darkoperator.monitoring.quantum_metrics import (
    QuantumMetricsManager, MetricsAggregator, QuantumMetric
)


class TestQuantumScheduler:
    """Test suite for quantum scheduler functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create quantum scheduler instance."""
        return QuantumScheduler(max_workers=2, quantum_annealing_steps=10)
    
    @pytest.fixture
    def sample_tasks(self, scheduler):
        """Create sample tasks for testing."""
        tasks = []
        
        # Simple computation task
        tasks.append(scheduler.submit_task(
            task_id="task_1",
            name="Simple Computation",
            operation=lambda x: x * 2,
            5,
            priority=TaskPriority.EXCITED_1,
            energy_requirement=1.0
        ))
        
        # Physics simulation task
        tasks.append(scheduler.submit_task(
            task_id="task_2", 
            name="Physics Simulation",
            operation=lambda n: {"events": n, "energy": n * 100.0},
            1000,
            priority=TaskPriority.GROUND_STATE,
            energy_requirement=10.0,
            dependencies=["task_1"]
        ))
        
        # Data processing task
        tasks.append(scheduler.submit_task(
            task_id="task_3",
            name="Data Processing", 
            operation=lambda data_size: {"processed": data_size, "time": 0.1},
            data_size=50000,
            priority=TaskPriority.EXCITED_2,
            energy_requirement=5.0
        ))
        
        return tasks
    
    def test_task_submission(self, scheduler):
        """Test task submission and validation."""
        task = scheduler.submit_task(
            task_id="test_task",
            name="Test Task",
            operation=lambda: "completed",
            priority=TaskPriority.EXCITED_1,
            energy_requirement=2.0
        )
        
        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.EXCITED_1
        assert task.energy_requirement == 2.0
        assert task.state == "superposition"
        assert task.is_coherent()
    
    def test_quantum_annealing_schedule(self, scheduler, sample_tasks):
        """Test quantum annealing for task scheduling."""
        scheduled_tasks = scheduler.quantum_anneal_schedule()
        
        # Should return list of tasks
        assert isinstance(scheduled_tasks, list)
        assert len(scheduled_tasks) <= 3  # At most the ready tasks
        
        # All returned tasks should be QuantumTask instances
        for task in scheduled_tasks:
            assert isinstance(task, QuantumTask)
    
    def test_task_dependencies(self, scheduler):
        """Test task dependency handling."""
        # Create tasks with dependencies
        task_a = scheduler.submit_task("task_a", "Task A", lambda: "A")
        task_b = scheduler.submit_task("task_b", "Task B", lambda: "B", dependencies=["task_a"])
        
        # task_a should be ready, task_b should not
        ready_tasks = scheduler.task_graph.get_ready_tasks()
        ready_ids = [t.task_id for t in ready_tasks]
        
        assert "task_a" in ready_ids
        assert "task_b" not in ready_ids
        
        # After completing task_a, task_b should be ready
        task_a.state = "executed"
        ready_tasks = scheduler.task_graph.get_ready_tasks()
        ready_ids = [t.task_id for t in ready_tasks]
        
        assert "task_b" in ready_ids
    
    def test_quantum_interference_pattern(self, scheduler, sample_tasks):
        """Test quantum interference calculations."""
        interference = scheduler.task_graph.compute_interference_pattern()
        
        assert isinstance(interference, dict)
        assert len(interference) > 0
        
        # All values should be non-negative
        for task_id, priority in interference.items():
            assert priority >= 0.0
    
    def test_quantum_execution(self, scheduler, sample_tasks):
        """Test complete quantum execution cycle."""
        results = scheduler.execute_quantum_schedule()
        
        assert 'results' in results
        assert 'statistics' in results
        
        stats = results['statistics']
        assert 'total_tasks' in stats
        assert 'successful_tasks' in stats
        assert 'failed_tasks' in stats
        
        # Should have some successful executions
        assert stats['successful_tasks'] > 0
        assert stats['quantum_efficiency'] > 0.0
    
    def test_quantum_metrics(self, scheduler):
        """Test quantum state metrics."""
        metrics = scheduler.get_quantum_metrics()
        
        assert 'global_phase' in metrics
        assert 'entanglement_entropy' in metrics
        assert 'coherent_tasks' in metrics
        assert 'total_tasks' in metrics
        
        # Global phase should be in [0, 2π]
        assert 0 <= metrics['global_phase'] <= 2 * np.pi


class TestAdaptivePlanner:
    """Test suite for adaptive planner."""
    
    @pytest.fixture
    def planner(self):
        """Create adaptive planner instance."""
        context = AdaptiveContext(
            energy_budget=100.0,
            time_horizon=1800.0,
            cpu_cores=2,
            exploration_rate=0.1
        )
        return AdaptivePlanner(context=context)
    
    @pytest.fixture
    def sample_objectives(self):
        """Create sample planning objectives."""
        return [
            {
                'type': 'physics_simulation',
                'n_events': 10000,
                'computational_cost': 5.0,
                'priority': 'high'
            },
            {
                'type': 'anomaly_detection',
                'data_size': 100000,
                'computational_cost': 3.0,
                'priority': 'critical'
            },
            {
                'type': 'data_processing',
                'data_size': 50000,
                'computational_cost': 2.0,
                'priority': 'medium'
            }
        ]
    
    def test_adaptive_plan_creation(self, planner, sample_objectives):
        """Test adaptive plan creation."""
        plan = planner.create_adaptive_plan(sample_objectives)
        
        assert 'task_schedule' in plan
        assert 'resource_allocation' in plan
        assert 'monitoring_plan' in plan
        assert 'constraints' in plan
        assert 'planning_time' in plan
        
        # Should have generated tasks
        assert len(plan['task_schedule']) > 0
        
        # Planning time should be reasonable
        assert plan['planning_time'] < 10.0  # Should complete quickly for small problems
    
    def test_physics_informed_optimization(self, planner, sample_objectives):
        """Test physics-informed optimization."""
        # Add physics operator to planner
        mock_operator = Mock()
        mock_operator.preserve_energy = True
        mock_operator.preserve_momentum = True
        planner.physics_operator = mock_operator
        
        plan = planner.create_adaptive_plan(sample_objectives)
        
        # Should indicate physics-informed planning
        assert plan['adaptation_metadata']['physics_informed'] == True
    
    def test_adaptive_execution(self, planner, sample_objectives):
        """Test adaptive plan execution."""
        plan = planner.create_adaptive_plan(sample_objectives)
        results = planner.execute_adaptive_plan(plan)
        
        assert 'task_results' in results
        assert 'execution_statistics' in results
        assert 'performance_metrics' in results
        assert 'adaptation_info' in results
        
        # Should have execution statistics
        stats = results['execution_statistics']
        assert 'quantum_efficiency' in stats
        
        # Should update adaptation weights
        adaptation_info = results['adaptation_info']
        assert 'updated_weights' in adaptation_info
        assert 'learning_progress' in adaptation_info
    
    def test_constraint_handling(self, planner, sample_objectives):
        """Test constraint handling in planning."""
        constraints = {
            'max_energy': 50.0,
            'max_execution_time': 300.0,
            'required_accuracy': 0.95
        }
        
        plan = planner.create_adaptive_plan(sample_objectives, constraints)
        
        # Should incorporate additional constraints
        assert plan['constraints']['max_energy'] == 50.0
        assert plan['constraints']['max_execution_time'] == 300.0
    
    def test_learning_adaptation(self, planner):
        """Test learning and adaptation over multiple executions."""
        objectives = [{'type': 'generic', 'priority': 'medium'}]
        
        initial_weights = planner.adaptation_weights.copy()
        
        # Execute multiple plans
        for _ in range(3):
            plan = planner.create_adaptive_plan(objectives)
            planner.execute_adaptive_plan(plan)
        
        # Weights should have been updated
        final_weights = planner.adaptation_weights
        
        # At least one weight should have changed
        assert any(abs(initial_weights[k] - final_weights[k]) > 1e-10 for k in initial_weights)
    
    def test_adaptation_state(self, planner):
        """Test adaptation state reporting."""
        state = planner.get_adaptation_state()
        
        assert 'adaptation_weights' in state
        assert 'task_history_size' in state
        assert 'learning_progress' in state
        assert 'quantum_metrics' in state
        assert 'context' in state
        
        # Should indicate learning status
        assert 'learning_rate' in state['context']
        assert 'exploration_rate' in state['context']


class TestNeuralTaskPlanner:
    """Test suite for neural task planner."""
    
    @pytest.fixture
    def planner(self):
        """Create neural task planner instance."""
        model_config = {
            'transformer': {
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'max_tasks': 10
            }
        }
        return NeuralTaskPlanner(model_config=model_config, device='cpu')
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for neural planning."""
        return [
            {
                'task_id': 'neural_task_1',
                'type': 'physics_simulation',
                'priority': TaskPriority.EXCITED_1,
                'energy_requirement': 5.0,
                'dependencies': []
            },
            {
                'task_id': 'neural_task_2',
                'type': 'anomaly_detection',
                'priority': TaskPriority.GROUND_STATE,
                'energy_requirement': 3.0,
                'dependencies': ['neural_task_1']
            },
            {
                'task_id': 'neural_task_3',
                'type': 'data_processing',
                'priority': TaskPriority.EXCITED_2,
                'energy_requirement': 2.0,
                'dependencies': []
            }
        ]
    
    def test_neural_plan_generation(self, planner, sample_tasks):
        """Test neural plan generation."""
        plan_result = planner.plan_tasks(sample_tasks, optimize=False)
        
        assert 'optimal_schedule' in plan_result
        assert 'resource_allocation' in plan_result
        assert 'physics_constraints' in plan_result
        assert 'performance_metrics' in plan_result
        
        # Should return valid schedule
        schedule = plan_result['optimal_schedule']
        assert isinstance(schedule, list)
        assert len(schedule) <= len(sample_tasks)
    
    def test_neural_optimization(self, planner, sample_tasks):
        """Test neural optimization with detailed analysis."""
        plan_result = planner.plan_tasks(sample_tasks, optimize=True)
        
        assert 'neural_insights' in plan_result
        
        insights = plan_result['neural_insights']
        assert 'priority_distribution' in insights
        assert 'resource_preferences' in insights
        assert 'dependency_strengths' in insights
        assert 'neural_confidence' in insights
    
    def test_task_transformer(self, planner):
        """Test task transformer architecture."""
        transformer = planner.task_transformer
        
        # Test with dummy input
        batch_size = 1
        n_tasks = 3
        feature_dim = 8
        
        task_features = torch.randn(batch_size, n_tasks, feature_dim)
        
        with torch.no_grad():
            outputs = transformer(task_features)
        
        assert 'priorities' in outputs
        assert 'resource_allocation' in outputs
        assert 'dependencies' in outputs
        assert 'schedule_times' in outputs
        
        # Check output shapes
        assert outputs['priorities'].shape == (batch_size, n_tasks)
        assert outputs['resource_allocation'].shape == (batch_size, n_tasks, 3)
    
    def test_physics_informed_planning(self, planner, sample_tasks):
        """Test physics-informed neural planning."""
        # Add physics constraints
        physics_constraints = PhysicsConstraints(
            conserve_energy=True,
            conserve_momentum=True,
            max_energy=100.0
        )
        planner.physics_optimizer.constraints = physics_constraints
        
        plan_result = planner.plan_tasks(sample_tasks, optimize=True)
        
        physics_analysis = plan_result['physics_constraints']
        assert 'energy_conservation_check' in physics_analysis
        assert 'momentum_conservation_check' in physics_analysis
        assert 'physics_score' in physics_analysis
    
    def test_model_training_interface(self, planner):
        """Test neural model training interface."""
        # Mock training data
        training_data = [
            {
                'tasks': [
                    {'task_id': 'train_1', 'type': 'physics', 'priority': TaskPriority.EXCITED_1}
                ],
                'results': {'success_rate': 0.9, 'execution_time': 5.0}
            }
        ]
        
        # Test training (with very few epochs for speed)
        training_results = planner.train_on_execution_data(training_data, n_epochs=5)
        
        assert 'epochs_trained' in training_results
        assert 'final_loss' in training_results
        assert 'training_losses' in training_results
        assert training_results['epochs_trained'] <= 5
    
    def test_model_summary(self, planner):
        """Test model summary generation."""
        summary = planner.get_model_summary()
        
        assert 'model_architecture' in summary
        assert 'training_status' in summary
        assert 'physics_integration' in summary
        assert 'device' in summary
        
        arch = summary['model_architecture']
        assert 'total_parameters' in arch
        assert 'trainable_parameters' in arch
        assert arch['total_parameters'] > 0


class TestPhysicsOptimizer:
    """Test suite for physics optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create physics optimizer instance."""
        constraints = PhysicsConstraints(
            conserve_energy=True,
            conserve_momentum=True,
            max_energy=1000.0,
            conservation_tolerance=1e-6
        )
        return PhysicsOptimizer(constraints)
    
    @pytest.fixture
    def physics_validator(self, optimizer):
        """Create physics validator."""
        return optimizer.validator
    
    def test_4vector_validation(self, physics_validator):
        """Test 4-vector physics validation."""
        # Valid 4-vector (E=5, px=3, py=4, pz=0, so E² = p² + m² with m²=0)
        valid_4vec = torch.tensor([[5.0, 3.0, 4.0, 0.0]])
        assert physics_validator.validate_4vector(valid_4vec) == True
        
        # Invalid 4-vector (tachyonic: E < p)
        invalid_4vec = torch.tensor([[3.0, 5.0, 0.0, 0.0]])
        assert physics_validator.validate_4vector(invalid_4vec) == False
        
        # Negative energy
        negative_energy = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        assert physics_validator.validate_4vector(negative_energy) == False
    
    def test_conservation_validation(self, physics_validator):
        """Test conservation law validation."""
        # Setup initial and final states
        initial_state = torch.tensor([[10.0, 1.0, 0.0, 0.0], [5.0, -1.0, 0.0, 0.0]])  # Two particles
        final_state = torch.tensor([[15.0, 0.0, 0.0, 0.0]])  # One particle
        
        conservation_results = physics_validator.validate_conservation(initial_state, final_state)
        
        assert 'energy' in conservation_results
        assert 'momentum' in conservation_results
        
        # Energy should be conserved (10 + 5 = 15)
        assert conservation_results['energy'] == True
        # Momentum should be conserved (1 - 1 = 0)
        assert conservation_results['momentum'] == True
    
    def test_parameter_optimization(self, optimizer):
        """Test physics-constrained parameter optimization."""
        # Define simple objective function (minimize squared distance from target)
        target_energy = torch.tensor(50.0)
        
        def objective_function(energy_param):
            return torch.square(energy_param - target_energy)
        
        # Initial parameters
        initial_params = {'energy_param': torch.tensor(10.0)}
        
        # Optimize
        results = optimizer.optimize_task_parameters(
            initial_params,
            objective_function,
            physics_constraints=None
        )
        
        assert 'optimized_params' in results
        assert 'final_objective' in results
        assert 'iterations' in results
        assert 'converged' in results
        
        # Should have improved objective
        optimized_energy = results['optimized_params']['energy_param']
        final_objective = results['final_objective']
        assert final_objective < 1600.0  # Much better than initial (10-50)² = 1600
    
    def test_neural_operator_optimization(self, optimizer):
        """Test neural operator optimization with physics constraints."""
        # Create simple neural network
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 1)
            
            def forward(self, x):
                return self.linear(x)
            
            def physics_loss(self, input_4vec, output):
                # Simple physics loss: energy conservation
                input_energy = input_4vec[:, 0].sum()
                output_energy = output.sum()
                return torch.square(input_energy - output_energy)
        
        net = SimpleNet()
        
        # Training data
        training_data = torch.randn(10, 4)  # 10 samples, 4 features (4-vectors)
        target_data = torch.randn(10, 1)    # 10 targets
        
        results = optimizer.optimize_neural_operator(
            net, training_data, target_data, physics_loss_weight=0.1
        )
        
        assert 'final_loss' in results
        assert 'epochs_trained' in results
        assert 'training_log' in results
        assert 'physics_validation' in results
        
        # Should have trained for some epochs
        assert results['epochs_trained'] > 0
        
        # Should have physics validation results
        validation = results['physics_validation']
        assert 'energy_conservation' in validation
        assert 'momentum_conservation' in validation
    
    def test_optimization_summary(self, optimizer):
        """Test optimization summary generation."""
        # Run a simple optimization first
        def dummy_objective(x):
            return torch.square(x)
        
        optimizer.optimize_task_parameters({'x': torch.tensor(5.0)}, dummy_objective)
        
        summary = optimizer.get_optimization_summary()
        
        assert 'total_optimizations' in summary
        assert 'total_iterations' in summary
        assert 'convergence_rate' in summary
        assert 'constraint_violations' in summary
        assert 'physics_constraints' in summary
        
        # Should have recorded the optimization
        assert summary['total_optimizations'] == 1


class TestPlanningSecurityManager:
    """Test suite for planning security manager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager instance."""
        return PlanningSecurityManager(security_level=SecurityLevel.STANDARD)
    
    @pytest.fixture
    def sample_task_data(self):
        """Create sample task data for security testing."""
        return {
            'task_id': 'security_test_task',
            'operation': lambda x: x * 2,
            'args': (5,),
            'kwargs': {},
            'energy_requirement': 1.0
        }
    
    def test_task_validation(self, security_manager, sample_task_data):
        """Test task security validation."""
        secured_task = security_manager.validate_and_secure_task(sample_task_data)
        
        assert 'security_signature' in secured_task
        assert 'security_level' in secured_task
        assert 'validated_timestamp' in secured_task
        
        # Original task data should be preserved
        assert secured_task['task_id'] == sample_task_data['task_id']
        assert secured_task['energy_requirement'] == sample_task_data['energy_requirement']
    
    def test_malicious_operation_detection(self, security_manager):
        """Test detection of malicious operations."""
        # Task with dangerous operation
        malicious_task = {
            'task_id': 'malicious_task',
            'operation': eval,  # Dangerous function
            'args': ('1+1',),
            'kwargs': {}
        }
        
        with pytest.raises(SecurityError):
            security_manager.validate_and_secure_task(malicious_task)
    
    def test_large_argument_detection(self, security_manager):
        """Test detection of excessively large arguments."""
        # Task with large arguments
        large_data = [0] * 1000000  # Large list
        large_task = {
            'task_id': 'large_task',
            'operation': lambda x: len(x),
            'args': (large_data,),
            'kwargs': {}
        }
        
        with pytest.raises(SecurityError):
            security_manager.validate_and_secure_task(large_task)
    
    def test_secure_execution(self, security_manager, sample_task_data):
        """Test secure task execution."""
        secured_task = security_manager.validate_and_secure_task(sample_task_data)
        result = security_manager.execute_secure_task(secured_task)
        
        assert 'result' in result
        assert 'execution_time' in result
        assert 'security_signature' in result
        assert 'security_level' in result
        
        # Should have executed successfully
        assert result['result'] == 10  # 5 * 2
        assert result['execution_time'] > 0
    
    def test_security_status(self, security_manager):
        """Test security status reporting."""
        status = security_manager.get_security_status()
        
        assert 'security_level' in status
        assert 'policy_config' in status
        assert 'validation_stats' in status
        assert 'monitoring_summary' in status
        
        # Should indicate security level
        assert status['security_level'] == SecurityLevel.STANDARD.name
    
    def test_quarantine_functionality(self, security_manager):
        """Test task quarantine functionality."""
        # Create task that should be quarantined
        suspicious_task = {
            'task_id': 'suspicious_task',
            'operation': lambda: __import__('os').system('echo test'),
            'args': (),
            'kwargs': {}
        }
        
        # Should quarantine the task
        with pytest.raises(SecurityError):
            security_manager.validate_and_secure_task(suspicious_task)
        
        # Task should be quarantined
        assert security_manager.monitor.is_task_quarantined('suspicious_task')


class TestQuantumOptimization:
    """Test suite for quantum optimization algorithms."""
    
    @pytest.fixture
    def config(self):
        """Create quantum optimization config."""
        return QuantumOptimizationConfig(
            annealing_steps=50,
            max_qubits=5,
            population_size=10,
            max_generations=20,
            use_gpu_acceleration=False  # Use CPU for tests
        )
    
    @pytest.fixture
    def quantum_annealer(self, config):
        """Create quantum annealer instance."""
        return QuantumAnnealer(config)
    
    @pytest.fixture
    def sample_quantum_tasks(self):
        """Create sample quantum tasks."""
        tasks = []
        for i in range(3):
            task = QuantumTask(
                task_id=f"quantum_task_{i}",
                name=f"Quantum Task {i}",
                operation=lambda: f"result_{i}",
                priority=TaskPriority.EXCITED_1,
                energy_requirement=float(i + 1)
            )
            tasks.append(task)
        return tasks
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization and operations."""
        n_qubits = 3
        quantum_state = QuantumState(n_qubits)
        
        # Should have correct number of amplitudes
        assert quantum_state.amplitudes.shape == (2**n_qubits,)
        
        # Should be normalized
        norm = torch.norm(quantum_state.amplitudes)
        assert abs(norm.item() - 1.0) < 1e-6
        
        # Test rotation
        quantum_state.apply_rotation(0, np.pi/2)
        
        # Test entanglement
        quantum_state.apply_entanglement(0, 1, 0.5)
        
        # Test measurement
        measured_state = quantum_state.measure()
        assert 0 <= measured_state < 2**n_qubits
        
        # Test entropy calculation
        entropy = quantum_state.get_entropy()
        assert entropy >= 0.0
    
    def test_quantum_annealing(self, quantum_annealer, sample_quantum_tasks):
        """Test quantum annealing optimization."""
        results = quantum_annealer.optimize_task_schedule(sample_quantum_tasks)
        
        assert 'optimal_schedule' in results
        assert 'energy' in results
        assert 'optimization_time' in results
        assert 'quantum_entropy' in results
        
        # Should return valid schedule
        schedule = results['optimal_schedule']
        assert isinstance(schedule, list)
        assert len(schedule) <= len(sample_quantum_tasks)
        
        # Should have finite energy
        assert np.isfinite(results['energy'])
        
        # Should complete in reasonable time
        assert results['optimization_time'] < 30.0  # Should be fast for small problems
    
    def test_parallel_quantum_optimization(self, config, sample_quantum_tasks):
        """Test parallel quantum optimization."""
        parallel_optimizer = ParallelQuantumOptimizer(config)
        
        results = parallel_optimizer.optimize_parallel(
            sample_quantum_tasks, 
            constraints=None,
            n_runs=2  # Small number for testing
        )
        
        assert 'optimal_schedule' in results
        assert 'energy' in results
        assert 'parallel_runs' in results
        assert 'total_optimization_time' in results
        
        # Should have completed multiple runs
        assert results['runs_completed'] <= 2
        assert results['runs_completed'] > 0
    
    def test_qubo_matrix_creation(self, quantum_annealer, sample_quantum_tasks):
        """Test QUBO matrix creation for optimization."""
        qubo_matrix = quantum_annealer._create_qubo_matrix(sample_quantum_tasks)
        
        # Should be square matrix
        assert qubo_matrix.shape[0] == qubo_matrix.shape[1]
        assert qubo_matrix.shape[0] > 0
        
        # Should be symmetric for valid QUBO
        assert torch.allclose(qubo_matrix, qubo_matrix.T, atol=1e-6)
    
    def test_solution_energy_computation(self, quantum_annealer, sample_quantum_tasks):
        """Test solution energy computation."""
        # Create simple schedule
        schedule = [task.task_id for task in sample_quantum_tasks]
        
        # Create dummy QUBO matrix
        n_tasks = len(sample_quantum_tasks)
        qubo_matrix = torch.ones(n_tasks, n_tasks)
        
        energy = quantum_annealer._compute_solution_energy(schedule, sample_quantum_tasks, qubo_matrix)
        
        assert np.isfinite(energy)
        assert isinstance(energy, float)


class TestQuantumMetrics:
    """Test suite for quantum metrics system."""
    
    @pytest.fixture
    def metrics_manager(self):
        """Create metrics manager instance."""
        config = {
            'auto_start': False,  # Don't start collection automatically in tests
            'retention_period': 3600,
            'collection_interval': 1.0
        }
        return QuantumMetricsManager(config)
    
    def test_task_execution_recording(self, metrics_manager):
        """Test task execution metrics recording."""
        metrics_manager.record_task_execution(
            task_id="test_task",
            execution_time=1.5,
            energy_consumed=5.0,
            success=True,
            metadata={'algorithm': 'quantum_annealing'}
        )
        
        # Force collection
        metrics_manager.aggregator.collect_all_metrics()
        
        # Should have recorded metrics
        all_metrics = list(metrics_manager.aggregator.metrics_storage)
        assert len(all_metrics) > 0
        
        # Check for specific metrics
        metric_names = [m.name for m in all_metrics]
        assert any('execution_time' in name for name in metric_names)
        assert any('energy' in name for name in metric_names)
        assert any('success_rate' in name for name in metric_names)
    
    def test_conservation_violation_recording(self, metrics_manager):
        """Test conservation violation recording."""
        metrics_manager.record_conservation_violation(
            law_type='energy',
            violation_magnitude=0.01,
            task_id='test_task',
            metadata={'tolerance': 1e-6}
        )
        
        # Force collection
        metrics_manager.aggregator.collect_all_metrics()
        
        # Should have physics metrics
        all_metrics = list(metrics_manager.aggregator.metrics_storage)
        physics_metrics = [m for m in all_metrics if 'physics' in m.name]
        assert len(physics_metrics) > 0
    
    def test_optimization_result_recording(self, metrics_manager):
        """Test optimization result recording."""
        metrics_manager.record_optimization_result(
            algorithm='quantum_annealing',
            optimization_time=5.0,
            final_energy=10.5,
            convergence_steps=100,
            metadata={'temperature_schedule': 'exponential'}
        )
        
        # Force collection
        metrics_manager.aggregator.collect_all_metrics()
        
        # Should have optimization metrics
        all_metrics = list(metrics_manager.aggregator.metrics_storage)
        opt_metrics = [m for m in all_metrics if 'optimization' in m.name]
        assert len(opt_metrics) > 0
    
    def test_performance_dashboard(self, metrics_manager):
        """Test performance dashboard generation."""
        # Record some sample data
        metrics_manager.record_task_execution("task1", 1.0, 2.0, True)
        metrics_manager.record_task_execution("task2", 1.5, 3.0, True)
        
        dashboard = metrics_manager.get_performance_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'system_overview' in dashboard
        assert 'performance_summaries' in dashboard
        assert 'alerts' in dashboard
        
        # Should have system overview
        overview = dashboard['system_overview']
        assert 'total_metrics_stored' in overview
        assert 'collection_running' in overview
    
    def test_metrics_export(self, metrics_manager):
        """Test metrics export functionality."""
        # Add some sample metrics
        sample_metric = QuantumMetric(
            name='test.metric',
            value=42.0,
            timestamp=time.time(),
            unit='test_units',
            tags={'source': 'test'}
        )
        metrics_manager.aggregator.metrics_storage.append(sample_metric)
        
        # Test JSON export
        json_export = metrics_manager.export_metrics(format_type='json')
        assert isinstance(json_export, str)
        assert 'test.metric' in json_export
        
        # Test CSV export  
        csv_export = metrics_manager.export_metrics(format_type='csv')
        assert isinstance(csv_export, str)
        assert 'test.metric' in csv_export
    
    def test_metrics_summary(self, metrics_manager):
        """Test metrics summary generation."""
        # Add sample metrics
        current_time = time.time()
        for i in range(10):
            metric = QuantumMetric(
                name='test.summary.metric',
                value=float(i),
                timestamp=current_time - i,
                unit='units'
            )
            metrics_manager.aggregator.metrics_storage.append(metric)
        
        summary = metrics_manager.get_metrics_summary(
            'test.summary.metric',
            time_range=(current_time - 20, current_time)
        )
        
        assert summary is not None
        assert summary.metric_name == 'test.summary.metric'
        assert summary.count == 10
        assert summary.mean == 4.5  # Mean of 0-9
        assert summary.min_value == 0.0
        assert summary.max_value == 9.0


# Integration test
class TestIntegration:
    """Integration tests for the complete quantum planning system."""
    
    def test_end_to_end_quantum_planning(self):
        """Test complete end-to-end quantum planning workflow."""
        # Initialize components
        scheduler = QuantumScheduler(max_workers=2, quantum_annealing_steps=20)
        neural_planner = NeuralTaskPlanner(device='cpu')
        metrics_manager = QuantumMetricsManager({'auto_start': False})
        
        # Create sample workflow
        tasks = [
            {
                'task_id': 'e2e_task_1',
                'type': 'physics_simulation',
                'priority': TaskPriority.EXCITED_1,
                'energy_requirement': 3.0,
                'operation': lambda n: {'events': n, 'energy': n * 50.0},
                'args': (1000,)
            },
            {
                'task_id': 'e2e_task_2',
                'type': 'anomaly_detection',
                'priority': TaskPriority.GROUND_STATE,
                'energy_requirement': 2.0,
                'operation': lambda size: {'anomalies': max(1, size // 10000), 'processed': size},
                'args': (50000,)
            }
        ]
        
        # Neural planning
        neural_plan = neural_planner.plan_tasks(tasks, optimize=False)
        assert len(neural_plan['optimal_schedule']) > 0
        
        # Submit to quantum scheduler
        quantum_tasks = []
        for task in tasks:
            qt = scheduler.submit_task(
                task_id=task['task_id'],
                name=task['task_id'],
                operation=task['operation'],
                *task['args'],
                priority=task['priority'],
                energy_requirement=task['energy_requirement']
            )
            quantum_tasks.append(qt)
        
        # Execute
        execution_results = scheduler.execute_quantum_schedule()
        
        # Record metrics
        for task_id, result in execution_results['results'].items():
            success = not isinstance(result, str)  # String results indicate errors
            metrics_manager.record_task_execution(
                task_id=task_id,
                execution_time=1.0,  # Mock execution time
                energy_consumed=2.0,  # Mock energy
                success=success
            )
        
        # Verify results
        assert execution_results['statistics']['total_tasks'] > 0
        assert execution_results['statistics']['successful_tasks'] > 0
        
        # Verify metrics
        metrics_manager.aggregator.collect_all_metrics()
        all_metrics = list(metrics_manager.aggregator.metrics_storage)
        assert len(all_metrics) > 0
        
        # Cleanup
        scheduler.shutdown()
        metrics_manager.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])