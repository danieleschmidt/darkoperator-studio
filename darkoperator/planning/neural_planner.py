"""
Neural task planner with deep learning optimization.

Combines transformer architectures with physics-informed neural networks
for intelligent task planning and adaptive scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
import json

from ..models.fno import FourierNeuralOperator
from .quantum_scheduler import QuantumTask, TaskPriority
from .physics_optimizer import PhysicsOptimizer, PhysicsConstraints


logger = logging.getLogger(__name__)


@dataclass
class PlanningContext:
    """Context embedding for neural planning."""
    
    # Task characteristics
    task_types: List[str]
    priority_levels: List[int]
    resource_requirements: List[float]
    
    # Physics context
    energy_scales: List[float]
    momentum_constraints: Optional[torch.Tensor] = None
    conservation_requirements: List[bool] = None
    
    # Temporal context
    deadlines: List[float]
    execution_history: Optional[List[Dict]] = None
    
    # System context
    available_resources: Dict[str, float] = None
    system_load: float = 0.5
    
    def to_tensor(self) -> torch.Tensor:
        """Convert context to tensor representation."""
        # Basic encoding of context features
        features = []
        
        # Task type encoding (one-hot or embedding)
        task_type_map = {'physics': 0, 'anomaly': 1, 'data': 2, 'generic': 3}
        type_encoding = torch.zeros(len(self.task_types), len(task_type_map))
        for i, task_type in enumerate(self.task_types):
            type_idx = task_type_map.get(task_type.split('_')[0], 3)
            type_encoding[i, type_idx] = 1.0
        
        features.append(type_encoding.flatten())
        
        # Priority and resource features
        features.append(torch.tensor(self.priority_levels, dtype=torch.float32))
        features.append(torch.tensor(self.resource_requirements, dtype=torch.float32))
        features.append(torch.tensor(self.energy_scales, dtype=torch.float32))
        features.append(torch.tensor(self.deadlines, dtype=torch.float32))
        
        # System features
        if self.available_resources:
            features.append(torch.tensor(list(self.available_resources.values()), dtype=torch.float32))
        features.append(torch.tensor([self.system_load], dtype=torch.float32))
        
        return torch.cat([f.flatten() for f in features])


class TaskTransformer(nn.Module):
    """
    Transformer architecture for task planning.
    
    Uses attention mechanisms to capture task dependencies and optimize
    scheduling decisions based on context.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_tasks: int = 100,
        n_task_types: int = 10
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_tasks = max_tasks
        
        # Task embedding layers
        self.task_type_embedding = nn.Embedding(n_task_types, d_model // 4)
        self.priority_embedding = nn.Embedding(5, d_model // 4)  # 5 priority levels
        self.resource_projection = nn.Linear(1, d_model // 4)
        self.temporal_projection = nn.Linear(1, d_model // 4)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_tasks, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.priority_head = nn.Linear(d_model, 1)  # Priority scoring
        self.resource_head = nn.Linear(d_model, 3)  # Resource allocation (CPU, GPU, Memory)
        self.dependency_head = nn.Linear(d_model, max_tasks)  # Dependency attention
        self.schedule_head = nn.Linear(d_model, 1)  # Scheduling timestamp
        
        # Physics-informed layer
        self.physics_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, task_features: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through task transformer.
        
        Args:
            task_features: Task feature tensor (batch, n_tasks, features)
            context_mask: Attention mask (batch, n_tasks)
            
        Returns:
            Dictionary of planning outputs
        """
        batch_size, n_tasks, _ = task_features.shape
        
        # Task embeddings
        task_types = task_features[:, :, 0].long()  # Assume first feature is task type
        priorities = task_features[:, :, 1].long()   # Second feature is priority
        resources = task_features[:, :, 2:3]         # Resource requirements
        temporal = task_features[:, :, 3:4]          # Temporal features
        
        # Create embeddings
        type_emb = self.task_type_embedding(task_types)
        priority_emb = self.priority_embedding(priorities)
        resource_emb = self.resource_projection(resources)
        temporal_emb = self.temporal_projection(temporal)
        
        # Combine embeddings
        task_embeddings = torch.cat([type_emb, priority_emb, resource_emb, temporal_emb], dim=-1)
        
        # Add positional encoding
        task_embeddings = task_embeddings + self.pos_encoding[:n_tasks].unsqueeze(0)
        
        # Apply transformer
        if context_mask is not None:
            # Convert mask for transformer (True = ignore)
            attention_mask = ~context_mask.bool()
        else:
            attention_mask = None
            
        encoded_tasks = self.transformer_encoder(task_embeddings, src_key_padding_mask=attention_mask)
        
        # Apply physics-informed processing
        physics_enhanced = self.physics_layer(encoded_tasks) + encoded_tasks  # Residual connection
        
        # Generate outputs
        outputs = {
            'priorities': self.priority_head(physics_enhanced).squeeze(-1),
            'resource_allocation': F.softmax(self.resource_head(physics_enhanced), dim=-1),
            'dependencies': F.softmax(self.dependency_head(physics_enhanced), dim=-1),
            'schedule_times': F.sigmoid(self.schedule_head(physics_enhanced)).squeeze(-1),
            'task_embeddings': physics_enhanced
        }
        
        return outputs


class PhysicsInformedPlanner(nn.Module):
    """
    Physics-informed neural planner combining FNO with task transformers.
    
    Learns optimal task scheduling policies while respecting physics constraints.
    """
    
    def __init__(
        self,
        task_transformer: TaskTransformer,
        physics_optimizer: Optional[PhysicsOptimizer] = None,
        fno_operator: Optional[FourierNeuralOperator] = None
    ):
        super().__init__()
        
        self.task_transformer = task_transformer
        self.physics_optimizer = physics_optimizer
        self.fno_operator = fno_operator
        
        # Planning policy network
        self.policy_network = nn.Sequential(
            nn.Linear(task_transformer.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Policy score
        )
        
        # Value function for reinforcement learning
        self.value_network = nn.Sequential(
            nn.Linear(task_transformer.d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(), 
            nn.Linear(128, 1)  # Value estimate
        )
        
        # Physics constraint predictor
        self.constraint_predictor = nn.Sequential(
            nn.Linear(task_transformer.d_model * 2, 128),  # Pairwise task features
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4)  # Energy, momentum, conservation, symmetry constraints
        )
    
    def forward(self, planning_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for planning optimization.
        
        Args:
            planning_input: Dictionary with task features and context
            
        Returns:
            Planning decisions and constraint predictions
        """
        task_features = planning_input['task_features']
        context_mask = planning_input.get('context_mask')
        
        # Transform tasks with attention
        transformer_outputs = self.task_transformer(task_features, context_mask)
        task_embeddings = transformer_outputs['task_embeddings']
        
        # Generate policy scores
        policy_scores = self.policy_network(task_embeddings)
        value_estimates = self.value_network(task_embeddings)
        
        # Predict physics constraints between task pairs
        batch_size, n_tasks, d_model = task_embeddings.shape
        constraint_predictions = torch.zeros(batch_size, n_tasks, n_tasks, 4)
        
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                pair_features = torch.cat([task_embeddings[:, i], task_embeddings[:, j]], dim=-1)
                constraint_predictions[:, i, j] = self.constraint_predictor(pair_features)
                constraint_predictions[:, j, i] = constraint_predictions[:, i, j]  # Symmetric
        
        # Combine with transformer outputs
        planning_outputs = {
            **transformer_outputs,
            'policy_scores': policy_scores.squeeze(-1),
            'value_estimates': value_estimates.squeeze(-1),
            'constraint_predictions': constraint_predictions
        }
        
        return planning_outputs
    
    def predict_optimal_schedule(
        self, 
        tasks: List[Dict[str, Any]], 
        context: Optional[PlanningContext] = None
    ) -> List[int]:
        """
        Predict optimal task execution schedule.
        
        Args:
            tasks: List of task dictionaries
            context: Planning context
            
        Returns:
            List of task indices in optimal execution order
        """
        self.eval()
        
        with torch.no_grad():
            # Convert tasks to tensor format
            task_features = self._tasks_to_tensor(tasks)
            
            if task_features.shape[1] == 0:  # No tasks
                return []
            
            # Add batch dimension
            task_features = task_features.unsqueeze(0)
            
            # Forward pass
            planning_input = {
                'task_features': task_features,
                'context_mask': None
            }
            
            outputs = self.forward(planning_input)
            
            # Extract scheduling information
            priorities = outputs['priorities'].squeeze(0)  # Remove batch dim
            schedule_times = outputs['schedule_times'].squeeze(0)
            policy_scores = outputs['policy_scores'].squeeze(0)
            
            # Combine scores for final ranking
            combined_scores = priorities + policy_scores + (1.0 - schedule_times)  # Earlier = better
            
            # Sort tasks by combined score (descending)
            sorted_indices = torch.argsort(combined_scores, descending=True)
            
            return sorted_indices.tolist()
    
    def _tasks_to_tensor(self, tasks: List[Dict[str, Any]]) -> torch.Tensor:
        """Convert task list to tensor representation."""
        if not tasks:
            return torch.zeros(0, 8)  # Empty tensor with correct feature size
        
        task_type_map = {'physics_simulation': 0, 'anomaly_detection': 1, 'data_processing': 2, 'generic': 3}
        priority_map = {
            TaskPriority.GROUND_STATE: 0,
            TaskPriority.EXCITED_1: 1,
            TaskPriority.EXCITED_2: 2, 
            TaskPriority.EXCITED_3: 3,
            TaskPriority.METASTABLE: 4
        }
        
        features = []
        
        for task in tasks:
            task_features = torch.zeros(8)
            
            # Task type encoding
            task_type = task.get('type', 'generic')
            task_features[0] = task_type_map.get(task_type, 3)
            
            # Priority encoding
            priority = task.get('priority', TaskPriority.EXCITED_2)
            if isinstance(priority, TaskPriority):
                task_features[1] = priority.value
            else:
                task_features[1] = priority_map.get(priority, 2)
            
            # Resource requirements
            task_features[2] = task.get('energy_requirement', 1.0)
            task_features[3] = task.get('computational_cost', 1.0)
            
            # Temporal features
            task_features[4] = task.get('deadline', 3600.0) / 3600.0  # Normalize to hours
            task_features[5] = len(task.get('dependencies', []))  # Dependency count
            
            # Physics features
            momentum = task.get('momentum', np.zeros(3))
            if hasattr(momentum, '__len__') and len(momentum) >= 3:
                task_features[6] = float(np.linalg.norm(momentum))
            
            # Context features
            task_features[7] = task.get('superposition_weight', 1.0)
            
            features.append(task_features)
        
        return torch.stack(features)


class NeuralTaskPlanner:
    """
    Main neural task planner integrating all components.
    
    Provides high-level interface for physics-informed task planning
    with neural optimization and quantum-inspired scheduling.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        physics_constraints: Optional[PhysicsConstraints] = None,
        device: str = 'auto'
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        model_config = model_config or {}
        self.task_transformer = TaskTransformer(**model_config.get('transformer', {}))
        
        self.physics_optimizer = PhysicsOptimizer(physics_constraints)
        
        self.neural_planner = PhysicsInformedPlanner(
            self.task_transformer,
            self.physics_optimizer
        )
        
        # Move to device
        self.neural_planner.to(self.device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.neural_planner.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        # Training history
        self.training_history = []
        self.best_performance = float('inf')
        self.best_state = None
        
        logger.info(f"Initialized neural task planner on device: {self.device}")
    
    def plan_tasks(
        self, 
        tasks: List[Dict[str, Any]], 
        context: Optional[PlanningContext] = None,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate optimal task plan using neural optimization.
        
        Args:
            tasks: List of task specifications
            context: Planning context and constraints
            optimize: Whether to perform neural optimization
            
        Returns:
            Comprehensive planning results
        """
        start_time = time.time()
        
        if not tasks:
            return {
                'optimal_schedule': [],
                'resource_allocation': {},
                'physics_constraints': {},
                'neural_insights': {},
                'planning_time': 0.0
            }
        
        # Generate optimal schedule
        optimal_schedule = self.neural_planner.predict_optimal_schedule(tasks, context)
        
        # Detailed analysis if optimization enabled
        detailed_analysis = {}
        if optimize:
            detailed_analysis = self._perform_detailed_analysis(tasks, optimal_schedule, context)
        
        # Physics constraint analysis
        physics_analysis = self._analyze_physics_constraints(tasks, optimal_schedule)
        
        # Resource allocation optimization
        resource_allocation = self._optimize_resource_allocation(tasks, optimal_schedule)
        
        planning_time = time.time() - start_time
        
        results = {
            'optimal_schedule': optimal_schedule,
            'scheduled_tasks': [tasks[i] for i in optimal_schedule],
            'resource_allocation': resource_allocation,
            'physics_constraints': physics_analysis,
            'neural_insights': detailed_analysis,
            'planning_time': planning_time,
            'performance_metrics': {
                'total_tasks': len(tasks),
                'physics_compliant': physics_analysis.get('compliant_tasks', 0),
                'resource_efficiency': resource_allocation.get('efficiency_score', 0.0)
            }
        }
        
        logger.info(f"Neural planning completed: {len(tasks)} tasks scheduled in {planning_time:.3f}s")
        return results
    
    def _perform_detailed_analysis(
        self, 
        tasks: List[Dict[str, Any]], 
        schedule: List[int],
        context: Optional[PlanningContext]
    ) -> Dict[str, Any]:
        """Perform detailed neural analysis of the planning decision."""
        
        self.neural_planner.eval()
        
        with torch.no_grad():
            task_features = self.neural_planner._tasks_to_tensor(tasks).unsqueeze(0).to(self.device)
            
            planning_input = {
                'task_features': task_features,
                'context_mask': None
            }
            
            outputs = self.neural_planner.forward(planning_input)
            
            # Extract insights
            insights = {
                'attention_patterns': self._analyze_attention_patterns(outputs),
                'priority_distribution': outputs['priorities'].cpu().numpy().tolist(),
                'resource_preferences': outputs['resource_allocation'].cpu().numpy().tolist(),
                'dependency_strengths': self._analyze_dependencies(outputs['dependencies']),
                'constraint_predictions': outputs['constraint_predictions'].cpu().numpy(),
                'neural_confidence': torch.softmax(outputs['policy_scores'], dim=0).cpu().numpy().tolist()
            }
        
        return insights
    
    def _analyze_attention_patterns(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze attention patterns in the transformer."""
        # This would extract attention weights from transformer layers
        # For now, return placeholder analysis
        return {
            'high_attention_pairs': [],
            'attention_entropy': 0.0,
            'focus_distribution': []
        }
    
    def _analyze_dependencies(self, dependency_matrix: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Analyze task dependency strengths."""
        dependencies = []
        matrix = dependency_matrix.squeeze(0).cpu().numpy()
        
        n_tasks = matrix.shape[0]
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i != j and matrix[i, j] > 0.1:  # Threshold for significant dependency
                    dependencies.append((i, j, float(matrix[i, j])))
        
        return sorted(dependencies, key=lambda x: x[2], reverse=True)[:10]  # Top 10
    
    def _analyze_physics_constraints(
        self, 
        tasks: List[Dict[str, Any]], 
        schedule: List[int]
    ) -> Dict[str, Any]:
        """Analyze physics constraint compliance."""
        
        analysis = {
            'energy_conservation_check': True,
            'momentum_conservation_check': True,
            'symmetry_preservation': True,
            'compliant_tasks': len(tasks),
            'constraint_violations': [],
            'physics_score': 1.0
        }
        
        # Check each task for physics compliance
        total_energy = 0.0
        total_momentum = np.zeros(3)
        
        for task_idx in schedule:
            task = tasks[task_idx]
            
            # Energy accumulation
            energy_req = task.get('energy_requirement', 1.0)
            total_energy += energy_req
            
            # Momentum accumulation
            momentum = task.get('momentum', np.zeros(3))
            if hasattr(momentum, '__len__') and len(momentum) >= 3:
                total_momentum += np.array(momentum)
        
        # Check conservation limits
        if total_energy > self.physics_optimizer.constraints.max_energy:
            analysis['energy_conservation_check'] = False
            analysis['constraint_violations'].append('Total energy exceeds limit')
        
        momentum_magnitude = np.linalg.norm(total_momentum)
        if momentum_magnitude > self.physics_optimizer.constraints.max_momentum:
            analysis['momentum_conservation_check'] = False
            analysis['constraint_violations'].append('Total momentum exceeds limit')
        
        # Update compliance metrics
        violations = len(analysis['constraint_violations'])
        analysis['compliant_tasks'] = max(0, len(tasks) - violations)
        analysis['physics_score'] = max(0.0, 1.0 - violations / len(tasks))
        
        return analysis
    
    def _optimize_resource_allocation(
        self, 
        tasks: List[Dict[str, Any]], 
        schedule: List[int]
    ) -> Dict[str, Any]:
        """Optimize resource allocation for scheduled tasks."""
        
        total_cpu = 0.0
        total_memory = 0.0
        total_energy = 0.0
        
        allocation = {}
        
        for i, task_idx in enumerate(schedule):
            task = tasks[task_idx]
            task_id = task.get('task_id', f'task_{task_idx}')
            
            # Simple allocation strategy
            cpu_req = task.get('energy_requirement', 1.0) * 0.5
            memory_req = task.get('computational_cost', 1.0) * 0.25
            energy_req = task.get('energy_requirement', 1.0)
            
            allocation[task_id] = {
                'cpu_cores': min(cpu_req, 2.0),
                'memory_gb': min(memory_req, 4.0),
                'energy_budget': energy_req,
                'execution_order': i
            }
            
            total_cpu += cpu_req
            total_memory += memory_req
            total_energy += energy_req
        
        # Calculate efficiency metrics
        max_available = {'cpu': 8.0, 'memory': 16.0, 'energy': 1000.0}
        efficiency_score = 1.0 - max(
            total_cpu / max_available['cpu'],
            total_memory / max_available['memory'],
            total_energy / max_available['energy']
        )
        
        return {
            'task_allocations': allocation,
            'total_resources': {
                'cpu_cores': total_cpu,
                'memory_gb': total_memory,
                'energy_budget': total_energy
            },
            'efficiency_score': max(0.0, efficiency_score)
        }
    
    def train_on_execution_data(
        self, 
        execution_data: List[Dict[str, Any]], 
        n_epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Train neural planner on historical execution data.
        
        Args:
            execution_data: List of execution records with tasks, schedules, and outcomes
            n_epochs: Number of training epochs
            
        Returns:
            Training results and performance metrics
        """
        
        if not execution_data:
            logger.warning("No training data provided")
            return {'message': 'No training data available'}
        
        self.neural_planner.train()
        
        training_losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_data in self._create_training_batches(execution_data):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.neural_planner.forward(batch_data['inputs'])
                
                # Compute loss
                loss = self._compute_training_loss(outputs, batch_data['targets'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.neural_planner.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Scheduler step
            self.scheduler.step()
            
            # Record epoch loss
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            
            # Save best model
            if avg_loss < self.best_performance:
                self.best_performance = avg_loss
                self.best_state = self.neural_planner.state_dict().copy()
            
            # Early stopping check
            if epoch > 20 and avg_loss > np.mean(training_losses[-10:]) * 1.05:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_state is not None:
            self.neural_planner.load_state_dict(self.best_state)
        
        training_results = {
            'epochs_trained': len(training_losses),
            'final_loss': training_losses[-1] if training_losses else float('inf'),
            'best_loss': self.best_performance,
            'training_losses': training_losses,
            'convergence_achieved': len(training_losses) < n_epochs
        }
        
        self.training_history.append(training_results)
        
        logger.info(f"Neural planner training completed: {len(training_losses)} epochs, "
                   f"best loss: {self.best_performance:.6f}")
        
        return training_results
    
    def _create_training_batches(self, execution_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create training batches from execution data."""
        # This is a simplified implementation
        # In practice, would create proper batches with padding, etc.
        
        batches = []
        
        for data in execution_data:
            tasks = data.get('tasks', [])
            if not tasks:
                continue
                
            task_features = self.neural_planner._tasks_to_tensor(tasks).unsqueeze(0).to(self.device)
            
            # Create targets from actual execution results
            targets = self._create_training_targets(data)
            
            batches.append({
                'inputs': {
                    'task_features': task_features,
                    'context_mask': None
                },
                'targets': targets
            })
        
        return batches
    
    def _create_training_targets(self, execution_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create training targets from execution results."""
        # Extract ground truth from execution data
        # This would be based on actual performance metrics
        
        tasks = execution_data.get('tasks', [])
        results = execution_data.get('results', {})
        
        # Create dummy targets - in practice would use real performance data
        n_tasks = len(tasks)
        
        targets = {
            'priorities': torch.ones(n_tasks).to(self.device),
            'resource_allocation': torch.ones(n_tasks, 3).to(self.device) / 3.0,
            'schedule_times': torch.linspace(0, 1, n_tasks).to(self.device),
            'policy_scores': torch.ones(n_tasks).to(self.device)
        }
        
        return targets
    
    def _compute_training_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute training loss."""
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Priority prediction loss
        if 'priorities' in outputs and 'priorities' in targets:
            priority_loss = F.mse_loss(outputs['priorities'], targets['priorities'])
            total_loss += priority_loss
        
        # Resource allocation loss
        if 'resource_allocation' in outputs and 'resource_allocation' in targets:
            resource_loss = F.mse_loss(outputs['resource_allocation'], targets['resource_allocation'])
            total_loss += resource_loss
        
        # Schedule timing loss
        if 'schedule_times' in outputs and 'schedule_times' in targets:
            schedule_loss = F.mse_loss(outputs['schedule_times'], targets['schedule_times'])
            total_loss += schedule_loss
        
        # Policy score loss
        if 'policy_scores' in outputs and 'policy_scores' in targets:
            policy_loss = F.mse_loss(outputs['policy_scores'], targets['policy_scores'])
            total_loss += policy_loss * 0.5  # Lower weight for policy loss
        
        return total_loss
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        checkpoint = {
            'model_state_dict': self.neural_planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_performance': self.best_performance,
            'device': str(self.device)
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.neural_planner.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.best_performance = checkpoint.get('best_performance', float('inf'))
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of neural planner model."""
        
        total_params = sum(p.numel() for p in self.neural_planner.parameters())
        trainable_params = sum(p.numel() for p in self.neural_planner.parameters() if p.requires_grad)
        
        return {
            'model_architecture': {
                'transformer_layers': self.task_transformer.transformer_encoder.num_layers,
                'attention_heads': 8,  # Default value
                'd_model': self.task_transformer.d_model,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'training_status': {
                'epochs_completed': sum(r['epochs_trained'] for r in self.training_history),
                'best_performance': self.best_performance,
                'training_runs': len(self.training_history)
            },
            'physics_integration': {
                'physics_optimizer': self.physics_optimizer is not None,
                'constraint_validation': True,
                'conservation_laws': ['energy', 'momentum']
            },
            'device': str(self.device)
        }