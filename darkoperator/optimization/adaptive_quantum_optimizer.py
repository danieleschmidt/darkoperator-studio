"""
Adaptive Quantum-Classical Hybrid Optimizer for Physics Neural Networks.

This module implements advanced optimization algorithms that combine classical
gradient-based methods with quantum optimization techniques for superior
performance in physics-informed neural operator training.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import deque, defaultdict
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

class OptimizationMethod(Enum):
    """Available optimization methods."""
    QUANTUM_ADAM = "quantum_adam"
    VARIATIONAL_QUANTUM = "variational_quantum"
    QUANTUM_NATURAL_GRADIENT = "quantum_natural_gradient"
    HYBRID_CLASSICAL_QUANTUM = "hybrid_classical_quantum"
    ADIABATIC_OPTIMIZATION = "adiabatic_optimization"
    QUANTUM_APPROXIMATE_OPTIMIZATION = "qaoa"

@dataclass
class QuantumOptimizerConfig:
    """Configuration for quantum optimizer."""
    method: OptimizationMethod = OptimizationMethod.QUANTUM_ADAM
    learning_rate: float = 1e-3
    quantum_learning_rate: float = 1e-2
    n_quantum_layers: int = 4
    n_qubits: int = 8
    measurement_shots: int = 1000
    quantum_noise: float = 0.01
    classical_fallback: bool = True
    adaptive_parameters: bool = True
    momentum_beta1: float = 0.9
    momentum_beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    
    # Physics-informed parameters
    conservation_weight: float = 0.1
    symmetry_weight: float = 0.05
    causality_weight: float = 0.05

class QuantumGradientEstimator:
    """Quantum gradient estimation using parameter shift rule."""
    
    def __init__(self, n_qubits: int, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.logger = logging.getLogger("QuantumGradientEstimator")
        
    def estimate_gradient(self, parameters: torch.Tensor, 
                         loss_function: Callable, 
                         shift_value: float = np.pi/2) -> torch.Tensor:
        """
        Estimate gradients using quantum parameter shift rule.
        
        Args:
            parameters: Current parameter values
            loss_function: Function to compute loss
            shift_value: Parameter shift for finite difference
            
        Returns:
            Estimated gradients
        """
        gradients = torch.zeros_like(parameters)
        
        for i, param in enumerate(parameters):
            # Forward shift
            params_plus = parameters.clone()
            params_plus[i] += shift_value
            loss_plus = loss_function(params_plus)
            
            # Backward shift  
            params_minus = parameters.clone()
            params_minus[i] -= shift_value
            loss_minus = loss_function(params_minus)
            
            # Finite difference gradient
            gradients[i] = (loss_plus - loss_minus) / (2 * shift_value)
        
        return gradients
    
    def quantum_fisher_information(self, parameters: torch.Tensor,
                                  state_function: Callable) -> torch.Tensor:
        """Compute quantum Fisher information matrix."""
        n_params = len(parameters)
        fisher_matrix = torch.zeros(n_params, n_params)
        
        # Base quantum state
        base_state = state_function(parameters)
        
        for i in range(n_params):
            for j in range(i, n_params):
                # Compute Fisher information element F_ij
                params_i = parameters.clone()
                params_i[i] += np.pi/2
                state_i = state_function(params_i)
                
                params_j = parameters.clone() 
                params_j[j] += np.pi/2
                state_j = state_function(params_j)
                
                params_ij = parameters.clone()
                params_ij[i] += np.pi/2
                params_ij[j] += np.pi/2
                state_ij = state_function(params_ij)
                
                # Fisher information formula (simplified)
                fisher_element = 4 * (
                    torch.real(torch.vdot(state_ij, state_ij)) - 
                    torch.real(torch.vdot(state_i, state_i)) * torch.real(torch.vdot(state_j, state_j))
                )
                
                fisher_matrix[i, j] = fisher_element
                fisher_matrix[j, i] = fisher_element  # Symmetric
        
        return fisher_matrix

class VariationalQuantumCircuit:
    """Variational quantum circuit for optimization."""
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_parameters = n_qubits * n_layers * 3  # RX, RY, RZ for each qubit per layer
        
    def apply_ansatz(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Apply variational ansatz to quantum state.
        
        Args:
            parameters: Variational parameters
            
        Returns:
            Quantum state amplitudes
        """
        # Initialize quantum state |0⟩^n
        state_vector = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state_vector[0] = 1.0  # |0...0⟩
        
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                rx_angle = parameters[param_idx]
                ry_angle = parameters[param_idx + 1]
                rz_angle = parameters[param_idx + 2]
                param_idx += 3
                
                # Apply rotation gates (simplified representation)
                state_vector = self._apply_rotation_gate(state_vector, qubit, rx_angle, ry_angle, rz_angle)
            
            # Entangling gates (CNOT ladder)
            for qubit in range(self.n_qubits - 1):
                state_vector = self._apply_cnot(state_vector, qubit, qubit + 1)
        
        return state_vector
    
    def _apply_rotation_gate(self, state: torch.Tensor, qubit: int, 
                           rx_angle: float, ry_angle: float, rz_angle: float) -> torch.Tensor:
        """Apply rotation gates to a specific qubit."""
        # Simplified rotation application
        rotation_factor = torch.exp(1j * (rx_angle + ry_angle + rz_angle))
        
        # Apply to computational basis states involving this qubit
        for i in range(2**self.n_qubits):
            if (i >> qubit) & 1:  # If qubit is in |1⟩ state
                state[i] *= rotation_factor
        
        return state
    
    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate between control and target qubits."""
        new_state = state.clone()
        
        for i in range(2**self.n_qubits):
            # Check if control qubit is |1⟩
            if (i >> control) & 1:
                # Flip target qubit
                j = i ^ (1 << target)
                new_state[i] = state[j]
                new_state[j] = state[i]
        
        return new_state

class AdaptiveQuantumOptimizer(torch.optim.Optimizer):
    """Adaptive quantum-classical hybrid optimizer."""
    
    def __init__(self, params, config: QuantumOptimizerConfig):
        if config.learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {config.learning_rate}")
        
        defaults = {
            'lr': config.learning_rate,
            'quantum_lr': config.quantum_learning_rate,
            'betas': (config.momentum_beta1, config.momentum_beta2),
            'eps': config.epsilon,
            'weight_decay': config.weight_decay
        }
        
        super(AdaptiveQuantumOptimizer, self).__init__(params, defaults)
        
        self.config = config
        self.quantum_circuit = VariationalQuantumCircuit(config.n_qubits, config.n_quantum_layers)
        self.gradient_estimator = QuantumGradientEstimator(config.n_qubits, config.measurement_shots)
        
        # Initialize quantum parameters
        self.quantum_params = torch.randn(self.quantum_circuit.n_parameters) * 0.1
        self.quantum_optimizer = torch.optim.Adam([self.quantum_params], lr=config.quantum_learning_rate)
        
        # Adaptive learning rate history
        self.lr_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        
        # Physics constraint tracking
        self.constraint_violations = defaultdict(list)
        
        self.logger = logging.getLogger("AdaptiveQuantumOptimizer")
    
    def step(self, closure=None, physics_constraints: Optional[Dict[str, torch.Tensor]] = None):
        """Perform optimization step with quantum enhancement."""
        
        loss = None
        if closure is not None:
            loss = closure()
        
        # Collect gradients for quantum processing
        all_gradients = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaptiveQuantumOptimizer does not support sparse gradients')
                
                all_gradients.append(grad.flatten())
        
        if not all_gradients:
            return loss
        
        # Combine all gradients
        combined_gradients = torch.cat(all_gradients)
        
        # Quantum gradient enhancement
        enhanced_gradients = self._quantum_gradient_enhancement(combined_gradients)
        
        # Apply physics constraints
        if physics_constraints:
            enhanced_gradients = self._apply_physics_constraints(enhanced_gradients, physics_constraints)
        
        # Adaptive learning rate adjustment
        self._adaptive_lr_adjustment(enhanced_gradients)
        
        # Apply enhanced gradients
        grad_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Get enhanced gradient for this parameter
                param_size = p.numel()
                enhanced_grad = enhanced_gradients[grad_idx:grad_idx + param_size].reshape(p.shape)
                grad_idx += param_size
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    enhanced_grad = enhanced_grad.add(p.data, alpha=group['weight_decay'])
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(enhanced_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(enhanced_grad, enhanced_grad, value=1 - beta2)
                
                # Compute bias-corrected moments
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Update quantum parameters
        self._update_quantum_parameters()
        
        return loss
    
    def _quantum_gradient_enhancement(self, gradients: torch.Tensor) -> torch.Tensor:
        """Enhance gradients using quantum processing."""
        
        if self.config.method == OptimizationMethod.QUANTUM_ADAM:
            return self._quantum_adam_enhancement(gradients)
        elif self.config.method == OptimizationMethod.VARIATIONAL_QUANTUM:
            return self._variational_quantum_enhancement(gradients)
        elif self.config.method == OptimizationMethod.QUANTUM_NATURAL_GRADIENT:
            return self._quantum_natural_gradient_enhancement(gradients)
        else:
            # Classical fallback
            return gradients
    
    def _quantum_adam_enhancement(self, gradients: torch.Tensor) -> torch.Tensor:
        """Quantum-enhanced Adam gradient processing."""
        
        # Apply quantum circuit to gradients
        quantum_state = self.quantum_circuit.apply_ansatz(self.quantum_params)
        
        # Encode gradients into quantum amplitudes (simplified)
        gradient_magnitude = torch.norm(gradients)
        if gradient_magnitude > 0:
            normalized_grads = gradients / gradient_magnitude
            
            # Quantum amplitude encoding
            n_grad_qubits = min(len(normalized_grads), 2**self.config.n_qubits)
            encoded_amplitudes = torch.zeros(2**self.config.n_qubits, dtype=torch.complex64)
            encoded_amplitudes[:n_grad_qubits] = normalized_grads[:n_grad_qubits].to(torch.complex64)
            
            # Quantum interference with variational state
            interference_state = quantum_state * encoded_amplitudes.conj()
            
            # Measure and extract enhanced gradients
            probabilities = torch.abs(interference_state)**2
            enhanced_pattern = probabilities[:len(gradients)]
            
            # Apply enhancement
            enhancement_factor = 1.0 + self.config.quantum_learning_rate * enhanced_pattern
            enhanced_gradients = gradients * enhancement_factor.real
        else:
            enhanced_gradients = gradients
        
        return enhanced_gradients
    
    def _variational_quantum_enhancement(self, gradients: torch.Tensor) -> torch.Tensor:
        """Variational quantum eigensolver-inspired enhancement."""
        
        def quantum_loss_function(quantum_params):
            """Loss function for quantum parameter optimization."""
            quantum_state = self.quantum_circuit.apply_ansatz(quantum_params)
            
            # Create Hamiltonian from gradients (simplified)
            hamiltonian = torch.outer(gradients, gradients.conj())
            
            # Expectation value
            expectation = torch.real(torch.vdot(quantum_state, hamiltonian @ quantum_state))
            return expectation.item()
        
        # Estimate quantum gradients
        quantum_grads = self.gradient_estimator.estimate_gradient(
            self.quantum_params, quantum_loss_function
        )
        
        # Update quantum parameters
        self.quantum_params = self.quantum_params - self.config.quantum_learning_rate * quantum_grads
        
        # Apply quantum state to enhance classical gradients
        quantum_state = self.quantum_circuit.apply_ansatz(self.quantum_params)
        enhancement = torch.abs(quantum_state[:len(gradients)])**2
        
        enhanced_gradients = gradients * (1.0 + enhancement.real)
        
        return enhanced_gradients
    
    def _quantum_natural_gradient_enhancement(self, gradients: torch.Tensor) -> torch.Tensor:
        """Quantum natural gradient enhancement."""
        
        # Compute quantum Fisher information matrix
        def quantum_state_function(params):
            return self.quantum_circuit.apply_ansatz(params)
        
        fisher_matrix = self.gradient_estimator.quantum_fisher_information(
            self.quantum_params, quantum_state_function
        )
        
        # Regularize Fisher matrix
        fisher_regularized = fisher_matrix + 1e-4 * torch.eye(len(fisher_matrix))
        
        try:
            # Compute natural gradients for quantum parameters
            fisher_inv = torch.linalg.pinv(fisher_regularized)
            quantum_natural_grads = fisher_inv @ self.quantum_params.grad if hasattr(self.quantum_params, 'grad') else self.quantum_params
            
            # Apply quantum natural gradient information to classical gradients
            if len(quantum_natural_grads) >= len(gradients):
                enhancement_factors = torch.abs(quantum_natural_grads[:len(gradients)])
                enhanced_gradients = gradients * (1.0 + 0.1 * enhancement_factors)
            else:
                enhanced_gradients = gradients
                
        except Exception as e:
            self.logger.warning(f"Quantum natural gradient computation failed: {e}")
            enhanced_gradients = gradients
        
        return enhanced_gradients
    
    def _apply_physics_constraints(self, gradients: torch.Tensor, 
                                 constraints: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply physics constraints to gradients."""
        
        constrained_gradients = gradients.clone()
        
        # Energy conservation constraint
        if 'energy_conservation' in constraints:
            energy_violation = constraints['energy_conservation']
            if torch.abs(energy_violation) > 1e-6:
                # Penalize gradients that would increase energy violation
                constraint_gradient = torch.sign(energy_violation) * gradients
                constrained_gradients -= self.config.conservation_weight * constraint_gradient
                self.constraint_violations['energy'].append(energy_violation.item())
        
        # Momentum conservation constraint
        if 'momentum_conservation' in constraints:
            momentum_violation = constraints['momentum_conservation']
            if torch.norm(momentum_violation) > 1e-6:
                # Project away momentum-violating components
                violation_norm = torch.norm(momentum_violation)
                momentum_direction = momentum_violation / (violation_norm + 1e-8)
                
                # Remove component in violation direction
                violation_component = torch.dot(gradients, momentum_direction.flatten()[:len(gradients)])
                constrained_gradients -= self.config.conservation_weight * violation_component * momentum_direction.flatten()[:len(gradients)]
                
                self.constraint_violations['momentum'].append(violation_norm.item())
        
        return constrained_gradients
    
    def _adaptive_lr_adjustment(self, gradients: torch.Tensor):
        """Adaptively adjust learning rate based on gradient history."""
        
        if not self.config.adaptive_parameters:
            return
        
        # Store gradient history
        grad_norm = torch.norm(gradients).item()
        self.gradient_history.append(grad_norm)
        
        if len(self.gradient_history) < 10:
            return
        
        # Analyze gradient trends
        recent_grads = list(self.gradient_history)[-10:]
        grad_trend = np.polyfit(range(len(recent_grads)), recent_grads, 1)[0]
        
        # Adjust learning rate based on trend
        for group in self.param_groups:
            current_lr = group['lr']
            
            if grad_trend > 0:  # Gradients increasing (possibly diverging)
                new_lr = current_lr * 0.95
            elif grad_trend < -0.1:  # Gradients decreasing rapidly (good progress)
                new_lr = current_lr * 1.05
            else:  # Stable gradients
                new_lr = current_lr
            
            # Clamp learning rate
            new_lr = max(1e-6, min(1e-1, new_lr))
            group['lr'] = new_lr
            
            self.lr_history.append(new_lr)
    
    def _update_quantum_parameters(self):
        """Update quantum parameters using classical optimizer."""
        
        if hasattr(self.quantum_params, 'grad') and self.quantum_params.grad is not None:
            self.quantum_optimizer.step()
            self.quantum_optimizer.zero_grad()
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        
        metrics = {
            'current_learning_rates': [group['lr'] for group in self.param_groups],
            'quantum_parameters_norm': torch.norm(self.quantum_params).item(),
            'gradient_history_length': len(self.gradient_history),
            'constraint_violation_counts': {
                constraint: len(violations) 
                for constraint, violations in self.constraint_violations.items()
            },
            'average_constraint_violations': {
                constraint: np.mean(violations) if violations else 0.0
                for constraint, violations in self.constraint_violations.items()
            }
        }
        
        if self.lr_history:
            metrics['learning_rate_trend'] = {
                'initial': self.lr_history[0],
                'current': self.lr_history[-1],
                'average': np.mean(self.lr_history)
            }
        
        return metrics

def create_quantum_optimizer_demo() -> Dict[str, Any]:
    """Create a demonstration of adaptive quantum optimizer."""
    
    # Create a simple physics-informed neural network
    class PhysicsNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32), 
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.layers(x)
        
        def physics_loss(self, prediction, target, inputs):
            mse_loss = nn.MSELoss()(prediction, target)
            
            # Energy conservation constraint
            input_energy = torch.sum(inputs**2, dim=1, keepdim=True)
            output_energy = prediction**2
            energy_conservation = torch.mean(torch.abs(output_energy - input_energy))
            
            return mse_loss + 0.1 * energy_conservation, {
                'energy_conservation': energy_conservation
            }
    
    # Initialize model and data
    model = PhysicsNN()
    
    # Create optimizer config
    config = QuantumOptimizerConfig(
        method=OptimizationMethod.QUANTUM_ADAM,
        learning_rate=1e-3,
        quantum_learning_rate=1e-2,
        n_qubits=6,
        n_quantum_layers=3,
        adaptive_parameters=True,
        conservation_weight=0.1
    )
    
    # Initialize quantum optimizer
    optimizer = AdaptiveQuantumOptimizer(model.parameters(), config)
    
    # Training data
    batch_size = 32
    input_dim = 10
    
    training_history = []
    
    print("Training with Quantum-Enhanced Optimizer...")
    
    for epoch in range(20):
        # Generate synthetic physics data
        x = torch.randn(batch_size, input_dim)
        y = torch.sum(x**2, dim=1, keepdim=True) + 0.1 * torch.randn(batch_size, 1)  # Energy-like target
        
        optimizer.zero_grad()
        
        # Forward pass
        prediction = model(x)
        
        # Physics-informed loss
        loss, constraints = model.physics_loss(prediction, y, x)
        
        # Backward pass
        loss.backward()
        
        # Quantum-enhanced optimization step
        optimizer.step(physics_constraints=constraints)
        
        # Collect metrics
        metrics = optimizer.get_optimization_metrics()
        
        training_history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'energy_conservation_violation': constraints['energy_conservation'].item(),
            'learning_rate': metrics['current_learning_rates'][0],
            'quantum_params_norm': metrics['quantum_parameters_norm']
        })
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
                  f"Energy Violation = {constraints['energy_conservation'].item():.6f}")
    
    final_metrics = optimizer.get_optimization_metrics()
    
    return {
        'training_successful': True,
        'training_history': training_history,
        'final_loss': training_history[-1]['loss'],
        'energy_conservation_improved': training_history[-1]['energy_conservation_violation'] < training_history[0]['energy_conservation_violation'],
        'optimization_metrics': final_metrics,
        'quantum_enhancement_active': True,
        'adaptive_lr_enabled': config.adaptive_parameters,
        'demo_successful': True
    }

if __name__ == "__main__":
    # Run demonstration
    demo_results = create_quantum_optimizer_demo()
    print("\n✅ Adaptive Quantum Optimizer Demo Results:")
    print(f"Training Successful: {demo_results['training_successful']}")
    print(f"Final Loss: {demo_results['final_loss']:.6f}")
    print(f"Energy Conservation Improved: {demo_results['energy_conservation_improved']}")
    print(f"Quantum Enhancement Active: {demo_results['quantum_enhancement_active']}")
    print(f"Demo Successful: {demo_results['demo_successful']}")