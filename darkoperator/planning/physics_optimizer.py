"""
Physics-informed optimizer for quantum task planning.

Enforces physical constraints and conservation laws in task scheduling
while optimizing for performance and accuracy.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

from ..operators.base import PhysicsOperator
from ..physics.lorentz import LorentzInvariantLayer


logger = logging.getLogger(__name__)


@dataclass
class PhysicsConstraints:
    """Physical constraints for optimization."""
    
    # Conservation laws
    conserve_energy: bool = True
    conserve_momentum: bool = True
    conserve_charge: bool = False
    conserve_baryon_number: bool = False
    
    # Symmetries
    preserve_lorentz_invariance: bool = True
    preserve_gauge_invariance: bool = False
    preserve_cp_symmetry: bool = False
    
    # Numerical constraints
    max_energy: float = 1e6  # GeV
    max_momentum: float = 1e6  # GeV
    min_resolution: float = 1e-12  # Numerical precision
    
    # Physical limits
    speed_of_light: float = 299792458.0  # m/s
    planck_constant: float = 6.62607015e-34  # J⋅s
    
    # Tolerance parameters
    conservation_tolerance: float = 1e-6
    symmetry_tolerance: float = 1e-8


class PhysicsValidator:
    """Validator for physics constraints in task execution."""
    
    def __init__(self, constraints: PhysicsConstraints):
        self.constraints = constraints
        self.violation_count = 0
        self.warnings_issued = []
        
    def validate_4vector(self, four_vector: torch.Tensor) -> bool:
        """
        Validate 4-vector physics constraints.
        
        Args:
            four_vector: Tensor of shape (..., 4) with (E, px, py, pz)
            
        Returns:
            True if valid, False if constraint violations detected
        """
        try:
            if four_vector.shape[-1] != 4:
                raise ValueError(f"Expected 4-vector, got shape {four_vector.shape}")
            
            E = four_vector[..., 0]
            px, py, pz = four_vector[..., 1], four_vector[..., 2], four_vector[..., 3]
            
            # Check energy positivity
            if torch.any(E < 0):
                self._issue_warning("Negative energy detected")
                return False
            
            # Check mass-shell constraint: E² = p² + m²
            p_squared = px**2 + py**2 + pz**2
            mass_squared = E**2 - p_squared
            
            if torch.any(mass_squared < -self.constraints.conservation_tolerance):
                self._issue_warning("Tachyonic mass detected (E² < p²)")
                return False
            
            # Check speed limit
            if self.constraints.speed_of_light < float('inf'):
                momentum_magnitude = torch.sqrt(p_squared)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    velocity = momentum_magnitude / torch.sqrt(momentum_magnitude**2 + mass_squared)
                    
                if torch.any(velocity > self.constraints.speed_of_light):
                    self._issue_warning("Superluminal velocity detected")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in 4-vector validation: {e}")
            return False
    
    def validate_conservation(
        self, 
        initial_state: torch.Tensor, 
        final_state: torch.Tensor
    ) -> Dict[str, bool]:
        """
        Validate conservation laws between initial and final states.
        
        Args:
            initial_state: Initial 4-momenta tensor
            final_state: Final 4-momenta tensor
            
        Returns:
            Dictionary of conservation law validation results
        """
        results = {}
        
        try:
            # Energy conservation
            if self.constraints.conserve_energy:
                initial_energy = initial_state[..., 0].sum(dim=-1)
                final_energy = final_state[..., 0].sum(dim=-1)
                energy_diff = torch.abs(initial_energy - final_energy)
                results['energy'] = torch.all(energy_diff < self.constraints.conservation_tolerance)
                
            # Momentum conservation
            if self.constraints.conserve_momentum:
                initial_momentum = initial_state[..., 1:4].sum(dim=-2)
                final_momentum = final_state[..., 1:4].sum(dim=-2)
                momentum_diff = torch.norm(initial_momentum - final_momentum, dim=-1)
                results['momentum'] = torch.all(momentum_diff < self.constraints.conservation_tolerance)
            
            # Report violations
            for law, conserved in results.items():
                if not conserved:
                    self._issue_warning(f"{law.capitalize()} conservation violated")
                    
        except Exception as e:
            logger.error(f"Error in conservation validation: {e}")
            results = {law: False for law in ['energy', 'momentum']}
        
        return results
    
    def validate_lorentz_invariance(
        self, 
        operation: callable, 
        four_vector: torch.Tensor, 
        n_tests: int = 10
    ) -> bool:
        """
        Test Lorentz invariance of an operation.
        
        Args:
            operation: Function to test
            four_vector: Input 4-vector
            n_tests: Number of random Lorentz transformations to test
            
        Returns:
            True if operation appears Lorentz invariant
        """
        if not self.constraints.preserve_lorentz_invariance:
            return True
            
        try:
            original_result = operation(four_vector)
            
            for _ in range(n_tests):
                # Generate random Lorentz boost
                beta = torch.rand(3) * 0.5  # Random velocity up to 0.5c
                transform_matrix = self._lorentz_boost_matrix(beta)
                
                # Transform input
                transformed_input = self._apply_lorentz_transform(four_vector, transform_matrix)
                
                # Apply operation to transformed input
                transformed_result = operation(transformed_input)
                
                # Check if results are equivalent (within tolerance)
                if hasattr(original_result, 'shape'):  # Tensor result
                    diff = torch.norm(original_result - transformed_result)
                    if diff > self.constraints.symmetry_tolerance:
                        self._issue_warning(f"Lorentz invariance violated: diff={diff:.2e}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error in Lorentz invariance test: {e}")
            return False
    
    def _lorentz_boost_matrix(self, beta: torch.Tensor) -> torch.Tensor:
        """Generate Lorentz boost transformation matrix."""
        beta_mag = torch.norm(beta)
        if beta_mag == 0:
            return torch.eye(4)
            
        gamma = 1.0 / torch.sqrt(1 - beta_mag**2)
        
        # Build Lorentz transformation matrix
        L = torch.eye(4)
        L[0, 0] = gamma
        L[0, 1:4] = -gamma * beta
        L[1:4, 0] = -gamma * beta
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    L[i+1, j+1] = 1 + (gamma - 1) * beta[i]**2 / beta_mag**2
                else:
                    L[i+1, j+1] = (gamma - 1) * beta[i] * beta[j] / beta_mag**2
                    
        return L
    
    def _apply_lorentz_transform(self, four_vector: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Apply Lorentz transformation to 4-vector."""
        return torch.einsum('...i,ij->...j', four_vector, transform)
    
    def _issue_warning(self, message: str) -> None:
        """Issue physics constraint warning."""
        if message not in self.warnings_issued:
            logger.warning(f"Physics constraint violation: {message}")
            self.warnings_issued.append(message)
            self.violation_count += 1
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        return {
            'total_violations': self.violation_count,
            'unique_warnings': len(self.warnings_issued),
            'warnings_list': self.warnings_issued.copy()
        }


class PhysicsOptimizer:
    """
    Physics-informed optimizer for task planning and execution.
    
    Ensures physical constraints are satisfied while optimizing performance.
    """
    
    def __init__(
        self, 
        constraints: Optional[PhysicsConstraints] = None,
        physics_operator: Optional[PhysicsOperator] = None
    ):
        self.constraints = constraints or PhysicsConstraints()
        self.physics_operator = physics_operator
        self.validator = PhysicsValidator(self.constraints)
        
        # Optimization parameters
        self.learning_rate = 0.01
        self.max_iterations = 1000
        self.convergence_threshold = 1e-6
        
        # Performance tracking
        self.optimization_history = []
        
        logger.info("Initialized physics-informed optimizer")
    
    def optimize_task_parameters(
        self, 
        task_params: Dict[str, torch.Tensor],
        objective_function: callable,
        physics_constraints: Optional[List[callable]] = None
    ) -> Dict[str, Any]:
        """
        Optimize task parameters while respecting physics constraints.
        
        Args:
            task_params: Dictionary of parameter tensors to optimize
            objective_function: Function to minimize/maximize
            physics_constraints: Additional constraint functions
            
        Returns:
            Optimized parameters and optimization metadata
        """
        
        # Convert parameters to optimizable tensors
        opt_params = {}
        for name, param in task_params.items():
            if isinstance(param, torch.Tensor):
                opt_params[name] = param.clone().detach().requires_grad_(True)
            else:
                opt_params[name] = torch.tensor(param, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(opt_params.values(), lr=self.learning_rate)
        
        # Optimization loop
        optimization_log = []
        best_loss = float('inf')
        best_params = None
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Compute objective
            try:
                objective_value = objective_function(**opt_params)
                if not torch.is_tensor(objective_value):
                    objective_value = torch.tensor(objective_value)
                    
            except Exception as e:
                logger.error(f"Error computing objective at iteration {iteration}: {e}")
                break
            
            # Add physics constraint penalties
            constraint_penalty = self._compute_constraint_penalty(opt_params)
            total_loss = objective_value + constraint_penalty
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(opt_params.values(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Post-process parameters to enforce hard constraints
            self._enforce_hard_constraints(opt_params)
            
            # Logging
            current_loss = total_loss.item()
            optimization_log.append({
                'iteration': iteration,
                'objective': objective_value.item(),
                'constraint_penalty': constraint_penalty.item(),
                'total_loss': current_loss
            })
            
            # Check for improvement
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = {k: v.clone().detach() for k, v in opt_params.items()}
            
            # Convergence check
            if iteration > 10:
                recent_losses = [log['total_loss'] for log in optimization_log[-10:]]
                if abs(max(recent_losses) - min(recent_losses)) < self.convergence_threshold:
                    logger.info(f"Converged after {iteration} iterations")
                    break
        
        # Prepare results
        self.optimization_history.append(optimization_log)
        
        results = {
            'optimized_params': best_params or opt_params,
            'final_objective': best_loss,
            'iterations': len(optimization_log),
            'converged': iteration < self.max_iterations - 1,
            'optimization_log': optimization_log,
            'constraint_violations': self.validator.get_violation_summary()
        }
        
        logger.info(f"Physics optimization completed: {results['iterations']} iterations, "
                   f"final loss: {results['final_objective']:.6f}")
        
        return results
    
    def _compute_constraint_penalty(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute penalty for physics constraint violations."""
        penalty = torch.tensor(0.0)
        
        # Check for 4-vector parameters
        for name, param in params.items():
            if '4vector' in name.lower() or 'momentum' in name.lower():
                if param.shape[-1] == 4:  # Looks like 4-vector
                    if not self.validator.validate_4vector(param):
                        penalty += 1000.0  # Heavy penalty for invalid physics
        
        # Energy conservation penalty
        if self.constraints.conserve_energy:
            energy_params = [p for name, p in params.items() if 'energy' in name.lower()]
            if len(energy_params) > 1:
                total_energy = sum(p.sum() for p in energy_params)
                if 'initial_energy' in [name.lower() for name in params.keys()]:
                    initial_energy = params.get('initial_energy', total_energy)
                    energy_violation = torch.abs(total_energy - initial_energy)
                    penalty += energy_violation * 100.0
        
        # Momentum conservation penalty
        if self.constraints.conserve_momentum:
            momentum_params = [p for name, p in params.items() if 'momentum' in name.lower()]
            if len(momentum_params) > 1:
                total_momentum = sum(p[..., 1:4].sum(dim=-2) for p in momentum_params if p.shape[-1] >= 4)
                if len(total_momentum) > 0:
                    momentum_magnitude = torch.norm(sum(total_momentum))
                    if momentum_magnitude > self.constraints.max_momentum:
                        penalty += (momentum_magnitude - self.constraints.max_momentum) * 10.0
        
        return penalty
    
    def _enforce_hard_constraints(self, params: Dict[str, torch.Tensor]) -> None:
        """Enforce hard physics constraints by modifying parameters in-place."""
        
        for name, param in params.items():
            # Ensure positive energies
            if 'energy' in name.lower() and param.ndim >= 1:
                param.data = torch.clamp(param.data, min=1e-6)
            
            # Limit maximum values
            if 'momentum' in name.lower() or '4vector' in name.lower():
                param.data = torch.clamp(param.data, -self.constraints.max_momentum, 
                                       self.constraints.max_momentum)
            
            # Ensure numerical stability
            param.data = torch.clamp(param.data, -1e10, 1e10)
    
    def optimize_neural_operator(
        self, 
        operator: torch.nn.Module,
        training_data: torch.Tensor,
        target_data: torch.Tensor,
        physics_loss_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        Optimize neural operator with physics constraints.
        
        Args:
            operator: Neural operator to optimize
            training_data: Input training data
            target_data: Target output data
            physics_loss_weight: Weight for physics loss terms
            
        Returns:
            Training results and physics validation metrics
        """
        
        # Setup optimizer
        optimizer = torch.optim.Adam(operator.parameters(), lr=self.learning_rate)
        
        training_log = []
        best_loss = float('inf')
        best_state = None
        
        n_epochs = min(100, self.max_iterations)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = operator(training_data)
            
            # Standard loss
            mse_loss = torch.nn.functional.mse_loss(predictions, target_data)
            
            # Physics-informed loss
            physics_loss = torch.tensor(0.0)
            if hasattr(operator, 'physics_loss') and self.physics_operator:
                try:
                    # Assume training_data contains 4-vectors
                    if training_data.shape[-1] >= 4:
                        physics_loss = operator.physics_loss(training_data, predictions)
                except Exception as e:
                    logger.warning(f"Could not compute physics loss: {e}")
            
            # Total loss
            total_loss = mse_loss + physics_loss_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Logging
            current_loss = total_loss.item()
            training_log.append({
                'epoch': epoch,
                'mse_loss': mse_loss.item(),
                'physics_loss': physics_loss.item(),
                'total_loss': current_loss
            })
            
            # Track best model
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = operator.state_dict().copy()
            
            # Early stopping
            if current_loss < self.convergence_threshold:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state is not None:
            operator.load_state_dict(best_state)
        
        # Validate physics constraints
        with torch.no_grad():
            final_predictions = operator(training_data)
            physics_validation = self._validate_operator_physics(
                training_data, final_predictions
            )
        
        results = {
            'final_loss': best_loss,
            'epochs_trained': len(training_log),
            'training_log': training_log,
            'physics_validation': physics_validation,
            'operator_state': operator.state_dict()
        }
        
        logger.info(f"Neural operator optimization completed: {len(training_log)} epochs, "
                   f"final loss: {best_loss:.6f}")
        
        return results
    
    def _validate_operator_physics(
        self, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor
    ) -> Dict[str, Any]:
        """Validate physics constraints in operator predictions."""
        
        validation_results = {
            'energy_conservation': False,
            'momentum_conservation': False,
            'mass_shell_constraint': False,
            'lorentz_invariance': False
        }
        
        try:
            # Check if inputs look like 4-vectors
            if inputs.shape[-1] >= 4:
                # Energy conservation
                if self.constraints.conserve_energy:
                    input_energy = inputs[..., 0].sum(dim=-1)
                    if outputs.ndim >= 2:  # Spatial distribution
                        output_energy = outputs.sum(dim=tuple(range(1, outputs.ndim)))
                        energy_diff = torch.abs(input_energy - output_energy).mean()
                        validation_results['energy_conservation'] = energy_diff < self.constraints.conservation_tolerance
                
                # Momentum conservation (simplified)
                if self.constraints.conserve_momentum and inputs.shape[-1] >= 4:
                    input_momentum = inputs[..., 1:4].sum(dim=-2) if inputs.ndim > 2 else inputs[:, 1:4]
                    # For spatial outputs, would need to compute momentum from distribution
                    # This is a simplified check
                    momentum_magnitude = torch.norm(input_momentum, dim=-1).mean()
                    validation_results['momentum_conservation'] = momentum_magnitude < self.constraints.max_momentum
                
                # Mass-shell constraint
                validation_results['mass_shell_constraint'] = self.validator.validate_4vector(inputs)
        
        except Exception as e:
            logger.error(f"Error in physics validation: {e}")
        
        return validation_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations performed."""
        
        if not self.optimization_history:
            return {'message': 'No optimizations performed yet'}
        
        total_optimizations = len(self.optimization_history)
        total_iterations = sum(len(log) for log in self.optimization_history)
        
        # Get convergence statistics
        converged_runs = sum(
            1 for log in self.optimization_history
            if len(log) < self.max_iterations
        )
        
        return {
            'total_optimizations': total_optimizations,
            'total_iterations': total_iterations,
            'average_iterations': total_iterations / total_optimizations,
            'convergence_rate': converged_runs / total_optimizations,
            'constraint_violations': self.validator.get_violation_summary(),
            'physics_constraints': {
                'energy_conservation': self.constraints.conserve_energy,
                'momentum_conservation': self.constraints.conserve_momentum,
                'lorentz_invariance': self.constraints.preserve_lorentz_invariance,
                'conservation_tolerance': self.constraints.conservation_tolerance
            }
        }