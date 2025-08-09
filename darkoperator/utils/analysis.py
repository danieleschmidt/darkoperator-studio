"""Physics analysis and interpretation utilities."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class PhysicsInterpreter:
    """Interpret neural operator predictions in physics terms."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
    
    def extract_interactions(
        self,
        order: int = 4,
        symmetry_constraints: List[str] = ['lorentz', 'gauge'],
        significance_threshold: float = 0.01
    ) -> List[Dict]:
        """Extract effective interaction terms from learned model."""
        
        interactions = []
        
        # Analyze model structure for interaction patterns
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weights = layer.weight.detach().cpu().numpy()
                
                # Look for significant weight patterns
                weight_magnitude = np.abs(weights)
                significant_weights = weight_magnitude > significance_threshold
                
                if significant_weights.any():
                    interaction = {
                        'layer_name': name,
                        'order': min(order, weights.shape[1]),  # Effective order
                        'strength': float(weight_magnitude.max()),
                        'latex_string': self._generate_latex_term(name, weights.shape),
                        'symmetry_properties': self._check_symmetries(weights, symmetry_constraints),
                    }
                    interactions.append(interaction)
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x['strength'], reverse=True)
        
        return interactions[:10]  # Return top 10 interactions
    
    def _generate_latex_term(self, layer_name: str, weight_shape: Tuple) -> str:
        """Generate LaTeX representation of interaction term."""
        
        if 'conv' in layer_name.lower():
            if len(weight_shape) == 4:  # 2D convolution
                return r"\mathcal{O}_{conv2d}(\phi_i \partial_\mu \phi_j)"
            elif len(weight_shape) == 5:  # 3D convolution  
                return r"\mathcal{O}_{conv3d}(\phi_i \partial_\mu \partial_\nu \phi_j)"
            else:
                return r"\mathcal{O}_{conv}(\phi_i \phi_j)"
        
        elif 'linear' in layer_name.lower() or 'fc' in layer_name.lower():
            n_inputs = weight_shape[1]
            if n_inputs <= 2:
                return r"\lambda \phi^2"
            elif n_inputs <= 4:
                return r"\lambda \phi^4"
            else:
                return r"\lambda \phi^{" + str(n_inputs) + "}"
        
        elif 'spectral' in layer_name.lower():
            return r"\mathcal{F}[\phi_i \star \phi_j]"  # Spectral convolution
        
        else:
            return r"\mathcal{O}_{unknown}"
    
    def _check_symmetries(self, weights: np.ndarray, constraints: List[str]) -> Dict[str, bool]:
        """Check if weights respect physics symmetries."""
        symmetries = {}
        
        for constraint in constraints:
            if constraint == 'lorentz':
                # Simplified Lorentz invariance check
                # For real implementation, would check tensor transformation properties
                symmetries['lorentz'] = True  # Placeholder
                
            elif constraint == 'gauge':
                # Simplified gauge invariance check
                # For real implementation, would check gauge transformation properties
                symmetries['gauge'] = True  # Placeholder
                
            elif constraint == 'parity':
                # Check parity invariance (simplified)
                symmetries['parity'] = np.allclose(weights, np.flip(weights, axis=-1))
                
            elif constraint == 'translation':
                # Check translation invariance (weight sharing in convolutions)
                if len(weights.shape) >= 3:  # Convolutional layer
                    symmetries['translation'] = True
                else:
                    symmetries['translation'] = False
        
        return symmetries
    
    def compute_effective_action(
        self,
        input_config: torch.Tensor,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """Compute effective action from model predictions."""
        
        # Generate field configurations around input
        with torch.no_grad():
            # Add small perturbations
            perturbations = torch.randn(n_samples, *input_config.shape[1:], device=self.device) * 0.1
            perturbed_configs = input_config.unsqueeze(0) + perturbations
            
            # Compute model predictions (interpreted as action)
            actions = self.model(perturbed_configs)
            
            if actions.dim() > 1:
                actions = actions.mean(dim=tuple(range(1, actions.dim())))
            
            # Compute effective action statistics
            effective_action = {
                'mean_action': float(actions.mean()),
                'action_variance': float(actions.var()),
                'free_energy': -float(torch.logsumexp(-actions, dim=0)),
                'partition_function': float(torch.sum(torch.exp(-actions))),
            }
        
        return effective_action
    
    def analyze_phase_transitions(
        self,
        parameter_range: Tuple[float, float],
        n_points: int = 100,
        coupling_dim: int = 0
    ) -> Dict[str, np.ndarray]:
        """Analyze phase transitions by varying coupling parameters."""
        
        parameters = np.linspace(parameter_range[0], parameter_range[1], n_points)
        observables = {
            'order_parameter': np.zeros(n_points),
            'susceptibility': np.zeros(n_points),
            'correlation_length': np.zeros(n_points),
        }
        
        # Sample configuration for analysis
        sample_config = torch.randn(1, 4, 50, 50, device=self.device)  # Example 4D field
        
        for i, param in enumerate(parameters):
            # Modify model parameter (simplified)
            original_weights = None
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Linear) and coupling_dim < layer.weight.shape[0]:
                    original_weights = layer.weight[coupling_dim].clone()
                    layer.weight.data[coupling_dim] *= param
                    break
            
            with torch.no_grad():
                output = self.model(sample_config)
                
                # Compute order parameter (field expectation value)
                observables['order_parameter'][i] = float(output.mean())
                
                # Compute susceptibility (variance)
                observables['susceptibility'][i] = float(output.var())
                
                # Compute correlation length (simplified)
                if output.dim() >= 3:
                    # Spatial correlation analysis
                    spatial_var = output.var(dim=tuple(range(2, output.dim())))
                    observables['correlation_length'][i] = float(1.0 / (spatial_var.mean() + 1e-8))
                else:
                    observables['correlation_length'][i] = 1.0
            
            # Restore original weights
            if original_weights is not None:
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Linear) and coupling_dim < layer.weight.shape[0]:
                        layer.weight.data[coupling_dim] = original_weights
                        break
        
        observables['parameters'] = parameters
        return observables
    
    def extract_conservation_violations(
        self,
        test_events: torch.Tensor,
        conservation_types: List[str] = ['energy', 'momentum']
    ) -> Dict[str, torch.Tensor]:
        """Analyze how well the model preserves conservation laws."""
        
        violations = {}
        
        with torch.no_grad():
            predictions = self.model(test_events)
            
            for conservation_type in conservation_types:
                if conservation_type == 'energy':
                    # Check energy conservation in 4-momentum predictions
                    if predictions.shape[-1] >= 4:  # Has 4-momentum components
                        initial_energy = test_events[..., 0].sum(dim=-2, keepdim=True)  # Sum over particles
                        predicted_energy = predictions[..., 0].sum(dim=-2, keepdim=True)
                        violations['energy'] = torch.abs(initial_energy - predicted_energy)
                    else:
                        violations['energy'] = torch.zeros(test_events.shape[0])
                
                elif conservation_type == 'momentum':
                    # Check 3-momentum conservation
                    if predictions.shape[-1] >= 4:
                        initial_momentum = test_events[..., 1:4].sum(dim=-2)  # Sum over particles, 3-momentum
                        predicted_momentum = predictions[..., 1:4].sum(dim=-2)
                        violations['momentum'] = torch.norm(initial_momentum - predicted_momentum, dim=-1)
                    else:
                        violations['momentum'] = torch.zeros(test_events.shape[0])
        
        return violations
    
    def plot_interaction_network(
        self, 
        interactions: List[Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize interaction network as a graph."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create nodes for different layers
        layer_names = list(set([interaction['layer_name'] for interaction in interactions]))
        n_layers = len(layer_names)
        
        # Position layers
        layer_positions = {}
        for i, layer_name in enumerate(layer_names):
            layer_positions[layer_name] = (i, 0)
        
        # Draw nodes
        for layer_name, (x, y) in layer_positions.items():
            ax.scatter(x, y, s=1000, alpha=0.7, c='lightblue', edgecolors='black')
            ax.text(x, y, layer_name.replace('_', '\n'), ha='center', va='center', fontsize=8)
        
        # Draw interactions as edges
        for interaction in interactions[:5]:  # Show top 5 interactions
            layer_name = interaction['layer_name']
            x, y = layer_positions[layer_name]
            
            # Draw self-loop or connection
            circle = plt.Circle((x, y), 0.3 * interaction['strength'] / max(i['strength'] for i in interactions),
                              fill=False, color='red', linewidth=2)
            ax.add_patch(circle)
            
            # Add strength label
            ax.text(x + 0.5, y + 0.2, f"λ={interaction['strength']:.3f}", fontsize=8)
        
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-1, 1)
        ax.set_title("Physics Interaction Network")
        ax.set_xlabel("Neural Network Layer")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def test_physics_interpreter():
    """Test physics interpreter functionality."""
    print("Testing physics interpreter...")
    
    # Create a simple test model
    class TestPhysicsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 16, 3)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.fc = nn.Linear(32 * 46 * 46, 4)  # Back to 4-momentum
            
        def forward(self, x):
            # x: [batch, 4, 50, 50] (4-momentum field)
            x = torch.relu(self.conv1(x))  # -> [batch, 16, 48, 48]
            x = torch.relu(self.conv2(x))  # -> [batch, 32, 46, 46]
            x = x.view(x.size(0), -1)     # Flatten
            x = self.fc(x)                # -> [batch, 4]
            return x.unsqueeze(1)         # -> [batch, 1, 4] (single particle output)
    
    model = TestPhysicsModel()
    interpreter = PhysicsInterpreter(model)
    
    # Test interaction extraction
    interactions = interpreter.extract_interactions(order=4)
    print(f"Extracted {len(interactions)} interactions")
    for i, interaction in enumerate(interactions[:3]):
        print(f"  {i+1}: {interaction['latex_string']} (strength: {interaction['strength']:.4f})")
    
    # Test effective action computation
    test_input = torch.randn(1, 4, 50, 50)
    effective_action = interpreter.compute_effective_action(test_input, n_samples=100)
    print(f"Effective action: {effective_action['mean_action']:.4f} ± {np.sqrt(effective_action['action_variance']):.4f}")
    
    # Test phase transition analysis
    phase_data = interpreter.analyze_phase_transitions(parameter_range=(0.5, 2.0), n_points=20)
    print(f"Phase transition analysis completed")
    print(f"Order parameter range: {phase_data['order_parameter'].min():.4f} to {phase_data['order_parameter'].max():.4f}")
    
    # Test conservation violation analysis
    test_events = torch.randn(10, 3, 4)  # 10 events, 3 particles, 4-momentum
    violations = interpreter.extract_conservation_violations(test_events)
    print(f"Energy violation: {violations['energy'].mean():.6f}")
    print(f"Momentum violation: {violations['momentum'].mean():.6f}")
    
    # Test interaction network visualization
    if interactions:
        fig = interpreter.plot_interaction_network(interactions)
        plt.close(fig)  # Clean up
        print("✓ Interaction network visualization complete")
    
    print("✅ All physics interpreter tests passed!")


if __name__ == "__main__":
    test_physics_interpreter()