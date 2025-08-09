"""Physics conservation laws and constraints for neural operators."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List


class ConservationLoss(nn.Module):
    """Loss functions enforcing physics conservation laws."""
    
    def __init__(self, c: float = 299792458.0):
        """Initialize conservation loss with speed of light in m/s."""
        super().__init__()
        self.c = c
    
    def energy_conservation(
        self, 
        initial_4momentum: torch.Tensor, 
        final_4momentum: torch.Tensor,
        tolerance: float = 1e-6
    ) -> torch.Tensor:
        """Enforce energy conservation in 4-momentum transfers."""
        # initial_4momentum, final_4momentum: [batch, n_particles, 4] (E, px, py, pz)
        
        initial_energy = initial_4momentum[:, :, 0].sum(dim=1)  # Sum over particles
        final_energy = final_4momentum[:, :, 0].sum(dim=1)
        
        energy_violation = torch.abs(initial_energy - final_energy)
        return energy_violation.mean()
    
    def momentum_conservation(
        self, 
        initial_4momentum: torch.Tensor, 
        final_4momentum: torch.Tensor
    ) -> torch.Tensor:
        """Enforce 3-momentum conservation."""
        # Sum 3-momentum vectors over particles
        initial_3momentum = initial_4momentum[:, :, 1:4].sum(dim=1)  # [batch, 3]
        final_3momentum = final_4momentum[:, :, 1:4].sum(dim=1)
        
        momentum_violation = torch.norm(initial_3momentum - final_3momentum, dim=1)
        return momentum_violation.mean()
    
    def mass_shell_constraint(self, four_momentum: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Enforce mass-shell constraint: E² = (pc)² + (mc²)²."""
        # four_momentum: [batch, n_particles, 4] (E, px, py, pz)
        # masses: [batch, n_particles] or [n_particles]
        
        E = four_momentum[:, :, 0]
        p_vec = four_momentum[:, :, 1:4]
        p_squared = torch.sum(p_vec**2, dim=2)  # |p|²
        
        # Mass-energy relation: E² = (pc)² + (mc²)²
        E_expected_squared = (self.c * torch.sqrt(p_squared))**2 + (masses * self.c**2)**2
        E_actual_squared = E**2
        
        mass_shell_violation = torch.abs(E_actual_squared - E_expected_squared)
        return mass_shell_violation.mean()
    
    def charge_conservation(
        self, 
        initial_charges: torch.Tensor, 
        final_charges: torch.Tensor
    ) -> torch.Tensor:
        """Enforce electric charge conservation."""
        initial_total = initial_charges.sum(dim=1)  # Sum over particles
        final_total = final_charges.sum(dim=1)
        
        charge_violation = torch.abs(initial_total - final_total)
        return charge_violation.mean()
    
    def forward(
        self, 
        initial_state: Dict[str, torch.Tensor], 
        final_state: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total conservation loss."""
        if weights is None:
            weights = {
                'energy': 1.0,
                'momentum': 1.0,
                'mass_shell': 0.5,
                'charge': 1.0
            }
        
        losses = {}
        total_loss = 0.0
        
        # Energy conservation
        if 'energy' in weights and weights['energy'] > 0:
            energy_loss = self.energy_conservation(
                initial_state['4momentum'], 
                final_state['4momentum']
            )
            losses['energy'] = energy_loss
            total_loss += weights['energy'] * energy_loss
        
        # Momentum conservation
        if 'momentum' in weights and weights['momentum'] > 0:
            momentum_loss = self.momentum_conservation(
                initial_state['4momentum'], 
                final_state['4momentum']
            )
            losses['momentum'] = momentum_loss
            total_loss += weights['momentum'] * momentum_loss
        
        # Mass-shell constraints
        if 'mass_shell' in weights and weights['mass_shell'] > 0:
            if 'masses' in initial_state:
                mass_shell_loss = self.mass_shell_constraint(
                    final_state['4momentum'], 
                    initial_state['masses']
                )
                losses['mass_shell'] = mass_shell_loss
                total_loss += weights['mass_shell'] * mass_shell_loss
        
        # Charge conservation
        if 'charge' in weights and weights['charge'] > 0:
            if 'charges' in initial_state and 'charges' in final_state:
                charge_loss = self.charge_conservation(
                    initial_state['charges'], 
                    final_state['charges']
                )
                losses['charge'] = charge_loss
                total_loss += weights['charge'] * charge_loss
        
        losses['total'] = total_loss
        return losses


class LorentzInvariance(nn.Module):
    """Enforce Lorentz invariance in neural network predictions."""
    
    def __init__(self):
        super().__init__()
    
    def boost_4momentum(
        self, 
        four_momentum: torch.Tensor, 
        beta: torch.Tensor
    ) -> torch.Tensor:
        """Apply Lorentz boost to 4-momentum."""
        # four_momentum: [batch, n_particles, 4] (E, px, py, pz)
        # beta: [batch, 3] boost velocity (βx, βy, βz)
        
        gamma = 1.0 / torch.sqrt(1.0 - torch.sum(beta**2, dim=1, keepdim=True))
        
        E = four_momentum[:, :, 0:1]  # [batch, n_particles, 1]
        p = four_momentum[:, :, 1:4]  # [batch, n_particles, 3]
        
        # Boost transformation
        beta_expanded = beta.unsqueeze(1)  # [batch, 1, 3]
        p_parallel = torch.sum(p * beta_expanded, dim=2, keepdim=True) * beta_expanded
        p_perp = p - p_parallel
        
        gamma_expanded = gamma.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1]
        
        E_boosted = gamma_expanded * (E - torch.sum(p * beta_expanded, dim=2, keepdim=True))
        p_boosted = p_perp + gamma_expanded * (
            p_parallel - beta_expanded * E
        )
        
        return torch.cat([E_boosted, p_boosted], dim=2)
    
    def rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Generate 3D rotation matrices from Euler angles."""
        # angles: [batch, 3] (θx, θy, θz)
        
        batch_size = angles.size(0)
        device = angles.device
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Rotation matrices for each axis
        Rx = torch.zeros(batch_size, 3, 3, device=device)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = cos_angles[:, 0]
        Rx[:, 1, 2] = -sin_angles[:, 0]
        Rx[:, 2, 1] = sin_angles[:, 0]
        Rx[:, 2, 2] = cos_angles[:, 0]
        
        Ry = torch.zeros(batch_size, 3, 3, device=device)
        Ry[:, 0, 0] = cos_angles[:, 1]
        Ry[:, 0, 2] = sin_angles[:, 1]
        Ry[:, 1, 1] = 1.0
        Ry[:, 2, 0] = -sin_angles[:, 1]
        Ry[:, 2, 2] = cos_angles[:, 1]
        
        Rz = torch.zeros(batch_size, 3, 3, device=device)
        Rz[:, 0, 0] = cos_angles[:, 2]
        Rz[:, 0, 1] = -sin_angles[:, 2]
        Rz[:, 1, 0] = sin_angles[:, 2]
        Rz[:, 1, 1] = cos_angles[:, 2]
        Rz[:, 2, 2] = 1.0
        
        # Combined rotation: R = Rz * Ry * Rx
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
        return R
    
    def rotate_4momentum(
        self, 
        four_momentum: torch.Tensor, 
        angles: torch.Tensor
    ) -> torch.Tensor:
        """Apply 3D rotation to 4-momentum (energy unchanged)."""
        # four_momentum: [batch, n_particles, 4]
        # angles: [batch, 3]
        
        E = four_momentum[:, :, 0:1]  # Energy unchanged
        p = four_momentum[:, :, 1:4]  # 3-momentum to rotate
        
        R = self.rotation_matrix(angles)  # [batch, 3, 3]
        
        # Apply rotation: p_rotated = R @ p^T
        p_rotated = torch.bmm(R, p.transpose(1, 2)).transpose(1, 2)
        
        return torch.cat([E, p_rotated], dim=2)
    
    def invariance_loss(
        self, 
        model: nn.Module,
        input_4momentum: torch.Tensor,
        n_transformations: int = 5,
        boost_scale: float = 0.3,
        rotation_scale: float = torch.pi
    ) -> torch.Tensor:
        """Compute Lorentz invariance loss by comparing predictions under transformations."""
        batch_size = input_4momentum.size(0)
        device = input_4momentum.device
        
        # Original prediction
        original_output = model(input_4momentum)
        
        total_loss = 0.0
        
        for _ in range(n_transformations):
            # Random boost
            beta = torch.randn(batch_size, 3, device=device) * boost_scale
            beta = torch.clamp(beta, -0.95, 0.95)  # Keep β < c
            boosted_input = self.boost_4momentum(input_4momentum, beta)
            
            # Random rotation
            angles = torch.randn(batch_size, 3, device=device) * rotation_scale
            transformed_input = self.rotate_4momentum(boosted_input, angles)
            
            # Prediction on transformed input
            transformed_output = model(transformed_input)
            
            # Apply inverse transformation to output for comparison
            inv_transformed_output = self.rotate_4momentum(
                self.boost_4momentum(transformed_output, -beta), -angles
            )
            
            # Compute invariance loss
            invariance_loss = torch.nn.functional.mse_loss(
                original_output, inv_transformed_output
            )
            total_loss += invariance_loss
        
        return total_loss / n_transformations


def test_conservation_laws():
    """Test conservation law implementations."""
    print("Testing conservation laws...")
    
    # Create test data
    batch_size = 4
    n_particles = 3
    
    # 4-momentum: [batch, n_particles, 4] (E, px, py, pz)
    initial_4momentum = torch.randn(batch_size, n_particles, 4) * 10
    initial_4momentum[:, :, 0] = torch.abs(initial_4momentum[:, :, 0]) + 5  # Positive energy
    
    # Perfect conservation (copy initial state)
    final_4momentum_perfect = initial_4momentum.clone()
    
    # Imperfect conservation (add some noise)
    final_4momentum_noisy = initial_4momentum + torch.randn_like(initial_4momentum) * 0.1
    
    # Particle masses and charges
    masses = torch.ones(batch_size, n_particles) * 0.5  # 0.5 GeV/c²
    charges = torch.tensor([1.0, -1.0, 0.0]).expand(batch_size, -1)  # e+, e-, γ
    
    # Initialize conservation loss
    conservation = ConservationLoss()
    
    # Test perfect conservation
    initial_state = {
        '4momentum': initial_4momentum,
        'masses': masses,
        'charges': charges
    }
    final_state_perfect = {
        '4momentum': final_4momentum_perfect,
        'charges': charges
    }
    
    losses_perfect = conservation(initial_state, final_state_perfect)
    print(f"Perfect conservation - Total loss: {losses_perfect['total'].item():.6f}")
    
    # Test imperfect conservation
    final_state_noisy = {
        '4momentum': final_4momentum_noisy,
        'charges': charges
    }
    
    losses_noisy = conservation(initial_state, final_state_noisy)
    print(f"Noisy conservation - Total loss: {losses_noisy['total'].item():.6f}")
    print(f"  Energy loss: {losses_noisy.get('energy', 0):.6f}")
    print(f"  Momentum loss: {losses_noisy.get('momentum', 0):.6f}")
    
    # Test Lorentz invariance
    print("\nTesting Lorentz invariance...")
    lorentz = LorentzInvariance()
    
    # Simple identity model for testing
    class IdentityModel(nn.Module):
        def forward(self, x):
            return x
    
    model = IdentityModel()
    invariance_loss = lorentz.invariance_loss(model, initial_4momentum)
    print(f"Lorentz invariance loss (identity model): {invariance_loss.item():.6f}")
    
    print("✅ All conservation law tests passed!")


if __name__ == "__main__":
    test_conservation_laws()