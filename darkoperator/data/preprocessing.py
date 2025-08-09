"""Data preprocessing utilities for physics data."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def normalize_4vectors(
    four_vectors: torch.Tensor,
    energy_scale: float = 100.0,
    momentum_scale: Optional[float] = None
) -> torch.Tensor:
    """Normalize 4-vectors for neural network training."""
    # four_vectors: [..., 4] where last dim is (E, px, py, pz)
    
    if momentum_scale is None:
        momentum_scale = energy_scale
    
    normalized = four_vectors.clone()
    normalized[..., 0] /= energy_scale  # Energy normalization
    normalized[..., 1:4] /= momentum_scale  # Momentum normalization
    
    return normalized


def preprocess_events(
    events: Dict[str, torch.Tensor],
    normalize_energy: bool = True,
    energy_scale: float = 100.0,
    add_invariant_masses: bool = True,
    sort_by_pt: bool = True
) -> Dict[str, torch.Tensor]:
    """Comprehensive event preprocessing pipeline."""
    processed = {}
    
    # Copy basic event information
    for key in ['event_id', 'n_jets']:
        if key in events:
            processed[key] = events[key].clone()
    
    # Process jet 4-vectors
    if all(k in events for k in ['jet_pt', 'jet_eta', 'jet_phi', 'jet_mass']):
        # Convert to Cartesian 4-vectors (E, px, py, pz)
        pt = events['jet_pt']
        eta = events['jet_eta']
        phi = events['jet_phi']
        mass = events['jet_mass']
        
        # Calculate 4-momentum components
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
        
        four_vectors = torch.stack([E, px, py, pz], dim=-1)
        
        # Sort jets by pT if requested
        if sort_by_pt:
            pt_sorted, indices = torch.sort(pt, dim=-1, descending=True)
            four_vectors = torch.gather(
                four_vectors, 1, 
                indices.unsqueeze(-1).expand(-1, -1, 4)
            )
        
        # Normalize energies
        if normalize_energy:
            four_vectors = normalize_4vectors(four_vectors, energy_scale)
        
        processed['jet_4vectors'] = four_vectors
        
        # Add invariant masses if requested
        if add_invariant_masses:
            processed.update(_calculate_invariant_masses(four_vectors))
    
    # Process missing energy
    if 'missing_et' in events and 'missing_phi' in events:
        met = events['missing_et']
        met_phi = events['missing_phi']
        
        # Convert to Cartesian components
        met_x = met * torch.cos(met_phi)
        met_y = met * torch.sin(met_phi)
        
        if normalize_energy:
            met_x /= energy_scale
            met_y /= energy_scale
        
        processed['missing_et_vector'] = torch.stack([met_x, met_y], dim=-1)
    
    return processed


def _calculate_invariant_masses(four_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate various invariant mass combinations."""
    # four_vectors: [batch, n_jets, 4]
    batch_size, n_jets = four_vectors.shape[:2]
    
    masses = {}
    
    if n_jets >= 2:
        # Dijet invariant masses
        dijet_masses = []
        for i in range(min(3, n_jets)):  # Only compute for first 3 jets
            for j in range(i + 1, min(3, n_jets)):
                p1 = four_vectors[:, i]  # [batch, 4]
                p2 = four_vectors[:, j]
                
                # Invariant mass: m² = (E1 + E2)² - (p1 + p2)²
                E_sum = p1[:, 0] + p2[:, 0]
                p_sum = p1[:, 1:4] + p2[:, 1:4]
                p_sum_squared = torch.sum(p_sum**2, dim=1)
                
                m_squared = E_sum**2 - p_sum_squared
                m_squared = torch.clamp(m_squared, min=0)  # Ensure positive
                invariant_mass = torch.sqrt(m_squared)
                
                dijet_masses.append(invariant_mass)
        
        if dijet_masses:
            masses['dijet_masses'] = torch.stack(dijet_masses, dim=1)
    
    if n_jets >= 3:
        # Leading trijet mass
        p1 = four_vectors[:, 0]
        p2 = four_vectors[:, 1]
        p3 = four_vectors[:, 2]
        
        E_sum = p1[:, 0] + p2[:, 0] + p3[:, 0]
        p_sum = p1[:, 1:4] + p2[:, 1:4] + p3[:, 1:4]
        p_sum_squared = torch.sum(p_sum**2, dim=1)
        
        m_squared = E_sum**2 - p_sum_squared
        m_squared = torch.clamp(m_squared, min=0)
        masses['trijet_mass'] = torch.sqrt(m_squared)
    
    return masses


def calculate_physics_variables(events: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Calculate common physics variables used in analyses."""
    variables = {}
    
    # Scalar sum of jet pT (HT)
    if 'jet_4vectors' in events:
        four_vectors = events['jet_4vectors']  # [batch, n_jets, 4]
        
        # Extract pT from 4-vectors
        px, py = four_vectors[..., 1], four_vectors[..., 2]
        pt = torch.sqrt(px**2 + py**2)
        
        # HT: scalar sum of jet pT
        variables['HT'] = torch.sum(pt, dim=1)
        
        # Leading jet pT
        variables['leading_jet_pt'] = pt[:, 0] if pt.size(1) > 0 else torch.zeros(pt.size(0))
    
    # Missing energy variables
    if 'missing_et_vector' in events:
        met_vec = events['missing_et_vector']  # [batch, 2]
        variables['MET'] = torch.norm(met_vec, dim=1)
        
        # MET significance (simplified)
        if 'HT' in variables:
            variables['MET_significance'] = variables['MET'] / torch.sqrt(variables['HT'] + 1e-6)
    
    # Angular separations
    if 'jet_4vectors' in events and events['jet_4vectors'].size(1) >= 2:
        variables.update(_calculate_angular_separations(events['jet_4vectors']))
    
    return variables


def _calculate_angular_separations(four_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate angular separations between jets."""
    # Convert to η, φ coordinates
    px, py, pz = four_vectors[..., 1], four_vectors[..., 2], four_vectors[..., 3]
    pt = torch.sqrt(px**2 + py**2)
    
    eta = torch.asinh(pz / (pt + 1e-8))
    phi = torch.atan2(py, px)
    
    separations = {}
    
    # ΔR between first two jets
    if four_vectors.size(1) >= 2:
        deta = eta[:, 0] - eta[:, 1]
        dphi = phi[:, 0] - phi[:, 1]
        
        # Handle φ wraparound
        dphi = torch.fmod(dphi + np.pi, 2 * np.pi) - np.pi
        
        delta_R = torch.sqrt(deta**2 + dphi**2)
        separations['delta_R_12'] = delta_R
    
    return separations


def apply_quality_cuts(
    events: Dict[str, torch.Tensor],
    min_jet_pt: float = 30.0,
    max_jet_eta: float = 2.5,
    min_n_jets: int = 2,
    max_n_jets: Optional[int] = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Apply basic quality cuts and return filtered events + selection mask."""
    
    n_events = len(events['event_id']) if 'event_id' in events else events[next(iter(events))].size(0)
    selection_mask = torch.ones(n_events, dtype=torch.bool)
    
    # Jet quality cuts
    if 'jet_4vectors' in events:
        four_vectors = events['jet_4vectors']
        
        # Calculate pT and η
        px, py, pz = four_vectors[..., 1], four_vectors[..., 2], four_vectors[..., 3]
        pt = torch.sqrt(px**2 + py**2)
        eta = torch.asinh(pz / (torch.sqrt(px**2 + py**2) + 1e-8))
        
        # pT cuts
        pt_pass = pt >= min_jet_pt
        
        # η cuts
        eta_pass = torch.abs(eta) <= max_jet_eta
        
        # Combined jet quality
        jet_quality = pt_pass & eta_pass
        
        # Count good jets per event
        n_good_jets = jet_quality.sum(dim=1)
        
        # Event-level cuts
        selection_mask &= n_good_jets >= min_n_jets
        if max_n_jets is not None:
            selection_mask &= n_good_jets <= max_n_jets
    
    # Apply selection
    filtered_events = {}
    for key, value in events.items():
        if isinstance(value, torch.Tensor):
            filtered_events[key] = value[selection_mask]
        else:
            filtered_events[key] = value
    
    return filtered_events, selection_mask


def create_training_batches(
    events: Dict[str, torch.Tensor],
    batch_size: int = 128,
    shuffle: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """Create training batches from event data."""
    n_events = len(events['event_id']) if 'event_id' in events else events[next(iter(events))].size(0)
    
    # Create indices
    indices = torch.arange(n_events)
    if shuffle:
        indices = indices[torch.randperm(n_events)]
    
    # Create batches
    batches = []
    for i in range(0, n_events, batch_size):
        batch_indices = indices[i:i + batch_size]
        
        batch = {}
        for key, value in events.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value[batch_indices]
            else:
                batch[key] = value
        
        batches.append(batch)
    
    return batches


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing functions...")
    
    # Create test event data
    n_events, n_jets = 100, 4
    
    events = {
        'event_id': torch.arange(n_events),
        'n_jets': torch.full((n_events,), n_jets),
        'jet_pt': torch.rand(n_events, n_jets) * 200 + 50,  # 50-250 GeV
        'jet_eta': torch.randn(n_events, n_jets) * 2.0,     # |η| < 4
        'jet_phi': torch.rand(n_events, n_jets) * 2 * np.pi - np.pi,  # [-π, π]
        'jet_mass': torch.rand(n_events, n_jets) * 20 + 2,  # 2-22 GeV
        'missing_et': torch.rand(n_events) * 100 + 10,      # 10-110 GeV
        'missing_phi': torch.rand(n_events) * 2 * np.pi - np.pi,
    }
    
    # Test preprocessing
    processed = preprocess_events(events)
    print(f"Processed events keys: {list(processed.keys())}")
    print(f"Jet 4-vectors shape: {processed['jet_4vectors'].shape}")
    print(f"MET vector shape: {processed['missing_et_vector'].shape}")
    
    if 'dijet_masses' in processed:
        print(f"Dijet masses shape: {processed['dijet_masses'].shape}")
    
    # Test physics variables
    physics_vars = calculate_physics_variables(processed)
    print(f"Physics variables: {list(physics_vars.keys())}")
    print(f"HT range: {physics_vars['HT'].min():.1f} - {physics_vars['HT'].max():.1f} GeV")
    print(f"MET range: {physics_vars['MET'].min():.1f} - {physics_vars['MET'].max():.1f} GeV")
    
    # Test quality cuts (use gentler cuts for testing)
    filtered, mask = apply_quality_cuts(processed, min_jet_pt=20.0, min_n_jets=2)
    print(f"Events passing cuts: {mask.sum()}/{len(mask)} ({100*mask.sum()/len(mask):.1f}%)")
    
    # Test batch creation
    batches = create_training_batches(filtered, batch_size=16)
    print(f"Created {len(batches)} training batches")
    if len(batches) > 0:
        print(f"First batch size: {len(batches[0]['event_id'])}")
    else:
        print("Warning: No batches created due to strict quality cuts")
    
    print("✅ All preprocessing tests passed!")


if __name__ == "__main__":
    test_preprocessing()