"""Synthetic data generation for testing and benchmarking."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def generate_background_events(
    n_events: int = 10000,
    energy_range: Tuple[float, float] = (10.0, 1000.0),
    n_jets: int = 2,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Generate synthetic QCD dijet background events."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate jet energies with falling spectrum typical of QCD
    log_e_min, log_e_max = np.log10(energy_range[0]), np.log10(energy_range[1])
    
    events = {
        'event_id': torch.arange(n_events, dtype=torch.long),
        'n_jets': torch.full((n_events,), n_jets, dtype=torch.long),
        'jet_pt': torch.zeros(n_events, n_jets),
        'jet_eta': torch.zeros(n_events, n_jets),
        'jet_phi': torch.zeros(n_events, n_jets),
        'jet_mass': torch.zeros(n_events, n_jets),
        'missing_et': torch.zeros(n_events),
        'missing_phi': torch.zeros(n_events),
    }
    
    for i in range(n_events):
        # Generate jet pT with falling spectrum: dN/dpT ~ pT^(-6)
        log_pt = torch.rand(n_jets) * (log_e_max - log_e_min) + log_e_min
        jet_pt = 10**log_pt
        
        # Sort jets by pT (highest first)
        jet_pt, _ = torch.sort(jet_pt, descending=True)
        events['jet_pt'][i] = jet_pt
        
        # Generate jet pseudorapidity (|η| < 2.5)
        events['jet_eta'][i] = torch.randn(n_jets) * 1.0  # Central jets
        
        # Generate jet azimuthal angle
        events['jet_phi'][i] = torch.rand(n_jets) * 2 * np.pi - np.pi
        
        # Generate jet masses (light quarks/gluons)
        events['jet_mass'][i] = torch.randn(n_jets).abs() * 5.0 + 2.0  # Small masses
        
        # Generate missing energy (small for QCD)
        events['missing_et'][i] = torch.randn(1).abs() * 20.0
        events['missing_phi'][i] = torch.rand(1) * 2 * np.pi - np.pi
    
    return events


def generate_signal_events(
    n_events: int = 1000,
    signal_type: str = 'dark_matter',
    dark_matter_mass: float = 100.0,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Generate synthetic signal events for various BSM scenarios."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if signal_type == 'dark_matter':
        return _generate_dark_matter_events(n_events, dark_matter_mass)
    elif signal_type == 'susy':
        return _generate_susy_events(n_events)
    elif signal_type == 'extra_dimensions':
        return _generate_extra_dim_events(n_events)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def _generate_dark_matter_events(n_events: int, dm_mass: float) -> Dict[str, torch.Tensor]:
    """Generate mono-jet + missing energy dark matter events."""
    events = {
        'event_id': torch.arange(n_events, dtype=torch.long),
        'n_jets': torch.ones(n_events, dtype=torch.long),  # Mono-jet
        'jet_pt': torch.zeros(n_events, 1),
        'jet_eta': torch.zeros(n_events, 1),
        'jet_phi': torch.zeros(n_events, 1),
        'jet_mass': torch.zeros(n_events, 1),
        'missing_et': torch.zeros(n_events),
        'missing_phi': torch.zeros(n_events),
    }
    
    for i in range(n_events):
        # High-pT jet recoiling against invisible dark matter
        jet_pt = torch.randn(1).abs() * 100.0 + 200.0  # High pT jets
        events['jet_pt'][i] = jet_pt
        
        # Central jet
        events['jet_eta'][i] = torch.randn(1) * 0.8
        events['jet_phi'][i] = torch.rand(1) * 2 * np.pi - np.pi
        events['jet_mass'][i] = torch.randn(1).abs() * 10.0 + 5.0
        
        # Large missing energy (dark matter particles)
        met = torch.randn(1).abs() * 50.0 + dm_mass * 2  # MET ~ 2*M_DM
        events['missing_et'][i] = met
        
        # MET roughly back-to-back with jet
        jet_phi = events['jet_phi'][i].item()
        met_phi = jet_phi + np.pi + torch.randn(1) * 0.3  # Small angular spread
        events['missing_phi'][i] = torch.fmod(met_phi, 2 * np.pi) - np.pi
    
    return events


def _generate_susy_events(n_events: int) -> Dict[str, torch.Tensor]:
    """Generate SUSY cascade decay events."""
    events = {
        'event_id': torch.arange(n_events, dtype=torch.long),
        'n_jets': torch.randint(3, 8, (n_events,)),  # Multiple jets from cascades
        'missing_et': torch.zeros(n_events),
        'missing_phi': torch.zeros(n_events),
    }
    
    max_jets = events['n_jets'].max().item()
    events['jet_pt'] = torch.zeros(n_events, max_jets)
    events['jet_eta'] = torch.zeros(n_events, max_jets)
    events['jet_phi'] = torch.zeros(n_events, max_jets)
    events['jet_mass'] = torch.zeros(n_events, max_jets)
    
    for i in range(n_events):
        n_jets_i = events['n_jets'][i].item()
        
        # Generate jets with SUSY-like pT spectrum
        jet_pt = torch.rand(n_jets_i) * 300.0 + 50.0
        jet_pt, _ = torch.sort(jet_pt, descending=True)
        events['jet_pt'][i, :n_jets_i] = jet_pt
        
        # Isotropic jet directions
        events['jet_eta'][i, :n_jets_i] = torch.randn(n_jets_i) * 2.0
        events['jet_phi'][i, :n_jets_i] = torch.rand(n_jets_i) * 2 * np.pi - np.pi
        events['jet_mass'][i, :n_jets_i] = torch.randn(n_jets_i).abs() * 8.0 + 3.0
        
        # Moderate missing energy from LSPs
        events['missing_et'][i] = torch.randn(1).abs() * 80.0 + 100.0
        events['missing_phi'][i] = torch.rand(1) * 2 * np.pi - np.pi
    
    return events


def _generate_extra_dim_events(n_events: int) -> Dict[str, torch.Tensor]:
    """Generate extra-dimensional black hole events."""
    events = {
        'event_id': torch.arange(n_events, dtype=torch.long),
        'n_jets': torch.randint(5, 15, (n_events,)),  # Many jets from BH evaporation
        'missing_et': torch.zeros(n_events),
        'missing_phi': torch.zeros(n_events),
    }
    
    max_jets = events['n_jets'].max().item()
    events['jet_pt'] = torch.zeros(n_events, max_jets)
    events['jet_eta'] = torch.zeros(n_events, max_jets)
    events['jet_phi'] = torch.zeros(n_events, max_jets)
    events['jet_mass'] = torch.zeros(n_events, max_jets)
    
    for i in range(n_events):
        n_jets_i = events['n_jets'][i].item()
        
        # High multiplicity, high pT jets from BH evaporation
        jet_pt = torch.rand(n_jets_i) * 500.0 + 100.0
        jet_pt, _ = torch.sort(jet_pt, descending=True)
        events['jet_pt'][i, :n_jets_i] = jet_pt
        
        # Isotropic, high-pT jets
        events['jet_eta'][i, :n_jets_i] = torch.randn(n_jets_i) * 2.5
        events['jet_phi'][i, :n_jets_i] = torch.rand(n_jets_i) * 2 * np.pi - np.pi
        events['jet_mass'][i, :n_jets_i] = torch.randn(n_jets_i).abs() * 15.0 + 5.0
        
        # Large missing energy from gravitons escaping to extra dimensions
        events['missing_et'][i] = torch.randn(1).abs() * 200.0 + 300.0
        events['missing_phi'][i] = torch.rand(1) * 2 * np.pi - np.pi
    
    return events


def generate_calorimeter_showers(
    event_data: Dict[str, torch.Tensor],
    n_events: int = 1000,
    calorimeter_shape: Tuple[int, int, int] = (50, 50, 25),
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate synthetic calorimeter shower data from event kinematics."""
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = n_events
    depth, height, width = calorimeter_shape
    showers = torch.zeros(batch_size, depth, height, width)
    
    for i in range(batch_size):
        n_jets = event_data['n_jets'][i].item()
        
        for j in range(n_jets):
            if j >= event_data['jet_pt'].size(1):
                break
                
            pt = event_data['jet_pt'][i, j].item()
            eta = event_data['jet_eta'][i, j].item()
            phi = event_data['jet_phi'][i, j].item()
            
            if pt == 0:  # Skip empty jets
                continue
            
            # Convert eta/phi to calorimeter coordinates
            eta_idx = int((eta + 2.5) / 5.0 * height)  # Map η ∈ [-2.5, 2.5]
            phi_idx = int((phi + np.pi) / (2 * np.pi) * width)  # Map φ ∈ [-π, π]
            
            eta_idx = max(0, min(height - 1, eta_idx))
            phi_idx = max(0, min(width - 1, phi_idx))
            
            # Simulate electromagnetic and hadronic showers
            for d in range(depth):
                # Exponential energy deposition with depth
                depth_fraction = np.exp(-d / (depth / 3))  # 1/3 radiation length
                energy_deposit = pt * depth_fraction / depth
                
                # Gaussian spread in η-φ
                sigma = 2 + d * 0.1  # Shower spreading with depth
                
                for deta in range(-3, 4):
                    for dphi in range(-3, 4):
                        eta_pos = eta_idx + deta
                        phi_pos = phi_idx + dphi
                        
                        if 0 <= eta_pos < height and 0 <= phi_pos < width:
                            # Gaussian shower profile
                            r_squared = deta**2 + dphi**2
                            shower_weight = np.exp(-r_squared / (2 * sigma**2))
                            
                            showers[i, d, eta_pos, phi_pos] += energy_deposit * shower_weight
    
    # Add noise
    noise = torch.randn_like(showers) * 0.5
    showers = torch.clamp(showers + noise, min=0)  # No negative energy
    
    return showers


def plot_event_distributions(events: Dict[str, torch.Tensor], title: str = "Event Distributions"):
    """Plot basic kinematic distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    
    # Leading jet pT
    leading_pt = events['jet_pt'][:, 0].numpy()
    axes[0, 0].hist(leading_pt, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Leading Jet pT [GeV]')
    axes[0, 0].set_ylabel('Events')
    axes[0, 0].set_yscale('log')
    
    # Missing ET
    met = events['missing_et'].numpy()
    axes[0, 1].hist(met, bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[0, 1].set_xlabel('Missing ET [GeV]')
    axes[0, 1].set_ylabel('Events')
    axes[0, 1].set_yscale('log')
    
    # Jet eta
    all_eta = events['jet_eta'][events['jet_eta'] != 0].numpy()
    axes[1, 0].hist(all_eta, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Jet η')
    axes[1, 0].set_ylabel('Jets')
    
    # Jet multiplicity
    n_jets = events['n_jets'].numpy()
    axes[1, 1].hist(n_jets, bins=np.arange(0.5, n_jets.max() + 1.5), alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_xlabel('Number of Jets')
    axes[1, 1].set_ylabel('Events')
    
    plt.tight_layout()
    return fig


def test_synthetic_data():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")
    
    # Generate background events
    bg_events = generate_background_events(n_events=1000, seed=42)
    print(f"Generated {len(bg_events['event_id'])} background events")
    print(f"Background leading jet pT range: {bg_events['jet_pt'][:, 0].min():.1f} - {bg_events['jet_pt'][:, 0].max():.1f} GeV")
    
    # Generate signal events
    dm_events = generate_signal_events(n_events=100, signal_type='dark_matter', seed=42)
    print(f"Generated {len(dm_events['event_id'])} dark matter signal events")
    print(f"DM signal leading jet pT range: {dm_events['jet_pt'][:, 0].min():.1f} - {dm_events['jet_pt'][:, 0].max():.1f} GeV")
    print(f"DM signal MET range: {dm_events['missing_et'].min():.1f} - {dm_events['missing_et'].max():.1f} GeV")
    
    # Generate SUSY events
    susy_events = generate_signal_events(n_events=100, signal_type='susy', seed=42)
    print(f"Generated {len(susy_events['event_id'])} SUSY signal events")
    print(f"SUSY average jet multiplicity: {susy_events['n_jets'].float().mean():.1f}")
    
    # Test calorimeter shower generation
    first_10_events = {k: v[:10] if isinstance(v, torch.Tensor) else v for k, v in bg_events.items()}
    showers = generate_calorimeter_showers(event_data=first_10_events, n_events=10)
    print(f"Generated calorimeter showers: {showers.shape}")
    print(f"Total energy in showers: {showers.sum():.1f} GeV")
    
    print("✅ All synthetic data tests passed!")


if __name__ == "__main__":
    test_synthetic_data()