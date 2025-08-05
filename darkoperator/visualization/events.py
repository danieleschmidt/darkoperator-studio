"""Event visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from typing import Optional


def visualize_event(event: np.ndarray, save_path: Optional[str] = None) -> None:
    """Visualize a single event in 2D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract 4-vectors
    if event.ndim == 2:  # Multiple particles
        energy = event[:, 0]
        px, py, pz = event[:, 1], event[:, 2], event[:, 3]
        
        # Compute eta, phi
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz / (pt + 1e-8))
        phi = np.arctan2(py, px)
        
        # Transverse view
        ax1.scatter(px, py, s=energy*2, alpha=0.7, c=energy, cmap='viridis')
        ax1.set_xlabel('px (GeV)')
        ax1.set_ylabel('py (GeV)')
        ax1.set_title('Transverse View')
        ax1.grid(True)
        
        # Eta-phi view
        ax2.scatter(eta, phi, s=energy*2, alpha=0.7, c=energy, cmap='viridis')
        ax2.set_xlabel('η (pseudorapidity)')
        ax2.set_ylabel('φ (radians)')
        ax2.set_title('η-φ View')
        ax2.grid(True)
        
        plt.colorbar(ax2.collections[0], ax=ax2, label='Energy (GeV)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_3d(
    event: np.ndarray,
    detector: str = "cms",
    overlays: Optional[list] = None,
    save_html: Optional[str] = None
) -> go.Figure:
    """Create interactive 3D event visualization."""
    fig = go.Figure()
    
    if event.ndim == 2:  # Multiple particles
        energy = event[:, 0]
        px, py, pz = event[:, 1], event[:, 2], event[:, 3]
        
        # Add particle tracks
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='markers',
            marker=dict(
                size=energy/10,
                color=energy,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Energy (GeV)")
            ),
            name='Particles'
        ))
        
        # Add detector outline (simplified)
        if detector.lower() == "cms":
            # Simplified CMS barrel outline
            theta = np.linspace(0, 2*np.pi, 50)
            r_barrel = 120  # cm
            z_barrel = 280  # cm
            
            # Barrel
            x_barrel = r_barrel * np.cos(theta)
            y_barrel = r_barrel * np.sin(theta)
            z_barrel_pos = np.full_like(theta, z_barrel)
            z_barrel_neg = np.full_like(theta, -z_barrel)
            
            fig.add_trace(go.Scatter3d(
                x=np.concatenate([x_barrel, x_barrel]),
                y=np.concatenate([y_barrel, y_barrel]),
                z=np.concatenate([z_barrel_pos, z_barrel_neg]),
                mode='lines',
                line=dict(color='gray', width=2),
                name='CMS Outline',
                showlegend=False
            ))
    
    fig.update_layout(
        title=f'3D Event Display ({detector.upper()})',
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)', 
            zaxis_title='Z (cm)',
            aspectmode='cube'
        )
    )
    
    if save_html:
        fig.write_html(save_html)
    
    return fig