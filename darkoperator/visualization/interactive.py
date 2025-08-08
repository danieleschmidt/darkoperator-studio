"""
Advanced 3D Interactive Visualization for DarkOperator Studio.

Provides sophisticated visualization capabilities for particle physics events,
detector geometries, and physics analysis results with interactive exploration.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, 3D visualization will be limited")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, fallback visualizations unavailable")


class Interactive3DVisualizer:
    """Advanced 3D visualization system for particle physics events."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize 3D visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different particle types
        self.particle_colors = {
            'electron': '#FF6B6B',     # Red
            'muon': '#4ECDC4',         # Teal
            'photon': '#FFE66D',       # Yellow
            'tau': '#95E1D3',          # Light green
            'neutrino': '#A8E6CF',     # Pale green
            'quark': '#FF8B94',        # Pink
            'gluon': '#C7CEEA',        # Light purple
            'jet': '#FFD93D',          # Golden
            'missing_et': '#FF5722',   # Deep orange
            'unknown': '#9E9E9E'       # Gray
        }
        
        # Detector color schemes
        self.detector_colors = {
            'tracker': '#2196F3',      # Blue
            'ecal': '#4CAF50',         # Green
            'hcal': '#FF9800',         # Orange
            'muon_chamber': '#9C27B0', # Purple
            'magnet': '#607D8B',       # Blue gray
            'beam_pipe': '#795548'     # Brown
        }
        
        logger.info(f"3D Visualizer initialized, output directory: {self.output_dir}")
    
    def visualize_event_3d(
        self,
        event_data: Dict[str, Any],
        detector_geometry: Optional[Dict[str, Any]] = None,
        show_tracks: bool = True,
        show_energy_deposits: bool = True,
        show_missing_et: bool = True,
        interactive: bool = True,
        save_html: Optional[str] = None
    ) -> Optional[str]:
        """
        Create comprehensive 3D visualization of particle physics event.
        
        Args:
            event_data: Event data with particles, tracks, energy deposits
            detector_geometry: Detector geometry information
            show_tracks: Show particle tracks
            show_energy_deposits: Show calorimeter energy deposits
            show_missing_et: Show missing transverse energy
            interactive: Create interactive plot
            save_html: Path to save HTML file
            
        Returns:
            Path to saved HTML file or None
        """
        
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for 3D event visualization")
            return None
        
        # Create 3D figure
        fig = go.Figure()
        
        # Add detector geometry
        if detector_geometry:
            self._add_detector_geometry(fig, detector_geometry)
        else:
            # Add default ATLAS/CMS-like geometry
            self._add_default_detector_geometry(fig)
        
        # Add particle tracks
        if show_tracks and 'particles' in event_data:
            self._add_particle_tracks(fig, event_data['particles'])
        
        # Add energy deposits
        if show_energy_deposits and 'energy_deposits' in event_data:
            self._add_energy_deposits(fig, event_data['energy_deposits'])
        
        # Add missing ET visualization
        if show_missing_et and 'missing_et' in event_data:
            self._add_missing_et(fig, event_data['missing_et'])
        
        # Add physics annotations
        if 'physics_info' in event_data:
            self._add_physics_annotations(fig, event_data['physics_info'])
        
        # Configure 3D scene
        self._configure_3d_scene(fig, event_data)
        
        # Save if requested
        if save_html:
            output_path = self.output_dir / save_html
            fig.write_html(str(output_path))
            logger.info(f"Saved 3D event visualization: {output_path}")
            return str(output_path)
        elif interactive:
            # Save with default name
            timestamp = int(time.time())
            output_path = self.output_dir / f"event_3d_{timestamp}.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved 3D event visualization: {output_path}")
            return str(output_path)
        
        return None
    
    def _add_detector_geometry(self, fig: go.Figure, geometry: Dict[str, Any]) -> None:
        """Add detector geometry to 3D plot."""
        
        # Barrel detector (cylindrical)
        if 'barrel' in geometry:
            barrel_config = geometry['barrel']
            radius = barrel_config.get('radius', 1.5)
            length = barrel_config.get('length', 6.0)
            
            # Create cylindrical surface
            theta = np.linspace(0, 2*np.pi, 50)
            z = np.linspace(-length/2, length/2, 20)
            
            theta_mesh, z_mesh = np.meshgrid(theta, z)
            x_mesh = radius * np.cos(theta_mesh)
            y_mesh = radius * np.sin(theta_mesh)
            
            fig.add_trace(go.Surface(
                x=x_mesh, y=y_mesh, z=z_mesh,
                colorscale='Viridis',
                opacity=0.2,
                showscale=False,
                name='Barrel Calorimeter',
                hovertemplate='Barrel Detector<br>R: %{x:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
        
        # Endcap detectors
        if 'endcaps' in geometry:
            endcap_config = geometry['endcaps']
            z_positions = endcap_config.get('z_positions', [-3.2, 3.2])
            inner_radius = endcap_config.get('inner_radius', 0.5)
            outer_radius = endcap_config.get('outer_radius', 2.0)
            
            for z_pos in z_positions:
                # Create disk surface
                r = np.linspace(inner_radius, outer_radius, 20)
                theta = np.linspace(0, 2*np.pi, 50)
                
                r_mesh, theta_mesh = np.meshgrid(r, theta)
                x_mesh = r_mesh * np.cos(theta_mesh)
                y_mesh = r_mesh * np.sin(theta_mesh)
                z_mesh = np.full_like(x_mesh, z_pos)
                
                fig.add_trace(go.Surface(
                    x=x_mesh, y=y_mesh, z=z_mesh,
                    colorscale='Plasma',
                    opacity=0.3,
                    showscale=False,
                    name=f'Endcap (z={z_pos}m)',
                    hovertemplate=f'Endcap z={z_pos:.1f}m<br>R: %{{x:.2f}}<extra></extra>'
                ))
    
    def _add_default_detector_geometry(self, fig: go.Figure) -> None:
        """Add default LHC detector geometry."""
        
        # Inner tracker (silicon)
        tracker_radius = 1.0
        self._add_cylindrical_detector(fig, tracker_radius, 5.0, 'Tracker', 
                                     self.detector_colors['tracker'], 0.1)
        
        # ECAL
        ecal_radius = 1.5
        self._add_cylindrical_detector(fig, ecal_radius, 6.0, 'ECAL',
                                     self.detector_colors['ecal'], 0.15)
        
        # HCAL
        hcal_radius = 2.0
        self._add_cylindrical_detector(fig, hcal_radius, 7.0, 'HCAL',
                                     self.detector_colors['hcal'], 0.15)
        
        # Muon chambers
        muon_radius = 3.5
        self._add_cylindrical_detector(fig, muon_radius, 10.0, 'Muon Chambers',
                                     self.detector_colors['muon_chamber'], 0.1)
    
    def _add_cylindrical_detector(
        self, 
        fig: go.Figure, 
        radius: float, 
        length: float, 
        name: str, 
        color: str,
        opacity: float
    ) -> None:
        """Add cylindrical detector component."""
        
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(-length/2, length/2, 15)
        
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)
        
        fig.add_trace(go.Surface(
            x=x_mesh, y=y_mesh, z=z_mesh,
            colorscale=[[0, color], [1, color]],
            opacity=opacity,
            showscale=False,
            name=name,
            hovertemplate=f'{name}<br>R: {radius:.1f}m<br>Z: %{{z:.2f}}m<extra></extra>'
        ))
    
    def _add_particle_tracks(self, fig: go.Figure, particles: List[Dict[str, Any]]) -> None:
        """Add particle tracks to 3D visualization."""
        
        for i, particle in enumerate(particles):
            # Get particle properties
            particle_type = particle.get('type', 'unknown')
            momentum = particle.get('momentum', [0, 0, 0])  # [px, py, pz]
            charge = particle.get('charge', 0)
            energy = particle.get('energy', 0)
            
            # Calculate track trajectory (simplified helical path)
            track_points = self._calculate_track_trajectory(momentum, charge)
            
            if track_points is not None:
                x_track, y_track, z_track = track_points
                
                # Color based on particle type
                color = self.particle_colors.get(particle_type, self.particle_colors['unknown'])
                
                # Track width based on energy
                line_width = max(2, min(8, energy / 10))
                
                fig.add_trace(go.Scatter3d(
                    x=x_track, y=y_track, z=z_track,
                    mode='lines+markers',
                    line=dict(color=color, width=line_width),
                    marker=dict(size=3, color=color),
                    name=f'{particle_type.capitalize()} (E={energy:.1f} GeV)',
                    hovertemplate=(f'{particle_type.capitalize()}<br>' +
                                 f'Energy: {energy:.1f} GeV<br>' +
                                 f'px: {momentum[0]:.1f} GeV<br>' +
                                 f'py: {momentum[1]:.1f} GeV<br>' +
                                 f'pz: {momentum[2]:.1f} GeV<extra></extra>')
                ))
    
    def _calculate_track_trajectory(
        self, 
        momentum: List[float], 
        charge: int,
        num_points: int = 100
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calculate helical track trajectory in magnetic field."""
        
        if len(momentum) < 3:
            return None
        
        px, py, pz = momentum
        pt = np.sqrt(px**2 + py**2)
        
        if pt == 0:
            return None
        
        # Magnetic field strength (Tesla) - typical LHC solenoid
        B = 3.8
        
        # Cyclotron radius (m)
        # R = pt (GeV) / (0.3 * B (T) * |charge|)
        if charge == 0:
            # Neutral particle - straight line
            phi0 = np.arctan2(py, px)
            eta = np.arcsinh(pz / pt)
            
            t = np.linspace(0, 3.0, num_points)  # 3 meters range
            x = t * np.cos(phi0)
            y = t * np.sin(phi0)
            z = t * np.sinh(eta)
            
            return x, y, z
        else:
            # Charged particle - helical trajectory
            R = pt / (0.3 * B * abs(charge))  # Cyclotron radius
            
            # Initial angle
            phi0 = np.arctan2(py, px)
            
            # Helix parameters
            omega = charge * 0.3 * B / pt  # Angular frequency
            
            # Parametric trajectory
            t = np.linspace(0, 2*np.pi/abs(omega), num_points)
            
            x = R * (np.cos(phi0 + omega * t) - np.cos(phi0))
            y = R * (np.sin(phi0 + omega * t) - np.sin(phi0))
            z = (pz / pt) * R * omega * t
            
            return x, y, z
    
    def _add_energy_deposits(self, fig: go.Figure, deposits: Dict[str, Any]) -> None:
        """Add calorimeter energy deposits visualization."""
        
        # ECAL deposits
        if 'ecal' in deposits:
            ecal_data = deposits['ecal']
            self._add_calorimeter_deposits(fig, ecal_data, 'ECAL', 1.5, '#4CAF50')
        
        # HCAL deposits
        if 'hcal' in deposits:
            hcal_data = deposits['hcal']
            self._add_calorimeter_deposits(fig, hcal_data, 'HCAL', 2.0, '#FF9800')
    
    def _add_calorimeter_deposits(
        self,
        fig: go.Figure,
        deposit_data: Dict[str, Any],
        detector_name: str,
        radius: float,
        base_color: str
    ) -> None:
        """Add energy deposit visualization for specific calorimeter."""
        
        if 'cells' not in deposit_data:
            return
        
        cells = deposit_data['cells']
        
        for cell in cells:
            eta = cell.get('eta', 0)
            phi = cell.get('phi', 0) 
            energy = cell.get('energy', 0)
            
            if energy <= 0:
                continue
            
            # Convert eta, phi to x, y, z coordinates
            theta = 2 * np.arctan(np.exp(-eta))  # eta to theta conversion
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            # Size and opacity based on energy
            marker_size = max(2, min(15, energy))
            marker_opacity = min(1.0, energy / 10.0)
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=base_color,
                    opacity=marker_opacity,
                    symbol='circle'
                ),
                name=f'{detector_name} Cell (E={energy:.2f} GeV)',
                hovertemplate=(f'{detector_name} Energy Deposit<br>' +
                             f'Energy: {energy:.2f} GeV<br>' +
                             f'eta: {eta:.3f}<br>' +
                             f'phi: {phi:.3f}<extra></extra>'),
                showlegend=False
            ))
    
    def _add_missing_et(self, fig: go.Figure, missing_et_data: Dict[str, Any]) -> None:
        """Add missing transverse energy visualization."""
        
        met_magnitude = missing_et_data.get('magnitude', 0)
        met_phi = missing_et_data.get('phi', 0)
        
        if met_magnitude <= 0:
            return
        
        # Create arrow showing missing ET direction
        # Place arrow at outer detector radius
        arrow_radius = 4.0
        
        x_start = 0
        y_start = 0
        z_start = 0
        
        x_end = arrow_radius * np.cos(met_phi)
        y_end = arrow_radius * np.sin(met_phi)
        z_end = 0  # MET is in transverse plane
        
        # Arrow shaft
        fig.add_trace(go.Scatter3d(
            x=[x_start, x_end], 
            y=[y_start, y_end], 
            z=[z_start, z_end],
            mode='lines+markers',
            line=dict(color=self.particle_colors['missing_et'], width=8),
            marker=dict(size=[5, 10], color=self.particle_colors['missing_et']),
            name=f'Missing ET ({met_magnitude:.1f} GeV)',
            hovertemplate=(f'Missing Transverse Energy<br>' +
                         f'Magnitude: {met_magnitude:.1f} GeV<br>' +
                         f'phi: {met_phi:.3f}<extra></extra>')
        ))
    
    def _add_physics_annotations(self, fig: go.Figure, physics_info: Dict[str, Any]) -> None:
        """Add physics analysis annotations."""
        
        # Invariant mass annotations
        if 'invariant_masses' in physics_info:
            for mass_info in physics_info['invariant_masses']:
                mass_value = mass_info.get('mass', 0)
                particles = mass_info.get('particles', [])
                
                if len(particles) >= 2:
                    # Add annotation at midpoint between particles
                    x_mid = np.mean([p.get('x', 0) for p in particles])
                    y_mid = np.mean([p.get('y', 0) for p in particles])
                    z_mid = np.mean([p.get('z', 0) for p in particles])
                    
                    fig.add_trace(go.Scatter3d(
                        x=[x_mid], y=[y_mid], z=[z_mid],
                        mode='markers+text',
                        marker=dict(size=8, color='red', symbol='diamond'),
                        text=[f'M = {mass_value:.1f} GeV'],
                        textposition='top center',
                        name=f'Invariant Mass: {mass_value:.1f} GeV',
                        showlegend=False
                    ))
    
    def _configure_3d_scene(self, fig: go.Figure, event_data: Dict[str, Any]) -> None:
        """Configure 3D scene layout and styling."""
        
        # Get event information for title
        event_id = event_data.get('event_id', 'Unknown')
        run_number = event_data.get('run_number', 0)
        
        title_text = f"Event {event_id} (Run {run_number})"
        if 'physics_info' in event_data:
            physics_info = event_data['physics_info']
            if 'total_energy' in physics_info:
                title_text += f" - Total Energy: {physics_info['total_energy']:.1f} GeV"
        
        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            scene=dict(
                xaxis=dict(
                    title='X (meters)',
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[-5, 5]
                ),
                yaxis=dict(
                    title='Y (meters)',
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[-5, 5]
                ),
                zaxis=dict(
                    title='Z (meters)',
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[-6, 6]
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube',
                bgcolor='white'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=700
        )
    
    def create_detector_cross_section(
        self,
        event_data: Dict[str, Any],
        plane: str = 'xy',
        detector_config: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create 2D cross-section view of detector with event overlay.
        
        Args:
            event_data: Event data
            plane: Cross-section plane ('xy', 'xz', 'yz')
            detector_config: Detector geometry configuration
            save_path: Path to save image
            
        Returns:
            Path to saved image
        """
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for cross-section visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw detector components
        self._draw_detector_cross_section(ax, plane, detector_config)
        
        # Add particle tracks
        if 'particles' in event_data:
            self._draw_particle_tracks_2d(ax, event_data['particles'], plane)
        
        # Add energy deposits
        if 'energy_deposits' in event_data:
            self._draw_energy_deposits_2d(ax, event_data['energy_deposits'], plane)
        
        # Configure plot
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{plane[0].upper()} (meters)', fontsize=12)
        ax.set_ylabel(f'{plane[1].upper()} (meters)', fontsize=12)
        
        event_id = event_data.get('event_id', 'Unknown')
        ax.set_title(f'Event {event_id} - {plane.upper()} Cross Section', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            output_path = self.output_dir / save_path
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved cross-section view: {output_path}")
            return str(output_path)
        
        plt.close()
        return None
    
    def _draw_detector_cross_section(
        self, 
        ax, 
        plane: str, 
        detector_config: Optional[Dict[str, Any]]
    ) -> None:
        """Draw detector components in cross-section."""
        
        # Default LHC detector radii
        detector_radii = {
            'beam_pipe': 0.02,
            'tracker': 1.0,
            'ecal': 1.5,
            'hcal': 2.0,
            'muon': 3.5
        }
        
        detector_colors = {
            'beam_pipe': '#795548',
            'tracker': '#2196F3',
            'ecal': '#4CAF50',
            'hcal': '#FF9800',
            'muon': '#9C27B0'
        }
        
        if detector_config:
            detector_radii.update(detector_config.get('radii', {}))
            detector_colors.update(detector_config.get('colors', {}))
        
        # Draw cylindrical detectors as circles (for xy plane) or rectangles (for xz, yz)
        for detector_name, radius in detector_radii.items():
            color = detector_colors[detector_name]
            
            if plane == 'xy':
                # Circular cross-section
                circle = plt.Circle((0, 0), radius, fill=False, 
                                  color=color, linewidth=2, label=detector_name.title())
                ax.add_patch(circle)
            else:
                # Rectangular cross-section for longitudinal view
                height = 6.0 if detector_name != 'beam_pipe' else 0.04
                rect = plt.Rectangle((-radius, -height/2), 2*radius, height,
                                   fill=False, color=color, linewidth=2, 
                                   label=detector_name.title())
                ax.add_patch(rect)
        
        # Set plot limits
        max_radius = max(detector_radii.values()) * 1.2
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
    
    def _draw_particle_tracks_2d(
        self, 
        ax, 
        particles: List[Dict[str, Any]], 
        plane: str
    ) -> None:
        """Draw particle tracks in 2D cross-section."""
        
        for particle in particles:
            momentum = particle.get('momentum', [0, 0, 0])
            particle_type = particle.get('type', 'unknown')
            charge = particle.get('charge', 0)
            
            if len(momentum) < 3:
                continue
            
            # Get trajectory points
            track_points = self._calculate_track_trajectory(momentum, charge, 50)
            if track_points is None:
                continue
            
            x_track, y_track, z_track = track_points
            
            # Select coordinates based on plane
            if plane == 'xy':
                x_plot, y_plot = x_track, y_track
            elif plane == 'xz':
                x_plot, y_plot = x_track, z_track
            elif plane == 'yz':
                x_plot, y_plot = y_track, z_track
            else:
                continue
            
            color = self.particle_colors.get(particle_type, self.particle_colors['unknown'])
            
            ax.plot(x_plot, y_plot, color=color, linewidth=2, 
                   label=f'{particle_type.capitalize()}', alpha=0.8)
    
    def _draw_energy_deposits_2d(
        self, 
        ax, 
        deposits: Dict[str, Any], 
        plane: str
    ) -> None:
        """Draw energy deposits in 2D cross-section."""
        
        for detector_name, deposit_data in deposits.items():
            if 'cells' not in deposit_data:
                continue
            
            cells = deposit_data['cells']
            
            for cell in cells:
                eta = cell.get('eta', 0)
                phi = cell.get('phi', 0)
                energy = cell.get('energy', 0)
                
                if energy <= 0:
                    continue
                
                # Convert to cartesian coordinates
                theta = 2 * np.arctan(np.exp(-eta))
                radius = 1.5 if detector_name == 'ecal' else 2.0
                
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                
                # Select coordinates for plot
                if plane == 'xy':
                    x_plot, y_plot = x, y
                elif plane == 'xz':
                    x_plot, y_plot = x, z
                elif plane == 'yz':
                    x_plot, y_plot = y, z
                else:
                    continue
                
                # Size based on energy
                marker_size = max(10, min(100, energy * 10))
                
                ax.scatter(x_plot, y_plot, s=marker_size, alpha=0.6,
                          c=self.detector_colors.get(detector_name, 'gray'),
                          label=f'{detector_name.upper()} Deposit')


def create_sample_event_data() -> Dict[str, Any]:
    """Create sample event data for demonstration."""
    
    return {
        'event_id': 'demo_001',
        'run_number': 12345,
        'particles': [
            {
                'type': 'electron',
                'momentum': [50.0, 30.0, 20.0],  # px, py, pz in GeV
                'energy': 65.0,
                'charge': -1
            },
            {
                'type': 'muon',
                'momentum': [-30.0, 40.0, -15.0],
                'energy': 55.0,
                'charge': 1
            },
            {
                'type': 'photon',
                'momentum': [25.0, -35.0, 10.0],
                'energy': 45.0,
                'charge': 0
            }
        ],
        'energy_deposits': {
            'ecal': {
                'cells': [
                    {'eta': 1.2, 'phi': 0.5, 'energy': 45.0},
                    {'eta': -0.8, 'phi': 2.1, 'energy': 15.0},
                    {'eta': 0.3, 'phi': -1.2, 'energy': 8.5}
                ]
            },
            'hcal': {
                'cells': [
                    {'eta': 1.0, 'phi': 0.6, 'energy': 25.0},
                    {'eta': -0.9, 'phi': 2.0, 'energy': 12.0}
                ]
            }
        },
        'missing_et': {
            'magnitude': 35.0,
            'phi': 3.8
        },
        'physics_info': {
            'total_energy': 165.0,
            'invariant_masses': [
                {
                    'mass': 91.2,
                    'particles': [
                        {'x': 1.2, 'y': 0.8, 'z': 0.5},
                        {'x': -0.9, 'y': 1.1, 'z': -0.3}
                    ]
                }
            ]
        }
    }


# Convenience functions
def visualize_event(
    event_data: Dict[str, Any],
    output_path: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """Visualize particle physics event in 3D."""
    
    visualizer = Interactive3DVisualizer()
    return visualizer.visualize_event_3d(
        event_data, 
        save_html=output_path,
        **kwargs
    )

def create_detector_view(
    event_data: Dict[str, Any],
    plane: str = 'xy',
    output_path: Optional[str] = None
) -> Optional[str]:
    """Create 2D detector cross-section view."""
    
    visualizer = Interactive3DVisualizer()
    return visualizer.create_detector_cross_section(
        event_data,
        plane=plane,
        save_path=output_path
    )