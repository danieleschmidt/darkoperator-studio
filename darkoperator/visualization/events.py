"""
Enhanced Event Visualization Utilities.

Provides advanced visualization capabilities for particle physics events,
including interactive 3D displays, analysis plots, and physics-aware visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, List, Any, Tuple
import logging

# Import the new interactive visualizer
try:
    from .interactive import Interactive3DVisualizer, create_sample_event_data
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


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


def create_physics_analysis_dashboard(
    events: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create comprehensive physics analysis dashboard.
    
    Args:
        events: List of event dictionaries with physics data
        save_path: Path to save HTML dashboard
        
    Returns:
        Plotly figure object
    """
    
    if not events:
        logger.warning("No events provided for dashboard")
        return None
    
    # Create subplot layout
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Energy Distribution', 'Transverse Momentum', 'Pseudorapidity',
            'Missing ET', 'Invariant Mass', 'Particle Multiplicity',
            'Energy vs Eta', 'Momentum Balance', 'Event Timeline'
        ],
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Extract physics quantities from events
    energies, pts, etas, missing_ets = [], [], [], []
    invariant_masses, multiplicities, event_times = [], [], []
    
    for i, event in enumerate(events):
        if 'particles' in event:
            particles = event['particles']
            event_energies = [p.get('energy', 0) for p in particles]
            event_pts = [np.sqrt(p.get('momentum', [0,0,0])[0]**2 + p.get('momentum', [0,0,0])[1]**2) 
                        for p in particles]
            event_etas = [calculate_pseudorapidity(p.get('momentum', [0,0,0])) for p in particles]
            
            energies.extend(event_energies)
            pts.extend(event_pts)
            etas.extend(event_etas)
            multiplicities.append(len(particles))
        else:
            multiplicities.append(0)
        
        # Missing ET
        if 'missing_et' in event:
            missing_ets.append(event['missing_et'].get('magnitude', 0))
        else:
            missing_ets.append(0)
        
        # Invariant mass (simplified - use total energy if available)
        if 'physics_info' in event and 'total_energy' in event['physics_info']:
            invariant_masses.append(event['physics_info']['total_energy'])
        else:
            invariant_masses.append(sum([p.get('energy', 0) for p in event.get('particles', [])]))
        
        event_times.append(i)
    
    # Row 1: Basic distributions
    fig.add_trace(go.Histogram(x=energies, nbinsx=30, name='Energy', showlegend=False,
                              marker_color='blue', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=pts, nbinsx=30, name='pT', showlegend=False,
                              marker_color='green', opacity=0.7), row=1, col=2)
    fig.add_trace(go.Histogram(x=etas, nbinsx=30, name='Eta', showlegend=False,
                              marker_color='red', opacity=0.7), row=1, col=3)
    
    # Row 2: Physics quantities
    fig.add_trace(go.Histogram(x=missing_ets, nbinsx=20, name='MET', showlegend=False,
                              marker_color='orange', opacity=0.7), row=2, col=1)
    fig.add_trace(go.Histogram(x=invariant_masses, nbinsx=25, name='Mass', showlegend=False,
                              marker_color='purple', opacity=0.7), row=2, col=2)
    fig.add_trace(go.Bar(x=list(range(len(multiplicities))), y=multiplicities, 
                        name='Multiplicity', showlegend=False,
                        marker_color='cyan', opacity=0.7), row=2, col=3)
    
    # Row 3: Correlations and trends
    fig.add_trace(go.Scatter(x=etas, y=energies, mode='markers', name='E vs η',
                            marker=dict(color='blue', size=4, opacity=0.6),
                            showlegend=False), row=3, col=1)
    
    # Momentum balance (px vs py)
    px_values = []
    py_values = []
    for event in events:
        if 'particles' in event:
            for particle in event['particles']:
                momentum = particle.get('momentum', [0, 0, 0])
                px_values.append(momentum[0])
                py_values.append(momentum[1])
    
    fig.add_trace(go.Scatter(x=px_values, y=py_values, mode='markers', name='p balance',
                            marker=dict(color='green', size=4, opacity=0.6),
                            showlegend=False), row=3, col=2)
    
    # Event timeline
    fig.add_trace(go.Scatter(x=event_times, y=missing_ets, mode='lines+markers',
                            name='MET Timeline', line=dict(color='red'),
                            showlegend=False), row=3, col=3)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Physics Analysis Dashboard ({len(events)} Events)',
            'x': 0.5,
            'font': {'size': 16}
        },
        height=900,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Energy (GeV)", row=1, col=1)
    fig.update_xaxes(title_text="pT (GeV)", row=1, col=2)
    fig.update_xaxes(title_text="η", row=1, col=3)
    fig.update_xaxes(title_text="Missing ET (GeV)", row=2, col=1)
    fig.update_xaxes(title_text="Invariant Mass (GeV)", row=2, col=2)
    fig.update_xaxes(title_text="Event Number", row=2, col=3)
    fig.update_xaxes(title_text="η", row=3, col=1)
    fig.update_xaxes(title_text="px (GeV)", row=3, col=2)
    fig.update_xaxes(title_text="Event Number", row=3, col=3)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Multiplicity", row=2, col=3)
    fig.update_yaxes(title_text="Energy (GeV)", row=3, col=1)
    fig.update_yaxes(title_text="py (GeV)", row=3, col=2)
    fig.update_yaxes(title_text="Missing ET (GeV)", row=3, col=3)
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved physics dashboard: {save_path}")
    
    return fig


def visualize_anomaly_detection_results(
    events: List[Dict[str, Any]],
    anomaly_scores: List[float],
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Visualize anomaly detection results with physics context.
    
    Args:
        events: List of event dictionaries
        anomaly_scores: Anomaly scores for each event
        threshold: Anomaly threshold
        save_path: Path to save visualization
        
    Returns:
        Plotly figure object
    """
    
    if len(events) != len(anomaly_scores):
        logger.error("Number of events and scores must match")
        return None
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Anomaly Score Distribution',
            'Score vs Energy',
            'Score vs Missing ET',
            'ROC-like Analysis'
        ]
    )
    
    # Extract physics quantities
    total_energies = []
    missing_ets = []
    
    for event in events:
        # Total energy
        if 'physics_info' in event and 'total_energy' in event['physics_info']:
            total_energies.append(event['physics_info']['total_energy'])
        else:
            # Calculate from particles
            energy = sum([p.get('energy', 0) for p in event.get('particles', [])])
            total_energies.append(energy)
        
        # Missing ET
        if 'missing_et' in event:
            missing_ets.append(event['missing_et'].get('magnitude', 0))
        else:
            missing_ets.append(0)
    
    # Identify anomalies
    anomalies = np.array(anomaly_scores) > threshold
    normal_indices = np.where(~anomalies)[0]
    anomaly_indices = np.where(anomalies)[0]
    
    # Plot 1: Score distribution
    fig.add_trace(go.Histogram(
        x=anomaly_scores, nbinsx=30, name='All Events',
        marker_color='blue', opacity=0.7
    ), row=1, col=1)
    
    # Add threshold line
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold}", row=1, col=1)
    
    # Plot 2: Score vs Energy
    fig.add_trace(go.Scatter(
        x=[total_energies[i] for i in normal_indices],
        y=[anomaly_scores[i] for i in normal_indices],
        mode='markers', name='Normal Events',
        marker=dict(color='blue', size=6, opacity=0.6)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[total_energies[i] for i in anomaly_indices],
        y=[anomaly_scores[i] for i in anomaly_indices],
        mode='markers', name='Anomalous Events',
        marker=dict(color='red', size=8, symbol='diamond', opacity=0.8)
    ), row=1, col=2)
    
    # Plot 3: Score vs Missing ET
    fig.add_trace(go.Scatter(
        x=[missing_ets[i] for i in normal_indices],
        y=[anomaly_scores[i] for i in normal_indices],
        mode='markers', name='Normal Events',
        marker=dict(color='blue', size=6, opacity=0.6),
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[missing_ets[i] for i in anomaly_indices],
        y=[anomaly_scores[i] for i in anomaly_indices],
        mode='markers', name='Anomalous Events',
        marker=dict(color='red', size=8, symbol='diamond', opacity=0.8),
        showlegend=False
    ), row=2, col=1)
    
    # Plot 4: ROC-like curve
    thresholds = np.linspace(0, 1, 100)
    tpr_values = []  # True positive rate (sensitivity)
    fpr_values = []  # False positive rate (1 - specificity)
    
    # For demonstration, assume we know true anomalies (top 10% by score)
    true_anomalies = np.argsort(anomaly_scores)[-len(anomaly_scores)//10:]
    true_anomaly_mask = np.zeros(len(anomaly_scores), dtype=bool)
    true_anomaly_mask[true_anomalies] = True
    
    for thresh in thresholds:
        predicted_anomalies = np.array(anomaly_scores) > thresh
        
        tp = np.sum(predicted_anomalies & true_anomaly_mask)
        fp = np.sum(predicted_anomalies & ~true_anomaly_mask)
        tn = np.sum(~predicted_anomalies & ~true_anomaly_mask)
        fn = np.sum(~predicted_anomalies & true_anomaly_mask)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    fig.add_trace(go.Scatter(
        x=fpr_values, y=tpr_values, mode='lines',
        name='ROC Curve', line=dict(color='purple', width=3)
    ), row=2, col=2)
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier', line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Anomaly Detection Analysis ({np.sum(anomalies)} anomalies detected)',
            'x': 0.5,
            'font': {'size': 16}
        },
        height=800
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_xaxes(title_text="Total Energy (GeV)", row=1, col=2)
    fig.update_xaxes(title_text="Missing ET (GeV)", row=2, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=1, col=2)
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved anomaly analysis: {save_path}")
    
    return fig


def create_model_performance_comparison(
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create comparison visualization for model performance metrics.
    
    Args:
        model_results: Dictionary mapping model names to performance metrics
        save_path: Path to save visualization
        
    Returns:
        Plotly figure object
    """
    
    if not model_results:
        logger.warning("No model results provided")
        return None
    
    # Extract data for plotting
    model_names = list(model_results.keys())
    metrics = ['inference_time_ms', 'throughput_events_per_sec', 'memory_usage_mb', 'physics_score']
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Inference Time (ms)',
            'Throughput (Events/sec)',
            'Memory Usage (MB)',
            'Physics Accuracy Score'
        ]
    )
    
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        values = [model_results[model].get(metric, 0) for model in model_names]
        
        # Use log scale for throughput
        if metric == 'throughput_events_per_sec':
            fig.add_trace(go.Bar(
                x=model_names, y=values, name=metric,
                marker_color=px.colors.qualitative.Set1[i],
                showlegend=False
            ), row=row, col=col)
            fig.update_yaxes(type="log", row=row, col=col)
        else:
            fig.add_trace(go.Bar(
                x=model_names, y=values, name=metric,
                marker_color=px.colors.qualitative.Set1[i],
                showlegend=False
            ), row=row, col=col)
        
        # Add value annotations
        for j, (model, value) in enumerate(zip(model_names, values)):
            fig.add_annotation(
                x=j, y=value,
                text=f'{value:.2f}',
                showarrow=False,
                yshift=10,
                row=row, col=col
            )
    
    fig.update_layout(
        title={
            'text': 'Model Performance Comparison',
            'x': 0.5,
            'font': {'size': 16}
        },
        height=700,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved model comparison: {save_path}")
    
    return fig


def calculate_pseudorapidity(momentum: List[float]) -> float:
    """Calculate pseudorapidity from momentum vector."""
    if len(momentum) < 3:
        return 0.0
    
    px, py, pz = momentum[0], momentum[1], momentum[2]
    pt = np.sqrt(px**2 + py**2)
    
    if pt == 0:
        return 0.0
    
    return np.arcsinh(pz / pt)


def create_interactive_event_explorer(
    events: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Create interactive 3D event explorer for multiple events.
    
    Args:
        events: List of event dictionaries
        output_dir: Output directory for HTML files
        
    Returns:
        List of paths to created HTML files
    """
    
    if not INTERACTIVE_AVAILABLE:
        logger.error("Interactive visualizer not available")
        return []
    
    visualizer = Interactive3DVisualizer(output_dir)
    created_files = []
    
    for i, event in enumerate(events):
        output_path = f"event_{i+1}_3d.html"
        
        try:
            file_path = visualizer.visualize_event_3d(
                event,
                interactive=True,
                save_html=output_path
            )
            
            if file_path:
                created_files.append(file_path)
                
        except Exception as e:
            logger.error(f"Failed to create visualization for event {i+1}: {e}")
    
    logger.info(f"Created {len(created_files)} interactive event visualizations")
    return created_files


# Enhanced convenience functions
def create_comprehensive_event_analysis(
    events: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Create comprehensive analysis suite for events."""
    
    results = {}
    
    try:
        # Physics dashboard
        dashboard_fig = create_physics_analysis_dashboard(events)
        if dashboard_fig and output_dir:
            dashboard_path = f"{output_dir}/physics_dashboard.html"
            dashboard_fig.write_html(dashboard_path)
            results['dashboard'] = dashboard_path
        
        # Interactive 3D events
        if INTERACTIVE_AVAILABLE:
            interactive_files = create_interactive_event_explorer(events, output_dir)
            results['interactive_events'] = interactive_files
        
        logger.info(f"Created comprehensive analysis with {len(results)} components")
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
    
    return results