"""Interface to LHC Open Data Portal."""

import os
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def list_opendata_datasets(
    experiment: Optional[List[str]] = None,
    years: Optional[List[int]] = None, 
    data_type: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    List available LHC Open Data datasets.
    
    Args:
        experiment: Filter by experiment (e.g., ['ATLAS', 'CMS'])
        years: Filter by data-taking years
        data_type: Filter by data type (e.g., 'jets', 'muons')
        
    Returns:
        List of dataset metadata dictionaries
    """
    # Mock implementation - in reality would query CERN Open Data Portal API
    mock_datasets = [
        {
            "name": "cms-jets-13tev-2016",
            "experiment": "CMS",
            "year": 2016,
            "energy": "13 TeV",
            "data_type": "jets",
            "size_gb": 15.2,
            "n_events": 1_000_000,
            "url": "http://opendata.cern.ch/record/12345",
            "description": "CMS jet events from 2016 data taking"
        },
        {
            "name": "atlas-muons-13tev-2015",
            "experiment": "ATLAS", 
            "year": 2015,
            "energy": "13 TeV",
            "data_type": "muons",
            "size_gb": 8.7,
            "n_events": 500_000,
            "url": "http://opendata.cern.ch/record/12346",
            "description": "ATLAS muon events from 2015 data taking"
        },
        {
            "name": "cms-run3-jets",
            "experiment": "CMS",
            "year": 2022,
            "energy": "13.6 TeV", 
            "data_type": "jets",
            "size_gb": 25.1,
            "n_events": 2_000_000,
            "url": "http://opendata.cern.ch/record/12347",
            "description": "CMS jet events from Run 3"
        }
    ]
    
    # Apply filters
    filtered_datasets = mock_datasets
    
    if experiment:
        filtered_datasets = [d for d in filtered_datasets if d["experiment"] in experiment]
    
    if years:
        filtered_datasets = [d for d in filtered_datasets if d["year"] in years]
        
    if data_type:
        filtered_datasets = [d for d in filtered_datasets if data_type.lower() in d["data_type"].lower()]
    
    logger.info(f"Found {len(filtered_datasets)} matching datasets")
    return filtered_datasets


def download_dataset(
    dataset_name: str,
    cache_dir: str = "./data",
    max_events: Optional[int] = None,
    force_download: bool = False
) -> str:
    """
    Download and cache LHC Open Data dataset.
    
    Args:
        dataset_name: Name of dataset to download
        cache_dir: Local directory for caching
        max_events: Maximum number of events to download
        force_download: Force re-download even if cached
        
    Returns:
        Path to downloaded/cached dataset
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already cached
    cached_file = cache_dir / f"{dataset_name}.npz"
    if cached_file.exists() and not force_download:
        logger.info(f"Using cached dataset: {cached_file}")
        return str(cached_file)
    
    # Find dataset metadata
    all_datasets = list_opendata_datasets()
    dataset_info = None
    for ds in all_datasets:
        if ds["name"] == dataset_name:
            dataset_info = ds
            break
    
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Size: {dataset_info['size_gb']:.1f} GB, Events: {dataset_info['n_events']:,}")
    
    # Mock download - generate synthetic data for demonstration
    n_events = min(max_events or dataset_info["n_events"], dataset_info["n_events"])
    synthetic_data = _generate_mock_data(dataset_info["data_type"], n_events)
    
    # Save to cache
    np.savez_compressed(cached_file, **synthetic_data)
    logger.info(f"Dataset cached to: {cached_file}")
    
    return str(cached_file)


def load_opendata(
    dataset_name: str,
    max_events: Optional[int] = None,
    cache_dir: str = "./data"
) -> np.ndarray:
    """
    Load LHC Open Data events.
    
    Args:
        dataset_name: Name of dataset
        max_events: Maximum events to load
        cache_dir: Cache directory
        
    Returns:
        Array of events with 4-vector information
    """
    # Download/cache dataset
    dataset_path = download_dataset(dataset_name, cache_dir, max_events)
    
    # Load from cache
    data = np.load(dataset_path)
    events = data["events"]
    
    if max_events and len(events) > max_events:
        events = events[:max_events]
    
    logger.info(f"Loaded {len(events)} events from {dataset_name}")
    return events


def _generate_mock_data(data_type: str, n_events: int) -> Dict[str, np.ndarray]:
    """Generate mock LHC data for demonstration."""
    np.random.seed(42)  # Reproducible
    
    if data_type == "jets":
        # Generate jet 4-vectors (E, px, py, pz)
        # Typical jet energies from 20 GeV to 1 TeV
        energy = np.random.lognormal(mean=4.0, sigma=1.5, size=(n_events, 4))
        energy = np.clip(energy, 20, 1000)  # Realistic energy range
        
        # Generate momentum components
        pt = energy * np.random.beta(0.8, 0.2, size=(n_events, 4))  # pT < E
        eta = np.random.normal(0, 2.5, size=(n_events, 4))  # Pseudorapidity
        phi = np.random.uniform(-np.pi, np.pi, size=(n_events, 4))  # Azimuthal angle
        
        # Convert to Cartesian
        px = pt * np.cos(phi)
        py = pt * np.sin(phi) 
        pz = pt * np.sinh(eta)
        
        # Ensure physical 4-vectors (E^2 >= p^2)
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        energy = np.maximum(energy, p_mag + 0.1)  # Small mass assumption
        
        events = np.stack([energy, px, py, pz], axis=-1)
        
    elif data_type == "muons":
        # Generate muon 4-vectors (typically lower energy than jets)
        energy = np.random.lognormal(mean=3.0, sigma=1.0, size=(n_events, 2))
        energy = np.clip(energy, 5, 200)
        
        pt = energy * np.random.beta(0.9, 0.1, size=(n_events, 2))
        eta = np.random.normal(0, 2.1, size=(n_events, 2))  # Muon detector coverage
        phi = np.random.uniform(-np.pi, np.pi, size=(n_events, 2))
        
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        
        # Add muon mass
        muon_mass = 0.106  # GeV
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        energy = np.sqrt(p_mag**2 + muon_mass**2)
        
        events = np.stack([energy, px, py, pz], axis=-1)
        
    else:
        # Generic particle events
        events = np.random.normal(0, 50, size=(n_events, 4, 4))
        events[:, :, 0] = np.abs(events[:, :, 0]) + 10  # Positive energy
    
    return {"events": events.astype(np.float32)}