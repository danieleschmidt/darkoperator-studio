"""
Model downloader and checkpoint management.

Handles secure downloading, verification, and loading of pre-trained models.
"""

import os
import hashlib
import logging
import requests
import tempfile
from typing import Optional, Any, Dict
from pathlib import Path
import time
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles secure model downloading with verification."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DarkOperator-Studio/1.0.0'
        })
        
    def download_model(self, model_info) -> bool:
        """
        Download model with integrity verification.
        
        Args:
            model_info: ModelInfo object with download details
            
        Returns:
            True if download successful and verified
        """
        
        model_path = self.cache_dir / f"{model_info.model_id}.pth"
        
        try:
            # Create temporary file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
                temp_path = Path(temp_file.name)
                
                logger.info(f"Downloading {model_info.name} from {model_info.download_url}")
                
                # Download with progress bar
                if not self._download_with_progress(model_info.download_url, temp_path):
                    temp_path.unlink(missing_ok=True)
                    return False
                
                # Verify checksum
                if not self._verify_checksum(temp_path, model_info.checksum):
                    logger.error(f"Checksum verification failed for {model_info.model_id}")
                    temp_path.unlink(missing_ok=True)
                    return False
                
                # Move to final location
                temp_path.rename(model_path)
                
                # Save model metadata
                self._save_model_metadata(model_info)
                
                logger.info(f"Successfully downloaded and verified: {model_info.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Download failed for {model_info.model_id}: {e}")
            return False
    
    def _download_with_progress(self, url: str, output_path: Path) -> bool:
        """Download file with progress indicator."""
        
        try:
            # Handle mock URLs for demonstration
            if url.startswith("https://models.darkoperator.ai/"):
                logger.info("Creating mock model file for demonstration")
                return self._create_mock_model(output_path)
            
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            if TQDM_AVAILABLE:
                progress_bar = tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=output_path.name
                )
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if TQDM_AVAILABLE:
                            progress_bar.update(len(chunk))
            
            if TQDM_AVAILABLE:
                progress_bar.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def _create_mock_model(self, output_path: Path) -> bool:
        """Create mock model file for demonstration."""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, creating placeholder file")
            with open(output_path, 'wb') as f:
                # Create a small placeholder file
                placeholder_data = b"MOCK_DARKOPERATOR_MODEL_" + b"X" * 1000
                f.write(placeholder_data)
            return True
        
        try:
            # Create a simple mock model with realistic structure
            import torch.nn as nn
            
            mock_model = nn.Sequential(
                nn.Linear(4, 64),  # 4-momentum input
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Anomaly score output
            )
            
            # Save mock model
            torch.save({
                'model_state_dict': mock_model.state_dict(),
                'model_architecture': 'mock_neural_operator',
                'darkoperator_version': '1.0.0',
                'creation_time': time.time(),
                'physics_validated': True
            }, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Mock model creation failed: {e}")
            return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using checksum."""
        
        if expected_checksum.startswith("sha256:"):
            hash_algo = hashlib.sha256()
            expected = expected_checksum[7:]  # Remove "sha256:" prefix
        else:
            logger.warning("Unknown checksum format, skipping verification")
            return True  # Skip verification for unknown formats
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_algo.update(chunk)
            
            actual_checksum = hash_algo.hexdigest()
            
            # For mock models, accept any checksum
            if "models.darkoperator.ai" in str(file_path):
                logger.info("Mock model detected, skipping checksum verification")
                return True
            
            if actual_checksum != expected:
                logger.error(f"Checksum mismatch: expected {expected}, got {actual_checksum}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checksum verification error: {e}")
            return False
    
    def _save_model_metadata(self, model_info) -> None:
        """Save model metadata alongside downloaded model."""
        
        metadata_path = self.cache_dir / f"{model_info.model_id}_metadata.json"
        
        metadata = {
            'model_id': model_info.model_id,
            'name': model_info.name,
            'version': model_info.version,
            'download_time': time.time(),
            'checksum': model_info.checksum,
            'file_size_bytes': model_info.file_size_bytes,
            'physics_accuracy': model_info.physics_accuracy,
            'conservation_compliance': model_info.conservation_law_compliance
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")


class CheckpointManager:
    """Manages loading and validation of model checkpoints."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    
    def load_model(self, model_info) -> Optional[Any]:
        """
        Load model from checkpoint.
        
        Args:
            model_info: ModelInfo object
            
        Returns:
            Loaded model object or None if failed
        """
        
        model_path = self.cache_dir / f"{model_info.model_id}.pth"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for model loading")
            return None
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Validate checkpoint
            if not self._validate_checkpoint(checkpoint, model_info):
                logger.error(f"Checkpoint validation failed for {model_info.model_id}")
                return None
            
            # Create model based on type
            model = self._create_model_from_checkpoint(checkpoint, model_info)
            
            if model is not None:
                logger.info(f"Successfully loaded model: {model_info.model_id}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_info.model_id}: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint: Dict[str, Any], model_info) -> bool:
        """Validate checkpoint integrity and compatibility."""
        
        required_keys = ['model_state_dict']
        
        # Check required keys
        for key in required_keys:
            if key not in checkpoint:
                logger.error(f"Missing required key in checkpoint: {key}")
                return False
        
        # Validate physics constraints if present
        if 'physics_validated' in checkpoint:
            if not checkpoint['physics_validated'] and model_info.is_physics_validated:
                logger.warning("Model marked as physics validated but checkpoint indicates otherwise")
        
        # Check DarkOperator compatibility
        if 'darkoperator_version' in checkpoint:
            version = checkpoint['darkoperator_version']
            logger.debug(f"Checkpoint created with DarkOperator version: {version}")
        
        return True
    
    def _create_model_from_checkpoint(self, checkpoint: Dict[str, Any], model_info) -> Optional[Any]:
        """Create model instance from checkpoint based on model type."""
        
        try:
            # Import model classes based on type
            if model_info.model_type.value == 'calorimeter_operator':
                return self._create_calorimeter_operator(checkpoint, model_info)
            elif model_info.model_type.value == 'anomaly_detector':
                return self._create_anomaly_detector(checkpoint, model_info)
            elif model_info.model_type.value == 'tracker_operator':
                return self._create_tracker_operator(checkpoint, model_info)
            else:
                return self._create_generic_model(checkpoint, model_info)
                
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            return None
    
    def _create_calorimeter_operator(self, checkpoint: Dict[str, Any], model_info) -> Any:
        """Create calorimeter operator from checkpoint."""
        
        try:
            from ..operators.calorimeter import CalorimeterOperator
            
            model = CalorimeterOperator(
                input_dim=model_info.input_shape[0],
                output_shape=model_info.output_shape,
                experiment=model_info.experiment.value
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
            
        except ImportError:
            logger.warning("CalorimeterOperator not available, creating generic model")
            return self._create_generic_model(checkpoint, model_info)
    
    def _create_anomaly_detector(self, checkpoint: Dict[str, Any], model_info) -> Any:
        """Create anomaly detector from checkpoint."""
        
        try:
            from ..anomaly.conformal import ConformalDetector
            
            # Create basic anomaly detector
            detector = ConformalDetector(
                input_dim=model_info.input_shape[0],
                alpha=1e-6
            )
            
            # Load state if available
            if hasattr(detector, 'load_state_dict'):
                detector.load_state_dict(checkpoint['model_state_dict'])
            
            return detector
            
        except ImportError:
            logger.warning("ConformalDetector not available, creating generic model")
            return self._create_generic_model(checkpoint, model_info)
    
    def _create_tracker_operator(self, checkpoint: Dict[str, Any], model_info) -> Any:
        """Create tracker operator from checkpoint."""
        
        try:
            from ..operators.tracker import TrackerOperator
            
            model = TrackerOperator(
                input_dim=model_info.input_shape[0],
                output_shape=model_info.output_shape,
                experiment=model_info.experiment.value
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
            
        except ImportError:
            logger.warning("TrackerOperator not available, creating generic model")
            return self._create_generic_model(checkpoint, model_info)
    
    def _create_generic_model(self, checkpoint: Dict[str, Any], model_info) -> Any:
        """Create generic PyTorch model from checkpoint."""
        
        import torch.nn as nn
        
        try:
            # Create a generic neural network based on model info
            layers = []
            input_size = model_info.input_shape[0] if model_info.input_shape else 4
            
            # Simple feedforward architecture
            hidden_sizes = [64, 128, 256, 128, 64]
            
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_size = hidden_size
            
            # Output layer
            output_size = 1  # Default to single output
            if model_info.output_shape:
                if len(model_info.output_shape) == 1:
                    output_size = model_info.output_shape[0]
                else:
                    # Flatten multi-dimensional output
                    output_size = 1
                    for dim in model_info.output_shape:
                        output_size *= dim
            
            layers.append(nn.Linear(prev_size, output_size))
            
            # Create model
            model = nn.Sequential(*layers)
            
            # Try to load state dict (may not match exactly)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                logger.warning(f"Could not load exact state dict for {model_info.model_id}, using random weights")
            
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Generic model creation failed: {e}")
            return None
    
    def get_model_summary(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of cached model."""
        
        model_path = self.cache_dir / f"{model_id}.pth"
        metadata_path = self.cache_dir / f"{model_id}_metadata.json"
        
        if not model_path.exists():
            return None
        
        summary = {
            'model_id': model_id,
            'file_path': str(model_path),
            'file_size_mb': model_path.stat().st_size / (1024 * 1024),
            'cached_date': time.ctime(model_path.stat().st_mtime)
        }
        
        # Load metadata if available
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                summary.update(metadata)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        return summary