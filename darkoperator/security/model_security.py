"""Model security and safe loading utilities."""

import torch
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class SecureModelLoader:
    """Secure model loading with integrity verification."""
    
    TRUSTED_DOMAINS = [
        'huggingface.co',
        'github.com', 
        'opendata.cern.ch',
        'zenodo.org'
    ]
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checksums_file = self.cache_dir / "checksums.json"
        self.checksums = self._load_checksums()
    
    def _load_checksums(self) -> Dict[str, str]:
        """Load stored checksums for verification."""
        if self.checksums_file.exists():
            try:
                with open(self.checksums_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checksums: {e}")
        return {}
    
    def _save_checksums(self):
        """Save checksums to disk."""
        try:
            with open(self.checksums_file, 'w') as f:
                json.dump(self.checksums, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checksums: {e}")
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _is_trusted_url(self, url: str) -> bool:
        """Check if URL is from a trusted domain."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return any(trusted in domain for trusted in self.TRUSTED_DOMAINS)
    
    def download_model(
        self,
        url: str,
        filename: str,
        expected_hash: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Securely download model from trusted source.
        
        Args:
            url: Download URL
            filename: Local filename
            expected_hash: Expected SHA-256 hash for verification
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded file
            
        Raises:
            SecurityError: If download source is untrusted or hash mismatch
        """
        if not self._is_trusted_url(url):
            raise SecurityError(f"Untrusted download source: {url}")
        
        filepath = self.cache_dir / filename
        
        # Check if already cached and valid
        if filepath.exists() and not force_download:
            if filename in self.checksums:
                stored_hash = self.checksums[filename]
                actual_hash = self._compute_file_hash(filepath)
                if stored_hash == actual_hash:
                    logger.info(f"Using cached model: {filepath}")
                    return filepath
                else:
                    logger.warning(f"Hash mismatch for cached file, re-downloading")
        
        # Download file
        logger.info(f"Downloading model from: {url}")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filepath} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            
        except Exception as e:
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            raise SecurityError(f"Failed to download model: {e}")
        
        # Verify hash if provided
        actual_hash = self._compute_file_hash(filepath)
        if expected_hash and actual_hash != expected_hash:
            filepath.unlink()  # Remove invalid file
            raise SecurityError(f"Hash verification failed. Expected: {expected_hash}, Got: {actual_hash}")
        
        # Store hash for future verification
        self.checksums[filename] = actual_hash
        self._save_checksums()
        
        return filepath
    
    def load_model_safe(
        self,
        filepath: Path,
        model: torch.nn.Module,
        map_location: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Safely load PyTorch model with security checks.
        
        Args:
            filepath: Path to model file
            model: Model instance to load into
            map_location: Device mapping for loading
            
        Returns:
            Model metadata
        """
        # Verify file integrity
        if filepath.name in self.checksums:
            expected_hash = self.checksums[filepath.name]
            actual_hash = self._compute_file_hash(filepath)
            if expected_hash != actual_hash:
                raise SecurityError(f"Model file integrity check failed: {filepath}")
        
        # Check file size (prevent loading extremely large files)
        max_size_gb = 5  # 5GB limit
        file_size = filepath.stat().st_size
        if file_size > max_size_gb * 1024**3:
            raise SecurityError(f"Model file too large: {file_size / (1024**3):.1f} GB > {max_size_gb} GB")
        
        # Load with security restrictions
        try:
            checkpoint = torch.load(
                filepath,
                map_location=map_location,
                weights_only=True  # Prevent arbitrary code execution
            )
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise SecurityError("Invalid checkpoint format")
            
            if 'model_state_dict' not in checkpoint:
                raise SecurityError("Missing model_state_dict in checkpoint")
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            logger.info(f"Successfully loaded model: {filepath}")
            return checkpoint.get('metadata', {})
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise SecurityError(f"Model loading failed: {e}")


class SecurityError(Exception):
    """Security-related exception."""
    pass


def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate checkpoint file for basic security.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        True if checkpoint passes basic security checks
    """
    filepath = Path(checkpoint_path)
    
    if not filepath.exists():
        raise SecurityError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check file extension
    if filepath.suffix not in ['.pt', '.pth', '.pkl']:
        raise SecurityError(f"Unsupported checkpoint format: {filepath.suffix}")
    
    # Check file size
    max_size = 2 * 1024**3  # 2GB
    if filepath.stat().st_size > max_size:
        raise SecurityError(f"Checkpoint file too large: {filepath.stat().st_size / (1024**3):.1f} GB")
    
    # Basic structure check (without loading)
    try:
        # Just check if it's a valid torch file without loading weights
        torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return True
    except Exception as e:
        raise SecurityError(f"Invalid checkpoint format: {e}")


# Pre-approved model checksums for common models
APPROVED_CHECKSUMS = {
    "darkoperator_calo_v1.pt": "a1b2c3d4e5f6...",  # Example - would be real hashes
    "darkoperator_tracker_v1.pt": "f6e5d4c3b2a1...",
}


def get_approved_models() -> List[str]:
    """Get list of pre-approved model files."""
    return list(APPROVED_CHECKSUMS.keys())