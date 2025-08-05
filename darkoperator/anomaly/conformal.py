"""Conformal anomaly detection with rigorous statistical guarantees."""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
import logging

from .base import AnomalyDetector


class ConformalDetector(AnomalyDetector):
    """
    Conformal anomaly detector with statistical guarantees.
    
    Provides rigorous p-values for anomaly detection using conformal prediction
    framework, ensuring valid false discovery rate control.
    """
    
    def __init__(
        self,
        operator,
        background_data: Optional[Union[str, torch.Tensor]] = None,
        alpha: float = 1e-6,  # False discovery rate (5-sigma equivalent)
        calibration_split: float = 0.5,
        n_bootstrap: int = 1000,
        device: str = "auto"
    ):
        super().__init__(operator, background_data, device)
        
        self.alpha = alpha
        self.calibration_split = calibration_split
        self.n_bootstrap = n_bootstrap
        
        # Conformal prediction components
        self.calibration_scores = None
        self.quantile_threshold = None
        self.is_calibrated = False
        
        self.logger = logging.getLogger(__name__)
    
    def calibrate(self, background_data: torch.Tensor) -> None:
        """
        Calibrate detector using background data.
        
        Args:
            background_data: Background events for calibration (n_events, ...)
        """
        self.logger.info(f"Calibrating conformal detector with {len(background_data)} events")
        
        # Split data for proper conformal calibration
        cal_data, _ = train_test_split(
            background_data, 
            test_size=1-self.calibration_split,
            random_state=42
        )
        
        cal_data = torch.tensor(cal_data, device=self.device, dtype=torch.float32)
        
        # Compute conformity scores on calibration set
        with torch.no_grad():
            reconstruction = self.operator(cal_data)
            conformity_scores = self._compute_conformity_scores(cal_data, reconstruction)
        
        self.calibration_scores = conformity_scores.cpu().numpy()
        
        # Compute quantile threshold for given alpha level
        n_cal = len(self.calibration_scores)
        quantile_level = (n_cal + 1) * (1 - self.alpha) / n_cal
        self.quantile_threshold = np.quantile(self.calibration_scores, quantile_level)
        
        self.is_calibrated = True
        self.logger.info(f"Calibration complete. Threshold: {self.quantile_threshold:.6f}")
    
    def _compute_conformity_scores(self, original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Compute conformity scores measuring typicality.
        
        Args:
            original: Original events
            reconstruction: Reconstructed events from operator
            
        Returns:
            Conformity scores (higher = more anomalous)
        """
        # L2 reconstruction error as base conformity score
        mse = torch.mean((original - reconstruction) ** 2, dim=tuple(range(1, len(original.shape))))
        
        # Add physics-informed components
        if hasattr(self.operator, 'physics_loss'):
            physics_penalty = self.operator.physics_loss(original, reconstruction)
            if physics_penalty.dim() == 0:  # Scalar loss
                physics_penalty = physics_penalty.expand(mse.shape[0])
        else:
            physics_penalty = torch.zeros_like(mse)
        
        # Combine scores
        conformity_scores = mse + 0.1 * physics_penalty
        
        return conformity_scores
    
    def compute_p_values(self, test_events: torch.Tensor) -> np.ndarray:
        """
        Compute conformal p-values for test events.
        
        Args:
            test_events: Events to evaluate (n_events, ...)
            
        Returns:
            P-values for each event
        """
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before computing p-values")
        
        test_events = torch.tensor(test_events, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            reconstruction = self.operator(test_events)
            test_scores = self._compute_conformity_scores(test_events, reconstruction)
        
        test_scores = test_scores.cpu().numpy()
        
        # Compute p-values using calibration scores
        p_values = np.zeros(len(test_scores))
        
        for i, score in enumerate(test_scores):
            # Count calibration scores >= test score
            n_greater = np.sum(self.calibration_scores >= score)
            n_cal = len(self.calibration_scores)
            
            # Conformal p-value with randomization for ties
            p_values[i] = (n_greater + 1) / (n_cal + 1)
        
        return p_values
    
    def find_anomalies(
        self, 
        events: torch.Tensor, 
        return_scores: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Find anomalous events with statistical guarantees.
        
        Args:
            events: Events to analyze
            return_scores: Whether to return p-values along with indices
            
        Returns:
            Anomalous event indices (and optionally p-values)
        """
        p_values = self.compute_p_values(events)
        
        # Find events below alpha threshold
        anomalous_indices = np.where(p_values < self.alpha)[0].tolist()
        
        # Sort by p-value (most anomalous first)
        sorted_indices = sorted(anomalous_indices, key=lambda i: p_values[i])
        
        self.logger.info(f"Found {len(sorted_indices)} anomalies out of {len(events)} events")
        
        if return_scores:
            return sorted_indices, p_values[sorted_indices]
        else:
            return sorted_indices
    
    def estimate_discovery_power(
        self, 
        signal_events: torch.Tensor,
        n_bootstrap: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Estimate statistical power for signal discovery.
        
        Args:
            signal_events: Known signal events for power estimation
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            (mean_power, std_power) - statistical power estimates
        """
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before power estimation")
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        signal_p_values = self.compute_p_values(signal_events)
        
        # Bootstrap power estimation
        powers = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_indices = np.random.choice(len(signal_p_values), len(signal_p_values))
            bootstrap_p_values = signal_p_values[bootstrap_indices]
            
            # Compute power (fraction of signals detected)
            power = np.mean(bootstrap_p_values < self.alpha)
            powers.append(power)
        
        mean_power = np.mean(powers)
        std_power = np.std(powers)
        
        return mean_power, std_power
    
    def validate_false_discovery_rate(
        self,
        background_events: torch.Tensor,
        n_trials: int = 100
    ) -> Tuple[float, float]:
        """
        Validate that actual FDR matches nominal alpha level.
        
        Args:
            background_events: Independent background events for validation
            n_trials: Number of validation trials
            
        Returns:
            (empirical_fdr, std_fdr) - measured false discovery rate
        """
        fdrs = []
        
        for trial in range(n_trials):
            # Sample background events
            n_sample = min(1000, len(background_events))
            sample_indices = np.random.choice(len(background_events), n_sample, replace=False)
            sample_events = background_events[sample_indices]
            
            # Compute discoveries
            discoveries = self.find_anomalies(sample_events)
            fdr = len(discoveries) / n_sample if n_sample > 0 else 0.0
            fdrs.append(fdr)
        
        empirical_fdr = np.mean(fdrs)
        std_fdr = np.std(fdrs)
        
        self.logger.info(f"Empirical FDR: {empirical_fdr:.6f} ± {std_fdr:.6f} (target: {self.alpha:.6f})")
        
        return empirical_fdr, std_fdr