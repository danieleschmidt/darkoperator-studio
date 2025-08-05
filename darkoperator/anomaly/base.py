"""Base anomaly detector class."""

import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class AnomalyDetector(ABC):
    """Base class for anomaly detectors."""
    
    def __init__(self, operator, background_data=None, device="auto"):
        self.operator = operator
        self.background_data = background_data
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)
    
    @abstractmethod
    def find_anomalies(self, events: torch.Tensor) -> List[int]:
        """Find anomalous events."""
        pass