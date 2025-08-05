"""Training utilities for neural operators."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm


class OperatorTrainer:
    """Training manager for physics operators."""
    
    def __init__(
        self,
        model: nn.Module,
        physics_loss: bool = True,
        symmetry_loss: bool = True,
        learning_rate: float = 1e-3,
        device: str = "auto"
    ):
        self.model = model
        self.physics_loss = physics_loss
        self.symmetry_loss = symmetry_loss
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """Train the operator model."""
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_data)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_data:
                val_loss = self._validate_epoch(val_data)
                history["val_loss"].append(val_loss)
                self.logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = nn.MSELoss()(outputs, targets)
            
            if self.physics_loss and hasattr(self.model, 'physics_loss'):
                physics_penalty = self.model.physics_loss(inputs, outputs)
                loss += 0.1 * physics_penalty
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = nn.MSELoss()(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)