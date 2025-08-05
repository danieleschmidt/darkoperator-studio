"""Multi-modal operator combining multiple detector systems."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .base import PhysicsOperator


class MultiModalOperator(nn.Module):
    """Combines multiple detector operators for joint analysis."""
    
    def __init__(self, operators: List[Tuple[str, PhysicsOperator]]):
        super().__init__()
        self.operators = nn.ModuleDict({name: op for name, op in operators})
        
        # Fusion layer
        total_features = sum(op.output_shape[0] * op.output_shape[1] * op.output_shape[2] 
                           for op in self.operators.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        features = []
        
        for name, operator in self.operators.items():
            output = operator(x)
            outputs[name] = output
            features.append(output.flatten(1))
        
        # Fuse features
        combined = torch.cat(features, dim=1)
        fusion_score = self.fusion(combined)
        outputs['fusion_score'] = fusion_score
        
        return outputs