"""
CNN Backbone for Hierarchical Multi-Task Learning
Semiconductor Transfer Robot Fault Diagnosis
"""
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    1D CNN backbone for vibration signal feature extraction
    Input: (batch_size, 2, 780) - 2 channels, 780 time steps
    Output: (batch_size, hidden_dim) - latent representation
    """
    def __init__(self, input_channels=2, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Block 1: 16 channels
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            nn.Dropout(0.1)
        )
        
        # Block 2: 32 channels
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Dropout(0.1)
        )
        
        # Block 3: 64 channels
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 2, 780)
        Returns:
            h: (batch_size, hidden_dim)
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = h.squeeze(-1)  # (batch_size, 64)
        h = self.fc(h)      # (batch_size, hidden_dim)
        return h
