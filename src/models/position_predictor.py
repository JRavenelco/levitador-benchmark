import torch
import torch.nn as nn
from src.kan_pinn.kan_layer import KANLayer

class PositionPredictor(nn.Module):
    """
    Position Predictor using KAN.
    
    Input: (u, i, phi_estimated)
    Output: Predicted Position (y)
    """
    def __init__(self, grid_size=10):
        super().__init__()
        # Input features: u, i, phi
        self.kan1 = KANLayer(3, 32, grid=grid_size)
        self.kan2 = KANLayer(32, 16, grid=grid_size)
        self.kan3 = KANLayer(16, 1, grid=grid_size)

    def forward(self, x):
        # x: [Batch, SeqLen, 3] -> (u, i, phi)
        out = self.kan1(x)
        out = self.kan2(out)
        out = self.kan3(out)  # [Batch, SeqLen, 1] -> y estimated
        return out
