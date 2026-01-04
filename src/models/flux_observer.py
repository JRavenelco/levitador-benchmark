import torch
import torch.nn as nn
from src.kan_pinn.hippo_layer import HiPPOLayer
from src.kan_pinn.kan_layer import KANLayer

class FluxObserverHiPPO(nn.Module):
    """
    Flux Observer using HiPPO + KAN.
    
    HiPPO captures the integral of (u - Ri) naturally, which corresponds to flux.
    Input: Sequence of (u, i)
    Output: Estimated Flux (phi)
    """
    def __init__(self, hippo_n=8, grid_size=10):
        super().__init__()
        self.hippo = HiPPOLayer(N=hippo_n)

        # Input: [u, i] -> 2 features
        # HiPPO output: 2 features * hippo_n coefficients
        self.kan1 = KANLayer(2 * hippo_n, 32, grid=grid_size)
        self.kan2 = KANLayer(32, 16, grid=grid_size)
        self.kan3 = KANLayer(16, 1, grid=grid_size)

    def forward(self, x_seq):
        # x_seq: [Batch, SeqLen, 2] -> (u, i)
        h = self.hippo(x_seq)  # [Batch, SeqLen, 2*N]
        
        # Process through KAN layers
        out = self.kan1(h)
        out = self.kan2(out)
        out = self.kan3(out)  # [Batch, SeqLen, 1] -> phi estimated
        
        return out
