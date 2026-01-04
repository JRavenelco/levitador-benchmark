"""
HiPPO Layer
===========

HiPPO-LegS (Legendre-Scaled) layer for online function approximation.
Optimized with TorchScript (JIT) for fast recurrence.
"""

import torch
import torch.nn as nn
import numpy as np

# JIT compiled function for fast recurrence
@torch.jit.script
def hippo_recurrence(x_seq: torch.Tensor, A_d: torch.Tensor, B_d: torch.Tensor) -> torch.Tensor:
    """
    Fast recurrence implementation using TorchScript.
    
    Parameters
    ----------
    x_seq : torch.Tensor
        Input sequence [Batch, SeqLen, Features]
    A_d : torch.Tensor
        Discretized A matrix [N, N]
    B_d : torch.Tensor
        Discretized B matrix [N, 1]
        
    Returns
    -------
    torch.Tensor
        Sequence of coefficients [Batch, SeqLen, Features * N]
    """
    B, S, F = x_seq.shape
    N = A_d.shape[0]
    
    # Initialize coefficients c (Batch, Features, N)
    c = torch.zeros(B, F, N, device=x_seq.device)
    out = []

    # Pre-transpose A for efficient matmul
    A_t = A_d.t()

    for t in range(S):
        # x_t: [Batch, Features, 1]
        x_t = x_seq[:, t, :].unsqueeze(-1)
        
        # Linear recurrence: c_t = c_{t-1} @ A^T + B * x_t
        # c: [B, F, N]
        # A_t: [N, N] -> broadcasted matmul
        # B_d: [N, 1] -> broadcasted add
        c = torch.matmul(c, A_t) + B_d * x_t
        
        # Flatten features and coefficients for output: [Batch, Features * N]
        out.append(c.reshape(B, -1))

    # Stack along sequence dimension: [Batch, SeqLen, Features * N]
    return torch.stack(out, dim=1)


class HiPPOLayer(nn.Module):
    """
    HiPPO-LegS layer for capturing temporal history.
    Uses Legendre polynomials to project history into a low-dimensional state.
    """
    def __init__(self, N=8, dt=0.01):
        super().__init__()
        self.N = N
        self.dt = dt

        # Legendre Matrix A calculation
        A_mat = np.zeros((N, N))
        for n in range(N):
            for m in range(N):
                if n > m:
                    A_mat[n, m] = np.sqrt(2*n + 1) * np.sqrt(2*m + 1)
                elif n == m:
                    A_mat[n, m] = n + 1

        # Bilinear Discretization (Tustin)
        I = np.eye(N)
        # Avoid singular matrix in rare cases, though usually fine for HiPPO
        inv_term = np.linalg.inv(I - (dt/2) * A_mat)
        
        A_disc = inv_term @ (I + (dt/2) * A_mat)
        B_disc = inv_term @ (dt * np.sqrt(2 * np.arange(N) + 1))
        B_disc = B_disc.reshape(-1, 1) # Ensure column vector

        # Register buffers (non-trainable constants)
        self.register_buffer("A_d", torch.tensor(A_disc, dtype=torch.float32))
        self.register_buffer("B_d", torch.tensor(B_disc, dtype=torch.float32))

    def forward(self, x_seq):
        """
        Forward pass.
        
        Parameters
        ----------
        x_seq : torch.Tensor
            Input [Batch, SeqLen, Features]
            
        Returns
        -------
        torch.Tensor
            Output [Batch, SeqLen, Features * N]
        """
        return hippo_recurrence(x_seq, self.A_d, self.B_d)
