import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer.
    Uses B-splines for learnable activation functions on edges.
    """
    def __init__(self, in_f, out_f, grid=10, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-2, 2]):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.grid_size = grid
        self.spline_order = spline_order
        
        # Grid parameters
        h = (grid_range[1] - grid_range[0]) / grid
        grid_pts = torch.linspace(grid_range[0], grid_range[1], grid + 1)
        
        # Extend grid for B-splines
        grid_ext = torch.cat([
            torch.linspace(grid_range[0] - spline_order * h, grid_range[0] - h, spline_order),
            grid_pts,
            torch.linspace(grid_range[1] + h, grid_range[1] + spline_order * h, spline_order)
        ])
        
        self.register_buffer('grid', grid_ext)
        
        # Base activation (silu by default)
        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        # Trainable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.spline_weight = nn.Parameter(torch.Tensor(out_f, in_f, grid + spline_order))
        
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_f, self.out_f) - 1/2) * scale_noise / self.grid_size
            # This initialization is simplified compared to original paper implementation for stability
            self.spline_weight.data.uniform_(-scale_noise, scale_noise)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline bases for input x.
        x: (batch, in_f)
        returns: (batch, in_f, grid + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_f
        
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        
        # 0-th order splines (step functions)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        
        # Recursive calculation for higher order splines
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[:, :, 1:]
        
        assert bases.size(2) == self.grid_size + self.spline_order
        return bases

    def forward(self, x):
        # Handle sequence dimension if present: [B, S, F] -> [B*S, F]
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, self.in_f)
            
        # Base activation branch
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline branch
        # x is assumed to be in range, consider clamping or normalizing if not
        # The notebook implementation used simple clamping
        x_clamped = torch.clamp(x, -2.5, 2.5) 
        
        # Notebook implementation of b_spline logic was slightly different/compact
        # Let's adapt to be closer to the efficient notebook version provided in context
        # But for robustness, I'll stick to a clean implementation.
        # Wait, the notebook implementation was:
        """
        def b_spline(self, x):
            x = torch.clamp(x.unsqueeze(-1), -2.5, 2.5)
            g = self.grid
            b = ((x >= g[:-1]) & (x < g[1:])).float()
            for p in range(1, 4):
                d1 = g[p:-1] - g[:-p-1] + 1e-8
                d2 = g[p+1:] - g[1:-p] + 1e-8
                b = (x - g[:-p-1])/d1 * b[...,:-1] + (g[p+1:] - x)/d2 * b[...,1:]
            return b
        """
        # I will use the notebook's logic exactly to ensure identical behavior
        spline_basis = self.compute_splines_notebook_style(x)
        
        # spline_weight: [out_f, in_f, grid+order]
        # spline_basis: [batch, in_f, grid+order]
        # output: [batch, out_f]
        spline_output = torch.einsum('bi...,oi...->bo', spline_basis, self.spline_weight)
        
        output = base_output + spline_output
        
        # Restore shape
        if len(original_shape) == 3:
            output = output.reshape(original_shape[0], original_shape[1], self.out_f)
            
        return output

    def compute_splines_notebook_style(self, x):
        """Exact implementation from the notebook for consistency"""
        x = torch.clamp(x.unsqueeze(-1), -2.5, 2.5)
        g = self.grid
        # g has size grid + 2*order + 1 ? 
        # In notebook: 
        # g = torch.cat([torch.linspace(-2-3*h, -2-h, 3), g, torch.linspace(2+h, 2+3*h, 3)])
        # That's 3 + (grid+1) + 3 = grid + 7 points for order 3?
        # My init creates grid+2*order + 1 points roughly. 
        
        b = ((x >= g[:-1]) & (x < g[1:])).float()
        for p in range(1, self.spline_order + 1):
            # Avoid division by zero with eps
            d1 = g[p:-1] - g[:-p-1] + 1e-8
            d2 = g[p+1:] - g[1:-p] + 1e-8
            b = (x - g[:-p-1])/d1 * b[...,:-1] + (g[p+1:] - x)/d2 * b[...,1:]
        return b

import math
