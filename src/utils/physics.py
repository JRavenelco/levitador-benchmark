import torch

class LevitationPhysics:
    """
    Physics helper for Magnetic Levitator.
    Based on Santana (2023) model.
    """
    def __init__(self, params):
        """
        Initialize with identified parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing K0, A, m, g, etc.
        """
        self.K0 = params.get('K0', 0.0657)
        self.A = params.get('A', 0.00498)
        self.K_offset = params.get('K_offset', 0.0)
        self.m = params.get('m', 0.009)
        self.g = params.get('g', 9.81)
        
    def inductance_L(self, y):
        """Calculates inductance L given position y."""
        return self.K_offset + self.K0 / (1 + y / self.A)

    def inductance_dL_dy(self, y):
        """Calculates the derivative of inductance with respect to y."""
        denom = 1 + y / self.A
        return -self.K0 / (self.A * denom**2)

    def magnetic_force_iy(self, i, y):
        """Calculates magnetic force given current i and position y."""
        dL = self.inductance_dL_dy(y)
        return 0.5 * (i**2) * dL

    def magnetic_force_iphi(self, i, phi):
        """
        Calculates magnetic force given current i and flux phi.
        Uses algebraic relation to avoid computing y explicitly.
        
        dL/dy = - (L - K_offset)^2 / (A * K0) * sign?
        Wait, derived from notebook:
        dL/dy = - (L - K_offset)^2 / (A * K0)
        """
        # Avoid division by zero
        i_safe = i + 1e-9 * torch.sign(i)
        L_val = phi / i_safe
        
        # dL/dy derivation check:
        # L = K0/(1+y/A) => 1+y/A = K0/L => y/A = K0/L - 1 => y = A(K0/L - 1)
        # dL/dy = -K0/(A(1+y/A)^2) = -K0/A * (L/K0)^2 = -L^2 / (A*K0)
        
        dL_dy_val = -((L_val - self.K_offset)**2) / (self.A * self.K0)
        return 0.5 * (i**2) * dL_dy_val
