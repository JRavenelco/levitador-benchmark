"""
Parameter Benchmark
===================

Phase 1: Physical parameter identification for the magnetic levitator.

This module identifies physical parameters:
- K0, A: Inductance L(y) = K0 / (1 + y/A)
- R0, α: Resistance R(t) ≈ R0 * (1 + α*(T(t) - T0))

The resistance R(t) cannot be directly measured (no temperature sensor).
It is estimated via Kirchhoff's law: u = R·i + dφ/dt
  => R_est(t) = (u(t) - dφ̂(t)/dt) / i(t)

Then R(t) is fitted parametrically or smoothed with regularization.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParameterBenchmark:
    """
    Benchmark for identifying physical parameters [K0, A, R0, α] or [K0, A, R(t)_params].
    
    The fitness function simulates the system dynamics and compares with real data.
    
    Physical Model:
    ---------------
    Mechanical equation:
        m·ÿ = (1/2)·(∂L/∂y)·i² + m·g
    
    Electrical equation:
        L(y)·(di/dt) + (∂L/∂y)·ẏ·i + R(t)·i = u
    
    Inductance model:
        L(y) = K0 / (1 + y/A)
        ∂L/∂y = -K0 / (A·(1 + y/A)²)
    
    Resistance model (no temperature sensor):
        R(t) ≈ R0 * (1 + α*(T_est(t) - T0))
        or smoothed R_est(t) from Kirchhoff
    
    Parameters to optimize:
    ----------------------
    θ = [K0, A, R0, α]
    - K0: Inductance numerator [H]
    - A: Geometric parameter [m]
    - R0: Base resistance [Ω]
    - α: Temperature coefficient [1/°C]
    
    Constraints:
    -----------
    - K0 > 0: Positive inductance
    - A > 0: Positive geometric parameter
    - R0 > 0: Positive resistance
    - α: Can be positive or negative (material dependent)
    - Avoid division by zero when i ≈ 0
    """
    
    def __init__(
        self,
        data_path: str,
        bounds: Optional[List[Tuple[float, float]]] = None,
        m: float = 0.018,
        g: float = 9.81,
        dt: float = 0.01,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 3,
        verbose: bool = True
    ):
        """
        Initialize the parameter benchmark.
        
        Parameters
        ----------
        data_path : str
            Path to experimental data file (t, y, i, u format)
        bounds : list of tuples, optional
            Parameter bounds [(K0_min, K0_max), (A_min, A_max), (R0_min, R0_max), (α_min, α_max)]
            Default: [(0.001, 0.15), (0.0001, 0.05), (1.0, 5.0), (-0.01, 0.01)]
        m : float
            Mass of the sphere [kg]
        g : float
            Gravitational acceleration [m/s²]
        dt : float
            Time step for integration [s]
        smoothing_window : int
            Window size for Savitzky-Golay filter (odd number)
        smoothing_polyorder : int
            Polynomial order for smoothing
        verbose : bool
            Enable verbose output
        """
        self.m = m
        self.g = g
        self.dt = dt
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.verbose = verbose
        
        # Load experimental data
        self._load_data(data_path)
        
        # Set parameter bounds
        if bounds is None:
            # Default bounds: [K0, A, R0, α]
            self.bounds = [
                (0.001, 0.15),   # K0: Inductance numerator [H]
                (0.0001, 0.05),  # A: Geometric parameter [m]
                (1.0, 5.0),      # R0: Base resistance [Ω]
                (-0.01, 0.01)    # α: Temperature coefficient [1/°C]
            ]
        else:
            self.bounds = bounds
        
        self.dim = len(self.bounds)
        self.variable_names = ['K0', 'A', 'R0', 'alpha']
        
        # Estimate initial flux from data using simple rectangular integration
        self._estimate_initial_flux()
        
        if self.verbose:
            logger.info(f"ParameterBenchmark initialized with {len(self.t_real)} data points")
            logger.info(f"Time range: [{self.t_real[0]:.3f}, {self.t_real[-1]:.3f}] s")
            logger.info(f"Parameter bounds: {self.bounds}")
    
    def _load_data(self, data_path: str):
        """Load experimental data from file."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data (format: t, y, i, u, ...)
        # Handle both space-separated and tab-separated files
        try:
            data = np.loadtxt(data_path)
        except Exception as e:
            raise ValueError(f"Error loading data file: {e}")
        
        if data.shape[1] < 4:
            raise ValueError(f"Data file must have at least 4 columns (t, y, i, u), got {data.shape[1]}")
        
        self.t_real = data[:, 0]
        self.y_real = data[:, 1]
        self.i_real = data[:, 2]
        self.u_real = data[:, 3]
        
        # Smooth current and voltage to reduce noise in derivatives
        if len(self.i_real) > self.smoothing_window:
            self.i_real_smooth = savgol_filter(
                self.i_real, 
                self.smoothing_window, 
                self.smoothing_polyorder
            )
            self.u_real_smooth = savgol_filter(
                self.u_real, 
                self.smoothing_window, 
                self.smoothing_polyorder
            )
        else:
            self.i_real_smooth = self.i_real.copy()
            self.u_real_smooth = self.u_real.copy()
    
    def _estimate_initial_flux(self):
        """
        Estimate flux φ(t) from experimental data using Kirchhoff's law.
        
        This provides an initial estimate of φ for resistance estimation:
        u = R·i + dφ/dt  =>  dφ/dt ≈ u - R_nominal·i
        
        We integrate this to get φ(t).
        """
        # Use a nominal resistance for initial flux estimation
        R_nominal = 2.5  # Typical value [Ω]
        
        # Calculate dφ/dt ≈ u - R·i
        dphi_dt = self.u_real_smooth - R_nominal * self.i_real_smooth
        
        # Integrate using cumulative trapezoidal rule
        self.phi_est = np.zeros_like(self.t_real)
        for i in range(1, len(self.t_real)):
            dt = self.t_real[i] - self.t_real[i-1]
            self.phi_est[i] = self.phi_est[i-1] + 0.5 * (dphi_dt[i-1] + dphi_dt[i]) * dt
    
    def estimate_resistance_curve(self, K0: float, A: float) -> np.ndarray:
        """
        Estimate R(t) from experimental data using Kirchhoff's law.
        
        Given K0 and A, we can calculate L(y) and estimate:
        R_est(t) = (u(t) - dφ̂(t)/dt) / i(t)
        
        where φ̂(t) = L(y(t)) · i(t)
        
        Parameters
        ----------
        K0 : float
            Inductance numerator
        A : float
            Geometric parameter
        
        Returns
        -------
        R_est : np.ndarray
            Estimated resistance curve [Ω]
        """
        # Calculate L(y) for each position
        L_y = K0 / (1 + self.y_real / A)
        
        # Calculate flux φ = L(y) · i
        phi = L_y * self.i_real_smooth
        
        # Calculate dφ/dt using central differences
        dphi_dt = np.gradient(phi, self.t_real)
        
        # Estimate R from Kirchhoff: R = (u - dφ/dt) / i
        # Avoid division by zero
        i_safe = np.where(np.abs(self.i_real_smooth) < 1e-6, 1e-6, self.i_real_smooth)
        R_est = (self.u_real_smooth - dphi_dt) / i_safe
        
        # Clip to reasonable bounds and smooth
        R_est = np.clip(R_est, 0.5, 10.0)
        
        if len(R_est) > self.smoothing_window:
            R_est = savgol_filter(R_est, self.smoothing_window, self.smoothing_polyorder)
        
        return R_est
    
    def resistance_model(self, t: np.ndarray, R0: float, alpha: float) -> np.ndarray:
        """
        Parametric resistance model: R(t) = R0 * (1 + α*ΔT(t))
        
        Without temperature sensor, we approximate ΔT(t) as proportional to
        integral of i²(t) (Joule heating).
        
        Parameters
        ----------
        t : np.ndarray
            Time points
        R0 : float
            Base resistance [Ω]
        alpha : float
            Temperature coefficient [1/°C]
        
        Returns
        -------
        R_t : np.ndarray
            Resistance curve [Ω]
        """
        # Approximate temperature rise from Joule heating: ΔT ∝ ∫ i²(t) dt
        # Interpolate i_real to requested time points
        i_interp = np.interp(t, self.t_real, self.i_real_smooth)
        
        # Calculate cumulative heating (normalized)
        i_squared = i_interp ** 2
        cumulative_heat = np.zeros_like(t)
        for i in range(1, len(t)):
            dt_local = t[i] - t[i-1]
            cumulative_heat[i] = cumulative_heat[i-1] + 0.5 * (i_squared[i-1] + i_squared[i]) * dt_local
        
        # Normalize to temperature scale (arbitrary units to degrees)
        # Assume max heating ~ 20°C above ambient
        if cumulative_heat[-1] > 0:
            T_rise = 20.0 * cumulative_heat / cumulative_heat[-1]
        else:
            T_rise = np.zeros_like(t)
        
        # R(t) = R0 * (1 + α * ΔT)
        R_t = R0 * (1 + alpha * T_rise)
        
        return np.clip(R_t, 0.5, 10.0)  # Physical bounds
    
    def _system_dynamics(self, state: np.ndarray, t: float, K0: float, A: float, 
                        R_interp_func, u_interp_func) -> np.ndarray:
        """
        System dynamics for ODE integration.
        
        State: [y, ẏ, i]
        
        Equations:
        - Mechanical: m·ÿ = (1/2)·(∂L/∂y)·i² + m·g
        - Electrical: L(y)·(di/dt) + (∂L/∂y)·ẏ·i + R(t)·i = u
        
        Parameters
        ----------
        state : np.ndarray
            Current state [y, y_dot, i]
        t : float
            Current time
        K0, A : float
            Inductance parameters
        R_interp_func : callable
            Resistance interpolation function R(t)
        u_interp_func : callable
            Voltage interpolation function u(t)
        
        Returns
        -------
        derivatives : np.ndarray
            [ẏ, ÿ, di/dt]
        """
        y, y_dot, i = state
        
        # Avoid negative or zero positions
        y = max(y, 1e-6)
        
        # Inductance and its derivative
        L = K0 / (1 + y / A)
        dL_dy = -K0 / (A * (1 + y / A) ** 2)
        
        # Get current voltage and resistance
        u = u_interp_func(t)
        R = R_interp_func(t)
        
        # Mechanical equation: ÿ = (1/(2m))·(∂L/∂y)·i² + g
        y_ddot = (0.5 / self.m) * dL_dy * i ** 2 + self.g
        
        # Electrical equation: di/dt = (u - (∂L/∂y)·ẏ·i - R·i) / L(y)
        if L > 1e-9:
            di_dt = (u - dL_dy * y_dot * i - R * i) / L
        else:
            di_dt = 0.0
        
        return np.array([y_dot, y_ddot, di_dt])
    
    def simulate(self, params: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate system dynamics with given parameters.
        
        Parameters
        ----------
        params : list
            Parameters [K0, A, R0, α]
        
        Returns
        -------
        y_sim : np.ndarray
            Simulated position trajectory
        i_sim : np.ndarray
            Simulated current trajectory
        """
        K0, A, R0, alpha = params
        
        # Generate resistance curve
        R_curve = self.resistance_model(self.t_real, R0, alpha)
        
        # Create interpolation functions
        from scipy.interpolate import interp1d
        R_interp = interp1d(self.t_real, R_curve, kind='linear', 
                           fill_value='extrapolate', bounds_error=False)
        u_interp = interp1d(self.t_real, self.u_real_smooth, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
        
        # Initial conditions [y0, ẏ0, i0]
        y0 = self.y_real[0]
        y_dot0 = 0.0  # Assume starting from rest
        i0 = self.i_real_smooth[0]
        state0 = [y0, y_dot0, i0]
        
        # Integrate ODE
        try:
            solution = odeint(
                self._system_dynamics,
                state0,
                self.t_real,
                args=(K0, A, R_interp, u_interp),
                rtol=1e-6,
                atol=1e-8
            )
            
            y_sim = solution[:, 0]
            i_sim = solution[:, 2]
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"Simulation failed: {e}")
            # Return arrays filled with large errors
            y_sim = np.full_like(self.y_real, np.nan)
            i_sim = np.full_like(self.i_real, np.nan)
        
        return y_sim, i_sim
    
    def fitness_function(self, params: List[float]) -> float:
        """
        Fitness function: MSE between simulated and real trajectories.
        
        Parameters
        ----------
        params : list
            Parameters [K0, A, R0, α]
        
        Returns
        -------
        mse : float
            Mean squared error (position + current weighted)
        """
        # Check bounds
        for i, (lb, ub) in enumerate(self.bounds):
            if not (lb <= params[i] <= ub):
                return 1e10  # Large penalty for out-of-bounds
        
        # Simulate
        y_sim, i_sim = self.simulate(params)
        
        # Check for invalid simulation
        if np.any(np.isnan(y_sim)) or np.any(np.isnan(i_sim)):
            return 1e10
        
        # Calculate MSE for position and current
        mse_y = np.mean((y_sim - self.y_real) ** 2)
        mse_i = np.mean((i_sim - self.i_real_smooth) ** 2)
        
        # Weighted combination (position is more important)
        mse_total = 0.8 * mse_y + 0.2 * mse_i
        
        return mse_total
    
    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds as separate arrays."""
        bounds_array = np.array(self.bounds)
        return bounds_array[:, 0], bounds_array[:, 1]
    
    def visualize_solution(self, params: List[float], save_path: Optional[str] = None):
        """
        Visualize the solution: compare simulated vs real trajectories.
        
        Parameters
        ----------
        params : list
            Parameters [K0, A, R0, α]
        save_path : str, optional
            Path to save the figure
        """
        import matplotlib.pyplot as plt
        
        y_sim, i_sim = self.simulate(params)
        K0, A, R0, alpha = params
        R_curve = self.resistance_model(self.t_real, R0, alpha)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Position
        axes[0].plot(self.t_real, self.y_real, 'b-', label='Real', alpha=0.7)
        axes[0].plot(self.t_real, y_sim, 'r--', label='Simulated', linewidth=2)
        axes[0].set_ylabel('Position [m]')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'K0={K0:.4f}, A={A:.5f}, R0={R0:.2f}, α={alpha:.5f}')
        
        # Current
        axes[1].plot(self.t_real, self.i_real, 'b-', label='Real', alpha=0.7)
        axes[1].plot(self.t_real, i_sim, 'r--', label='Simulated', linewidth=2)
        axes[1].set_ylabel('Current [A]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Resistance
        axes[2].plot(self.t_real, R_curve, 'g-', linewidth=2)
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Resistance [Ω]')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Estimated Resistance R(t)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                logger.info(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
