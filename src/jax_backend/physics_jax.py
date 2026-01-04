"""
JAX-Vectorized Physics Model for Magnetic Levitator
====================================================

Uses JAX's vmap to evaluate the ENTIRE population in parallel on GPU.
Instead of simulating 1 sphere at a time, we simulate N spheres simultaneously.

Key Optimizations:
1. vmap: Automatic vectorization over population dimension
2. jit: Just-In-Time compilation for GPU kernels
3. scan: Efficient sequential operations (ODE integration)
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.lax import scan
from functools import partial


class LevitadorPhysicsJAX:
    """
    JAX-compatible physics model for the magnetic levitator.
    All operations are designed to be vectorizable with vmap.
    """
    
    def __init__(self, m: float = 0.009, g: float = 9.81, dt: float = 0.01):
        """
        Parameters
        ----------
        m : float
            Mass of sphere [kg]
        g : float
            Gravitational acceleration [m/s²]
        dt : float
            Time step for integration [s]
        """
        self.m = m
        self.g = g
        self.dt = dt
    
    @staticmethod
    @jit
    def inductance(y: jnp.ndarray, K0: float, A: float) -> jnp.ndarray:
        """Calculate inductance L(y) = K0 / (1 + y/A)"""
        return K0 / (1.0 + y / A)
    
    @staticmethod
    @jit
    def dL_dy(y: jnp.ndarray, K0: float, A: float) -> jnp.ndarray:
        """Calculate dL/dy = -K0 / (A * (1 + y/A)²)"""
        denom = 1.0 + y / A
        return -K0 / (A * denom ** 2)
    
    @staticmethod
    @jit
    def magnetic_force(i: jnp.ndarray, y: jnp.ndarray, K0: float, A: float) -> jnp.ndarray:
        """Calculate magnetic force F = 0.5 * i² * dL/dy"""
        dL = LevitadorPhysicsJAX.dL_dy(y, K0, A)
        return 0.5 * i ** 2 * dL


@partial(jit, static_argnums=(6, 7, 8))
def simulate_single(params: jnp.ndarray, 
                    t_data: jnp.ndarray, 
                    u_data: jnp.ndarray, 
                    i_data: jnp.ndarray,
                    y0: float,
                    y_dot0: float,
                    m: float,
                    g: float,
                    dt: float) -> jnp.ndarray:
    """
    Simulate the levitator dynamics for a single parameter set.
    
    Parameters
    ----------
    params : jnp.ndarray
        [K0, A, R0, alpha] - parameters to evaluate
    t_data : jnp.ndarray
        Time points [N]
    u_data : jnp.ndarray
        Voltage data [N]
    i_data : jnp.ndarray
        Current data [N]
    y0, y_dot0 : float
        Initial conditions
    m, g, dt : float
        Physical constants
    
    Returns
    -------
    y_sim : jnp.ndarray
        Simulated position trajectory [N]
    """
    K0, A, R0, alpha = params[0], params[1], params[2], params[3]
    
    N = t_data.shape[0]
    
    # Initial state: [y, y_dot]
    state0 = jnp.array([y0, y_dot0])
    
    def step_fn(state, inputs):
        """Single integration step using Euler method (fast, GPU-friendly)"""
        y, y_dot = state[0], state[1]
        t, u, i = inputs[0], inputs[1], inputs[2]
        
        # Clamp position to avoid numerical issues
        y = jnp.clip(y, 1e-6, 0.05)
        
        # Inductance and derivative
        L = K0 / (1.0 + y / A)
        dL_dy_val = -K0 / (A * (1.0 + y / A) ** 2)
        
        # Magnetic force (always attractive, negative direction)
        F_mag = 0.5 * i ** 2 * dL_dy_val
        
        # Acceleration: m*a = F_mag + m*g (gravity pulls down, F_mag pulls up if dL/dy < 0)
        # Sign convention: y increases downward (gap increases = falling)
        # F_mag < 0 (attracts), so a = F_mag/m + g
        a = F_mag / m + g
        
        # Euler integration
        y_new = y + y_dot * dt
        y_dot_new = y_dot + a * dt
        
        # Clamp velocity for stability
        y_dot_new = jnp.clip(y_dot_new, -10.0, 10.0)
        
        new_state = jnp.array([y_new, y_dot_new])
        return new_state, y_new
    
    # Stack inputs for scan
    inputs = jnp.stack([t_data, u_data, i_data], axis=1)
    
    # Run simulation using scan (efficient sequential op on GPU)
    _, y_trajectory = scan(step_fn, state0, inputs)
    
    return y_trajectory


@partial(jit, static_argnums=(6, 7, 8))
def fitness_single(params: jnp.ndarray,
                   t_data: jnp.ndarray,
                   u_data: jnp.ndarray,
                   i_data: jnp.ndarray,
                   y_real: jnp.ndarray,
                   y0: float,
                   m: float,
                   g: float,
                   dt: float) -> float:
    """
    Compute fitness (MSE) for a single parameter set.
    Lower is better.
    """
    y_sim = simulate_single(params, t_data, u_data, i_data, y0, 0.0, m, g, dt)
    
    # MSE between simulated and real position
    mse = jnp.mean((y_sim - y_real) ** 2)
    
    # Add penalty for parameters outside reasonable bounds
    K0, A, R0, alpha = params[0], params[1], params[2], params[3]
    
    penalty = 0.0
    penalty += jnp.where(K0 < 0.001, 1e6, 0.0)
    penalty += jnp.where(K0 > 0.15, 1e6, 0.0)
    penalty += jnp.where(A < 0.0001, 1e6, 0.0)
    penalty += jnp.where(A > 0.05, 1e6, 0.0)
    penalty += jnp.where(R0 < 1.0, 1e6, 0.0)
    penalty += jnp.where(R0 > 5.0, 1e6, 0.0)
    
    return mse + penalty


def create_vectorized_fitness(t_data: jnp.ndarray,
                               u_data: jnp.ndarray,
                               i_data: jnp.ndarray,
                               y_real: jnp.ndarray,
                               y0: float,
                               m: float = 0.009,
                               g: float = 9.81,
                               dt: float = 0.01):
    """
    Create a vectorized fitness function that evaluates the ENTIRE population
    in a single GPU call using vmap.
    
    Returns
    -------
    vectorized_fitness_fn : callable
        Function that takes population [pop_size, 4] and returns fitness [pop_size]
    """
    
    # Partial application of fixed data
    @jit
    def fitness_fn(params):
        return fitness_single(params, t_data, u_data, i_data, y_real, y0, m, g, dt)
    
    # MAGIC: vmap automatically parallelizes over the population dimension!
    # This is where the GPU acceleration happens.
    vectorized_fn = jit(vmap(fitness_fn, in_axes=0))
    
    return vectorized_fn


# Convenience function for direct use
def vectorized_fitness(population: jnp.ndarray,
                       t_data: jnp.ndarray,
                       u_data: jnp.ndarray,
                       i_data: jnp.ndarray,
                       y_real: jnp.ndarray,
                       y0: float,
                       m: float = 0.009,
                       g: float = 9.81,
                       dt: float = 0.01) -> jnp.ndarray:
    """
    Evaluate fitness for entire population in parallel on GPU.
    
    Parameters
    ----------
    population : jnp.ndarray
        Population matrix [pop_size, 4] where each row is [K0, A, R0, alpha]
    t_data, u_data, i_data, y_real : jnp.ndarray
        Experimental data arrays
    y0 : float
        Initial position
    m, g, dt : float
        Physical constants
    
    Returns
    -------
    fitness : jnp.ndarray
        Fitness values [pop_size] - lower is better
    """
    vfn = create_vectorized_fitness(t_data, u_data, i_data, y_real, y0, m, g, dt)
    return vfn(population)
