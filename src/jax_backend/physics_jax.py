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


@jit
def simulate_single(params: jnp.ndarray,
                    u_data: jnp.ndarray,
                    dt_steps: jnp.ndarray,
                    t_rise: jnp.ndarray,
                    y0: float,
                    i0: float,
                    m: float,
                    g: float,
                    internal_steps: int,
                    yddot_weight: float,
                    didt_weight: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate the levitator dynamics for a single parameter set.
    
    This matches the original ODE model used in ParameterBenchmark:
    State: [y, y_dot, i]
    - Mechanical: m*y_ddot = 0.5*dL/dy*i^2 + m*g
    - Electrical: L(y)*di/dt + (dL/dy)*y_dot*i + R(t)*i = u
    
    Parameters
    ----------
    params : jnp.ndarray
        [K0, A, R0, alpha] - parameters to evaluate
    u_data : jnp.ndarray
        Voltage data [N]
    dt_steps : jnp.ndarray
        Time step per sample [N]
    t_rise : jnp.ndarray
        Approximated temperature rise [N] from Joule heating proxy
    y0 : float
        Initial position
    i0 : float
        Initial current
    m, g : float
        Physical constants
    
    Returns
    -------
    y_sim : jnp.ndarray
        Simulated position trajectory [N]
    i_sim : jnp.ndarray
        Simulated current trajectory [N]
    """
    K0, A, R0, alpha = params[0], params[1], params[2], params[3]

    state0 = jnp.array([y0, 0.0, i0])

    def step_fn(state, inputs):
        y, y_dot, i = state[0], state[1], state[2]
        u, dt, Tr = inputs[0], inputs[1], inputs[2]

        y = jnp.clip(y, 1e-6, 0.05)

        L = K0 / (1.0 + y / A)
        dL_dy_val = -K0 / (A * (1.0 + y / A) ** 2)

        R_t = R0 * (1.0 + alpha * Tr)
        R_t = jnp.clip(R_t, 0.5, 30.0)

        dt_sub = dt / internal_steps

        def substep(k, carry):
            st, cost = carry
            y_s, ydot_s, i_s = st[0], st[1], st[2]

            y_s = jnp.clip(y_s, 1e-6, 0.05)
            L_s = K0 / (1.0 + y_s / A)
            dLdy_s = -K0 / (A * (1.0 + y_s / A) ** 2)

            y_ddot_s = (0.5 / m) * dLdy_s * i_s ** 2 + g
            di_dt_s = jnp.where(
                L_s > 1e-9,
                (u - dLdy_s * ydot_s * i_s - R_t * i_s) / L_s,
                0.0,
            )
            di_dt_s = jnp.clip(di_dt_s, -500.0, 500.0)

            cost = cost + (yddot_weight * (y_ddot_s ** 2) + didt_weight * (di_dt_s ** 2)) * dt_sub

            y_next = y_s + ydot_s * dt_sub
            ydot_next = jnp.clip(ydot_s + y_ddot_s * dt_sub, -10.0, 10.0)
            i_next = jnp.clip(i_s + di_dt_s * dt_sub, -3.0, 3.0)
            return (jnp.array([y_next, ydot_next, i_next]), cost)

        init_carry = (state, 0.0)
        new_state, cost_step = jax.lax.cond(
            dt <= 0.0,
            lambda _: init_carry,
            lambda _: jax.lax.fori_loop(0, internal_steps, substep, init_carry),
            operand=None,
        )

        out = jnp.array([new_state[0], new_state[2], cost_step])
        return new_state, out

    inputs = jnp.stack([u_data, dt_steps, t_rise], axis=1)
    _, out_traj = scan(step_fn, state0, inputs)

    y_traj = out_traj[:, 0]
    i_traj = out_traj[:, 1]
    cost_traj = out_traj[:, 2]
    return y_traj, i_traj, cost_traj


@jit
def fitness_single(params: jnp.ndarray,
                   t_data: jnp.ndarray,
                   u_data: jnp.ndarray,
                   i_data: jnp.ndarray,
                   y_real: jnp.ndarray,
                   y0: float,
                   m: float,
                   g: float,
                   dt_steps: jnp.ndarray,
                   t_rise: jnp.ndarray,
                   internal_steps: int,
                   i_soft_limit: float,
                   i_penalty_weight: float,
                   action_weight: float,
                   yddot_weight: float,
                   didt_weight: float) -> float:
    """
    Compute fitness (MSE) for a single parameter set.
    Lower is better.
    """
    i0 = i_data[0]
    y_sim, i_sim, cost_traj = simulate_single(
        params,
        u_data,
        dt_steps,
        t_rise,
        y0,
        i0,
        m,
        g,
        internal_steps,
        yddot_weight,
        didt_weight,
    )

    mse_y = jnp.mean((y_sim - y_real) ** 2)
    mse_i = jnp.mean((i_sim - i_data) ** 2)
    mse_total = 0.8 * mse_y + 0.2 * mse_i

    over_i = jnp.maximum(jnp.abs(i_sim) - i_soft_limit, 0.0)
    penalty_i = i_penalty_weight * jnp.mean(over_i ** 2)

    action_penalty = action_weight * jnp.mean(cost_traj)

    bad = jnp.any(jnp.isnan(y_sim)) | jnp.any(jnp.isnan(i_sim))
    return jnp.where(bad, 1e10, mse_total + penalty_i + action_penalty)


def create_vectorized_fitness(t_data: jnp.ndarray,
                               u_data: jnp.ndarray,
                               i_data: jnp.ndarray,
                               y_real: jnp.ndarray,
                               y0: float,
                               m: float = 0.009,
                               g: float = 9.81,
                               dt: float = 0.01,
                               internal_steps: int = 10,
                               i_soft_limit: float = 2.0,
                               i_penalty_weight: float = 10.0,
                               action_weight: float = 0.0,
                               yddot_weight: float = 1.0,
                               didt_weight: float = 1e-4):
    """
    Create a vectorized fitness function that evaluates the ENTIRE population
    in a single GPU call using vmap.
    
    Returns
    -------
    vectorized_fitness_fn : callable
        Function that takes population [pop_size, 4] and returns fitness [pop_size]
    """
    
    del dt

    dt_steps = jnp.diff(t_data, prepend=t_data[0])

    i_sq = i_data ** 2
    heat_inc = 0.5 * (i_sq + jnp.concatenate([i_sq[:1], i_sq[:-1]], axis=0)) * dt_steps
    cumulative_heat = jnp.cumsum(heat_inc)
    heat_final = jnp.maximum(cumulative_heat[-1], 1e-12)
    t_rise = 20.0 * cumulative_heat / heat_final

    @jit
    def fitness_fn(params):
        return fitness_single(
            params,
            t_data,
            u_data,
            i_data,
            y_real,
            y0,
            m,
            g,
            dt_steps,
            t_rise,
            internal_steps,
            i_soft_limit,
            i_penalty_weight,
            action_weight,
            yddot_weight,
            didt_weight,
        )
    
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
