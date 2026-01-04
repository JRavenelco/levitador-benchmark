"""
Shrimp Optimization Algorithm
==============================

Implementation of the Shrimp Optimization Algorithm inspired by the
social behavior of mantis shrimps.
"""

import numpy as np
import math
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class ShrimpOptimizer(BaseOptimizer):
    """
    Shrimp Optimization Algorithm (SOA).
    
    Inspired by the social behavior of mantis shrimps. The algorithm
    combines exploration (random movement) with exploitation (following
    the leader).
    
    Algorithm Phases:
    1. Exploration: Lévy flight-based movement
    2. Attack: Movement towards prey (best solution)
    3. Defense: Movement away from threats
    4. Social: Following the group
    
    Algorithm Steps:
    1. Initialize shrimp population randomly
    2. For each iteration:
       a. Update exploration/exploitation weight
       b. For each shrimp:
          - Exploration phase: Lévy flight movement
          - Exploitation phase: Move towards best
       c. Evaluate and update best
    3. Return best solution
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Population size (default: 30)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import ShrimpOptimizer
    >>> problema = LevitadorBenchmark()
    >>> optimizer = ShrimpOptimizer(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Shrimp Optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size
        max_iter : int
            Maximum number of iterations
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Shrimp Optimization Algorithm.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness)
        """
        # Initialize population
        shrimps = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(s) for s in shrimps])
        
        best_idx = np.argmin(fitness)
        best_solution = shrimps[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # Exploration/exploitation weight (decreases from 1 to 0)
            w = 1 - t / self.max_iter
            
            # Prepare new positions array
            new_shrimps = np.zeros_like(shrimps)
            
            # Generate random numbers for decision
            r = self._rng.random((self.pop_size, 1))
            
            # Exploration indices (Lévy flight)
            explore_mask = (r < 0.5).flatten()
            n_explore = np.sum(explore_mask)
            
            # Exploitation indices (Move towards best)
            exploit_mask = ~explore_mask
            n_exploit = np.sum(exploit_mask)
            
            if n_explore > 0:
                levy = self._levy_flight(n_explore)
                new_shrimps[explore_mask] = shrimps[explore_mask] + w * levy * (self.ub - self.lb)
            
            if n_exploit > 0:
                r1 = self._rng.random((n_exploit, 1))
                r2 = self._rng.random((n_exploit, 1))
                
                # Random shrimp indices for second term
                rand_indices = self._rng.integers(0, self.pop_size, n_exploit)
                random_shrimps = shrimps[rand_indices]
                
                term1 = r1 * (best_solution - shrimps[exploit_mask])
                term2 = r2 * (random_shrimps - shrimps[exploit_mask])
                
                new_shrimps[exploit_mask] = shrimps[exploit_mask] + term1 + term2
            
            # Clip bounds
            new_shrimps = np.clip(new_shrimps, self.lb, self.ub)
            
            # Evaluate batch
            if hasattr(self.problema, 'evaluate_batch'):
                new_fitness = self.problema.evaluate_batch(new_shrimps)
                self.evaluations += self.pop_size
            else:
                new_fitness = np.array([self._evaluate(s) for s in new_shrimps])
            
            # Update best and population
            improved = new_fitness < fitness
            shrimps[improved] = new_shrimps[improved]
            fitness[improved] = new_fitness[improved]
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = shrimps[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
    
    def _levy_flight(self, n_steps=1, beta=1.5):
        """
        Generate Lévy flight steps.
        
        Uses the Mantegna method to generate Lévy-distributed random steps.
        
        Parameters
        ----------
        n_steps : int
            Number of steps to generate
        beta : float
            Lévy exponent (default: 1.5)
        
        Returns
        -------
        np.ndarray
            Lévy flight step vectors of shape (n_steps, dim)
        """
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        if n_steps == 1:
            u = self._rng.normal(0, sigma, self.dim)
            v = self._rng.normal(0, 1, self.dim)
            return u / (np.abs(v)**(1 / beta))
        else:
            u = self._rng.normal(0, sigma, (n_steps, self.dim))
            v = self._rng.normal(0, 1, (n_steps, self.dim))
            return u / (np.abs(v)**(1 / beta))
