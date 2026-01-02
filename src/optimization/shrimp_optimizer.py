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
            
            for i in range(self.pop_size):
                r = self._rng.random()
                
                if r < 0.5:
                    # Exploration phase: Lévy flight
                    levy = self._levy_flight()
                    new_pos = shrimps[i] + w * levy * (self.ub - self.lb)
                else:
                    # Exploitation phase: Move towards best
                    r1, r2 = self._rng.random(2)
                    new_pos = shrimps[i] + r1 * (best_solution - shrimps[i]) + r2 * (shrimps[self._rng.integers(self.pop_size)] - shrimps[i])
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    shrimps[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_solution = new_pos.copy()
                        best_fitness = new_fitness
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
    
    def _levy_flight(self, beta=1.5):
        """
        Generate a Lévy flight step.
        
        Uses the Mantegna method to generate Lévy-distributed random steps.
        Fixed to use math.gamma instead of deprecated np.math.gamma.
        
        Parameters
        ----------
        beta : float
            Lévy exponent (default: 1.5)
        
        Returns
        -------
        np.ndarray
            Lévy flight step vector
        """
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = self._rng.normal(0, sigma, self.dim)
        v = self._rng.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1 / beta))
