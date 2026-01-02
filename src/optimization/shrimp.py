"""
Shrimp Optimization Algorithm
=============================

Shrimp Optimization Algorithm inspired by the social behavior of mantis shrimp.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class ShrimpOptimizer(BaseOptimizer):
    """
    Shrimp Optimization Algorithm (SOA).
    
    Inspired by the social behavior of mantis shrimp. Combines exploration
    (random movement) with exploitation (following the leader).
    """
    
    def __init__(self, problema, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Shrimp Optimizer.
        
        Args:
            problema: Instance of LevitadorBenchmark
            pop_size: Population size (number of shrimp)
            max_iter: Maximum number of iterations
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute Shrimp Optimization Algorithm."""
        # Initialize shrimp population
        shrimps = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(s) for s in shrimps])
        
        best_idx = np.argmin(fitness)
        best_solution = shrimps[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # Exploration/exploitation factor (decreases from 1 to 0)
            w = 1 - t / self.max_iter
            
            for i in range(self.pop_size):
                r = self._rng.random()
                
                if r < 0.5:
                    # Exploration phase: Lévy flight
                    levy = self._levy_flight()
                    new_pos = shrimps[i] + w * levy * (self.ub - self.lb)
                else:
                    # Exploitation phase: move towards best
                    r1, r2 = self._rng.random(2)
                    rand_shrimp = shrimps[self._rng.integers(self.pop_size)]
                    new_pos = (shrimps[i] + r1 * (best_solution - shrimps[i]) + 
                              r2 * (rand_shrimp - shrimps[i]))
                
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
                print(f"  Iter {t:3d}: Best = {best_fitness:.6e}")
        
        return best_solution, best_fitness
    
    def _levy_flight(self, beta=1.5):
        """Generate Lévy flight step."""
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = self._rng.normal(0, sigma, self.dim)
        v = self._rng.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1 / beta))
