"""
Random Search Optimizer
======================

Baseline random search algorithm.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Random Search optimizer.
    
    Baseline algorithm that generates random solutions uniformly
    distributed in the search space.
    """
    
    def __init__(self, problema, n_iterations: int = 1000,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Random Search.
        
        Args:
            problema: Instance of LevitadorBenchmark
            n_iterations: Number of random samples to evaluate
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.n_iterations = n_iterations
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute random search optimization."""
        best_error = float('inf')
        best_solution = None
        
        for i in range(self.n_iterations):
            solution = self._rng.uniform(self.lb, self.ub)
            error = self._evaluate(solution)
            
            if error < best_error:
                best_error = error
                best_solution = solution.copy()
                if self.verbose:
                    print(f"  Iter {i+1}: New best = {error:.6e}")
            
            self.history.append(best_error)
        
        return best_solution, best_error
