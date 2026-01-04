"""
Random Search Optimizer
========================

Baseline optimization algorithm that generates random solutions
uniformly distributed in the search space.

This serves as a baseline to compare against more sophisticated
optimization algorithms.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class RandomSearch(BaseOptimizer):
    """
    Random Search Optimization Algorithm.
    
    A simple baseline algorithm that generates random solutions uniformly
    distributed within the search space bounds. Each iteration generates
    a new random solution and keeps track of the best solution found.
    
    This algorithm is useful as:
    - A baseline for comparing more sophisticated algorithms
    - A sanity check for the optimization problem
    - A simple method for problems where gradient information is unavailable
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem instance
    n_iterations : int, optional
        Number of random evaluations (default: 1000)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import RandomSearch
    >>> problema = LevitadorBenchmark()
    >>> optimizer = RandomSearch(problema, n_iterations=500, random_seed=42)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, n_iterations: int = 1000,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Random Search optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        n_iterations : int
            Number of random samples to evaluate
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.n_iterations = n_iterations
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Random Search optimization.
        
        Generates random solutions uniformly in the search space and
        keeps track of the best solution found.
        
        Returns
        -------
        tuple
            (best_solution, best_error) where best_solution is np.ndarray
            and best_error is the corresponding fitness value
        """
        best_error = float('inf')
        best_solution = None
        
        # Batch size for parallel evaluation (avoid memory issues with huge iterations)
        batch_size = 1000
        
        remaining_iters = self.n_iterations
        
        while remaining_iters > 0:
            current_batch = min(batch_size, remaining_iters)
            
            # Generate batch of random solutions
            solutions = self._rng.uniform(self.lb, self.ub, (current_batch, self.dim))
            
            if hasattr(self.problema, 'evaluate_batch'):
                fitnesses = self.problema.evaluate_batch(solutions)
                self.evaluations += current_batch
            else:
                fitnesses = np.array([self._evaluate(s) for s in solutions])
            
            # Find best in batch
            min_idx = np.argmin(fitnesses)
            min_fitness = fitnesses[min_idx]
            
            if min_fitness < best_error:
                best_error = min_fitness
                best_solution = solutions[min_idx].copy()
                if self.verbose:
                    print(f"  Iter {self.n_iterations - remaining_iters + min_idx + 1}: Nuevo mejor = {best_error:.6e}")
            
            # Update history (fill for the whole batch for consistency in plot length)
            # This is an approximation since we processed a batch at once
            for _ in range(current_batch):
                self.history.append(best_error)
                
            remaining_iters -= current_batch
        
        return best_solution, best_error
