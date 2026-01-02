"""
Grey Wolf Optimizer
===================

Implementation of the Grey Wolf Optimizer algorithm inspired by the
leadership hierarchy and hunting behavior of grey wolves.

Reference: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO).
    
    Inspired by the social hierarchy and hunting behavior of grey wolves.
    Wolves are divided into four categories:
    - Alpha (α): Best solution
    - Beta (β): Second best solution
    - Delta (δ): Third best solution
    - Omega (ω): Remaining solutions
    
    Algorithm Steps:
    1. Initialize wolf population randomly
    2. Identify Alpha, Beta, and Delta wolves (top 3 solutions)
    3. For each iteration:
       a. Update coefficient 'a' (linearly decreases from 2 to 0)
       b. Update position of each wolf based on Alpha, Beta, Delta
       c. Clip positions to bounds
       d. Evaluate and update hierarchy
    4. Return Alpha (best solution)
    
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
    >>> from src.optimization import GreyWolfOptimizer
    >>> problema = LevitadorBenchmark()
    >>> optimizer = GreyWolfOptimizer(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None, 
                 verbose: bool = True):
        """
        Initialize Grey Wolf Optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Number of wolves (population size)
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
        Execute Grey Wolf Optimizer.
        
        Returns
        -------
        tuple
            (alpha, alpha_score) - Best solution and its fitness
        """
        # Initialize wolf population
        wolves = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(w) for w in wolves])
        
        # Sort and get Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        alpha_score = fitness[sorted_idx[0]]
        beta_score = fitness[sorted_idx[1]]
        delta_score = fitness[sorted_idx[2]]
        
        # Main loop
        for t in range(self.max_iter):
            # Update 'a' linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.pop_size):
                for d in range(self.dim):
                    # Random coefficients for Alpha
                    r1, r2 = self._rng.random(), self._rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    # Random coefficients for Beta
                    r1, r2 = self._rng.random(), self._rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    # Random coefficients for Delta
                    r1, r2 = self._rng.random(), self._rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    # Distance to Alpha, Beta, Delta
                    D_alpha = abs(C1 * alpha[d] - wolves[i, d])
                    D_beta = abs(C2 * beta[d] - wolves[i, d])
                    D_delta = abs(C3 * delta[d] - wolves[i, d])
                    
                    # Position candidates
                    X1 = alpha[d] - A1 * D_alpha
                    X2 = beta[d] - A2 * D_beta
                    X3 = delta[d] - A3 * D_delta
                    
                    # New position (average)
                    wolves[i, d] = (X1 + X2 + X3) / 3
                
                # Enforce bounds
                wolves[i] = np.clip(wolves[i], self.lb, self.ub)
            
            # Evaluate and update hierarchy
            fitness = np.array([self._evaluate(w) for w in wolves])
            
            for i in range(self.pop_size):
                if fitness[i] < alpha_score:
                    delta, delta_score = beta.copy(), beta_score
                    beta, beta_score = alpha.copy(), alpha_score
                    alpha, alpha_score = wolves[i].copy(), fitness[i]
                elif fitness[i] < beta_score:
                    delta, delta_score = beta.copy(), beta_score
                    beta, beta_score = wolves[i].copy(), fitness[i]
                elif fitness[i] < delta_score:
                    delta, delta_score = wolves[i].copy(), fitness[i]
            
            self.history.append(alpha_score)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Alpha = {alpha_score:.6e}")
        
        return alpha, alpha_score
